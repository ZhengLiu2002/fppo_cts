# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .ppo import PPO


class PPOLagrange(PPO):
    """PPO with per-constraint Lagrange multipliers for CMDPs."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        cost_gamma=None,
        cost_lam=None,
        value_loss_coef=1.0,
        cost_value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        normalize_cost_advantage: bool = False,
        cost_limit=0.0,
        lagrange_lr=1e-2,
        lagrange_max=100.0,
        lagrangian_multiplier_init: float = 0.0,
        lagrange_optimizer: str = "Adam",
        cost_viol_loss_coef: float = 0.0,
        k_value: float = 1.0,
        k_growth: float = 1.0,
        k_max: float = 1.0,
        k_decay: float = 1.0,
        k_min: float = 0.0,
        k_violation_threshold: float = 0.02,
        constraint_limits: list[float] | tuple[float, ...] | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            cost_gamma=cost_gamma,
            cost_lam=cost_lam,
            value_loss_coef=value_loss_coef,
            cost_value_loss_coef=cost_value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            normalize_cost_advantage=normalize_cost_advantage,
            cost_limit=cost_limit,
            cost_viol_loss_coef=cost_viol_loss_coef,
            k_value=k_value,
            k_growth=k_growth,
            k_max=k_max,
            k_decay=k_decay,
            k_min=k_min,
            k_violation_threshold=k_violation_threshold,
            constraint_limits=constraint_limits,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        if not hasattr(optim, lagrange_optimizer):
            raise AttributeError(f"Optimizer={lagrange_optimizer} not found in torch.optim.")
        self.lagrange_lr = float(lagrange_lr)
        self.lagrange_max = float(lagrange_max)
        self._lagrange_optimizer_cls = getattr(optim, lagrange_optimizer)
        self._lagrangian_multiplier_init = max(float(lagrangian_multiplier_init), 0.0)
        self._lagrange_multiplier: torch.nn.Parameter | None = None
        self._lagrange_optimizer: torch.optim.Optimizer | None = None
        self._ensure_lagrange(self._initial_constraint_count())
        self.train_metrics: dict[str, float] = {}

    def _initial_constraint_count(self) -> int:
        if self.constraint_limits is None:
            return 1
        return max(int(self.constraint_limits.numel()), 1)

    def _ensure_lagrange(self, num_constraints: int) -> None:
        num_constraints = max(int(num_constraints), 1)
        if self._lagrange_multiplier is not None and self._lagrange_multiplier.numel() == num_constraints:
            return
        init = torch.full(
            (num_constraints,),
            self._lagrangian_multiplier_init,
            device=self.device,
            dtype=torch.float32,
        )
        if self._lagrange_multiplier is not None:
            old = self._lagrange_multiplier.detach().to(device=self.device, dtype=torch.float32)
            rows = min(old.numel(), num_constraints)
            init[:rows] = old[:rows]
        self._lagrange_multiplier = torch.nn.Parameter(init, requires_grad=True)
        self._lagrange_optimizer = self._lagrange_optimizer_cls(
            [self._lagrange_multiplier],
            lr=self.lagrange_lr,
        )

    @property
    def lagrange_multiplier(self) -> torch.Tensor:
        if self._lagrange_multiplier is None:
            self._ensure_lagrange(self._initial_constraint_count())
        return self._lagrange_multiplier.detach()

    def _estimate_rollout_costs(self) -> torch.Tensor:
        if self.storage.cost_term_rewards is not None:
            rollout_costs = self._sanitize_tensor(
                self.storage.cost_term_rewards,
                nan=0.0,
                posinf=1.0e4,
                neginf=0.0,
                clamp=1.0e4,
            )
            mean_costs = rollout_costs.sum(dim=0).mean(dim=0)
            return self._all_reduce_mean(mean_costs.reshape(-1))
        rollout_costs = self._sanitize_tensor(
            self.storage.cost_rewards,
            nan=0.0,
            posinf=1.0e4,
            neginf=0.0,
            clamp=1.0e4,
        )
        mean_cost = rollout_costs.sum(dim=0).mean().reshape(1)
        return self._all_reduce_mean(mean_cost)

    def _update_lagrange_multiplier(self, rollout_costs: torch.Tensor) -> None:
        rollout_costs = rollout_costs.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        self._ensure_lagrange(rollout_costs.numel())
        d_limits = self._resolve_constraint_limits(rollout_costs.numel(), device=self.device).to(
            dtype=rollout_costs.dtype
        )
        self._lagrange_optimizer.zero_grad()
        lambda_loss = -torch.sum(self._lagrange_multiplier * (rollout_costs - d_limits.detach()))
        lambda_loss.backward()
        self._lagrange_optimizer.step()
        self._lagrange_multiplier.data.clamp_(0.0, self.lagrange_max)

    def _combined_advantages(
        self,
        reward_advantages: torch.Tensor,
        cost_advantages: torch.Tensor,
    ) -> torch.Tensor:
        if cost_advantages.ndim == 1:
            cost_advantages = cost_advantages.unsqueeze(-1)
        self._ensure_lagrange(cost_advantages.shape[1])
        penalty = self.lagrange_multiplier.to(device=cost_advantages.device, dtype=cost_advantages.dtype)
        weighted_cost_adv = self._constraint_weighted_advantages(cost_advantages, penalty)
        combined = (reward_advantages - weighted_cost_adv) / (1.0 + penalty.sum())
        return self._sanitize_tensor(
            combined,
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
            clamp=1.0e3,
        )

    def state_dict(self) -> dict:
        lagrange_state = None
        if self._lagrange_multiplier is not None:
            lagrange_state = {
                "lagrangian_multiplier": self._lagrange_multiplier.detach().clone(),
                "lambda_optimizer": (
                    self._lagrange_optimizer.state_dict() if self._lagrange_optimizer is not None else None
                ),
            }
        return {
            "lagrange": lagrange_state,
            "learning_rate": self.learning_rate,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        lagrange_state = state_dict.get("lagrange")
        if lagrange_state is not None:
            multiplier = lagrange_state.get("lagrangian_multiplier")
            if multiplier is not None:
                self._ensure_lagrange(int(multiplier.numel()))
                self._lagrange_multiplier.data.copy_(
                    multiplier.to(device=self.device, dtype=self._lagrange_multiplier.dtype)
                )
            optimizer_state = lagrange_state.get("lambda_optimizer")
            if optimizer_state is not None and self._lagrange_optimizer is not None:
                self._lagrange_optimizer.load_state_dict(optimizer_state)
        learning_rate = state_dict.get("learning_rate")
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def update(self):  # noqa: C901
        self._update_lagrange_multiplier(self._estimate_rollout_costs())

        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_viol_loss = 0.0
        mean_cost_return = 0.0
        mean_cost_violation = 0.0
        mean_cost_margin = 0.0
        mean_current_max_violation = 0.0
        mean_kl = 0.0
        skipped_updates = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        num_updates = self.num_learning_epochs * self.num_mini_batches

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            cost_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            *extra_batch,
        ) in generator:
            cost_term_returns_batch = extra_batch[0] if len(extra_batch) > 0 else None
            cost_term_advantages_batch = extra_batch[1] if len(extra_batch) > 1 else None
            cost_term_values_batch = extra_batch[3] if len(extra_batch) > 3 else None
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )
                    if self.normalize_cost_advantage:
                        cost_advantages_batch = (
                            cost_advantages_batch - cost_advantages_batch.mean()
                        ) / (cost_advantages_batch.std() + 1e-8)
                        if cost_term_advantages_batch is not None:
                            mean = cost_term_advantages_batch.mean(dim=0, keepdim=True)
                            std = cost_term_advantages_batch.std(dim=0, keepdim=True)
                            cost_term_advantages_batch = (cost_term_advantages_batch - mean) / (
                                std + 1e-8
                            )

            returns_batch = self._sanitize_tensor(
                returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_returns_batch = self._sanitize_tensor(
                cost_returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            target_values_batch = self._sanitize_tensor(
                target_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_values_batch = self._sanitize_tensor(
                cost_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_terms_ret, cost_terms_adv, cost_terms_val = self._prepare_cost_term_batches(
                cost_returns_batch=cost_returns_batch,
                cost_advantages_batch=cost_advantages_batch,
                cost_term_returns_batch=cost_term_returns_batch,
                cost_term_advantages_batch=cost_term_advantages_batch,
                cost_term_values_batch=cost_term_values_batch,
            )
            constraint_stats = self._constraint_batch_stats(cost_terms_ret)
            batch_cost_return = constraint_stats["aggregate_cost_return"]
            batch_cost_violation = constraint_stats["batch_cost_violation"]
            batch_cost_margin = constraint_stats["min_cost_margin"]
            current_max_violation = constraint_stats["max_c_hat"]

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            cost_value_batch = self.policy.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[2]
            )
            cost_value_batch = self._sanitize_tensor(
                cost_value_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            if cost_value_batch.ndim == 1:
                cost_value_batch = cost_value_batch.unsqueeze(-1)
            elif cost_value_batch.ndim > 2:
                cost_value_batch = cost_value_batch.view(cost_value_batch.shape[0], -1)
            pred_cost_terms = self._match_cost_heads(cost_value_batch, cost_terms_ret.shape[1])
            old_cost_terms = self._match_cost_heads(cost_terms_val, cost_terms_ret.shape[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            with torch.inference_mode():
                kl = self._safe_kl(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)
                kl_mean = self._all_reduce_mean(torch.mean(kl))

            if self.desired_kl is not None and self.schedule == "adaptive":
                if self.gpu_global_rank == 0:
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            if torch.isfinite(kl_mean) and kl_mean.item() > self._kl_hard_limit():
                if self.gpu_global_rank == 0:
                    self.learning_rate = max(1e-5, self.learning_rate / 2.0)
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
                mean_kl += kl_mean.item()
                mean_cost_return += batch_cost_return.item()
                mean_cost_violation += batch_cost_violation.item()
                mean_cost_margin += batch_cost_margin.item()
                mean_current_max_violation += current_max_violation.item()
                skipped_updates += 1
                continue

            ratio = self._safe_ratio(actions_log_prob_batch, old_actions_log_prob_batch)
            combined_advantages = self._combined_advantages(
                torch.squeeze(advantages_batch),
                cost_terms_adv,
            )
            surrogate = -combined_advantages * ratio
            surrogate_clipped = -combined_advantages * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            ratio_cost = torch.clamp(
                ratio,
                1.0 - self.cost_ratio_clip,
                1.0 + self.cost_ratio_clip,
            )
            cost_surrogates = self._constraint_surrogate_terms(cost_terms_adv, ratio_cost)
            viol_loss = self._positive_cost_penalty_per_constraint(cost_surrogates, constraint_stats["c_hat"])
            surrogate_loss = self._sanitize_tensor(
                surrogate_loss, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
            )
            viol_loss = self._sanitize_tensor(
                viol_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            value_loss = self._sanitize_tensor(
                value_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            if self.use_clipped_value_loss:
                cost_value_clipped = old_cost_terms + (
                    pred_cost_terms - old_cost_terms
                ).clamp(-self.clip_param, self.clip_param)
                cost_value_losses = (pred_cost_terms - cost_terms_ret).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_terms_ret).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_terms_ret - pred_cost_terms).pow(2).mean()
            cost_value_loss = self._sanitize_tensor(
                cost_value_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            loss = (
                surrogate_loss
                + viol_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            loss = self._sanitize_tensor(loss, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6)
            if not torch.isfinite(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._step_constraint_scale(batch_cost_violation.item())

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_viol_loss += viol_loss.item()
            mean_cost_return += batch_cost_return.item()
            mean_cost_violation += batch_cost_violation.item()
            mean_cost_margin += batch_cost_margin.item()
            mean_current_max_violation += current_max_violation.item()
            mean_kl += kl_mean.item()

        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_viol_loss /= num_updates
        mean_cost_return /= num_updates
        mean_cost_violation /= num_updates
        mean_cost_margin /= num_updates
        mean_current_max_violation /= num_updates
        mean_kl /= num_updates
        kl_skip_rate = skipped_updates / num_updates

        self.storage.clear()
        self.train_metrics = {
            "mean_cost_return": mean_cost_return,
            "cost_limit_margin": mean_cost_margin,
            "cost_violation_rate": mean_cost_violation,
            "viol_loss": mean_viol_loss,
            "k_value": self.k_value,
            "kl": mean_kl,
            "kl_skip_rate": kl_skip_rate,
            "lagrange_multiplier_mean": float(self.lagrange_multiplier.mean().item()),
            "lagrange_multiplier_max": float(self.lagrange_multiplier.max().item()),
            "current_max_violation": mean_current_max_violation,
        }

        return {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "viol": mean_viol_loss,
        }
