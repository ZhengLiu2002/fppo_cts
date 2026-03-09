# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .ppo import PPO


class FPPO(PPO):
    """First-order constrained PPO with a PPO predictor and projection corrector."""

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
        step_size=1e-3,
        cost_limit=0.0,
        delta_safe=0.01,
        epsilon_safe=0.0,
        delta_kl: float | None = None,
        backtrack_coeff=0.5,
        max_corrections=10,
        max_backtracks: int | None = None,
        projection_eps=1e-8,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        use_clipped_surrogate: bool = True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        normalize_cost_advantage: bool = False,
        cost_viol_loss_coef: float = 0.0,
        k_value: float = 1.0,
        k_growth: float = 1.0,
        k_max: float = 1.0,
        k_decay: float = 1.0,
        k_min: float = 0.0,
        k_violation_threshold: float = 0.02,
        use_preconditioner: bool = True,
        preconditioner_beta: float = 0.999,
        preconditioner_eps: float = 1.0e-8,
        use_momentum: bool = True,
        momentum_beta: float = 0.9,
        slack_penalty: float = 1.0,
        active_set_threshold: float = 0.05,
        softproj_max_iters: int = 40,
        softproj_tol: float = 1.0e-6,
        constraint_limits: list[float] | tuple[float, ...] | None = None,
        constraint_limits_start: list[float] | tuple[float, ...] | None = None,
        constraint_limits_final: list[float] | tuple[float, ...] | None = None,
        adaptive_constraint_curriculum: bool = False,
        constraint_names: list[str] | tuple[str, ...] | None = None,
        constraint_curriculum_names: list[str] | tuple[str, ...] | None = None,
        constraint_curriculum_ema_decay: float = 0.95,
        constraint_curriculum_check_interval: int = 20,
        constraint_curriculum_alpha: float = 0.8,
        constraint_curriculum_shrink: float = 0.97,
        feasible_first: bool = True,
        feasible_first_coef: float = 1.0,
        projection_scale_clip: float = 1.0e3,
        feasible_cost_margin: float = 1.0e-3,
        infeasible_improve_ratio: float = 0.01,
        infeasible_improve_abs: float = 1.0e-3,
        min_step_size: float = 1.0e-7,
        relax_cost_margin: float = 0.2,
        step_size_adaptive: bool = True,
        step_size_up: float = 1.02,
        step_size_down: float = 0.7,
        step_size_min: float = 5.0e-5,
        step_size_max: float = 2.0e-3,
        target_accept_rate: float = 0.75,
        step_size_cost_margin: float = 0.2,
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
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.step_size = float(step_size)
        self.delta_safe = float(delta_safe) if delta_safe is not None else None
        self.epsilon_safe = float(epsilon_safe)
        self.delta_kl = float(delta_kl) if delta_kl is not None else None
        self.backtrack_coeff = float(backtrack_coeff)
        self.max_corrections = int(max_backtracks if max_backtracks is not None else max_corrections)
        self.max_backtracks = self.max_corrections
        self.projection_eps = float(projection_eps)
        self.use_clipped_surrogate = bool(use_clipped_surrogate)
        self.use_preconditioner = bool(use_preconditioner)
        self.preconditioner_beta = float(preconditioner_beta)
        self.preconditioner_eps = float(preconditioner_eps)
        self.use_momentum = bool(use_momentum)
        self.momentum_beta = float(momentum_beta)
        self.slack_penalty = max(float(slack_penalty), 1.0e-8)
        self.active_set_threshold = float(active_set_threshold)
        self.softproj_max_iters = max(int(softproj_max_iters), 1)
        self.softproj_tol = max(float(softproj_tol), 1.0e-12)
        self.constraint_limits = (
            torch.as_tensor(constraint_limits, dtype=torch.float32)
            if constraint_limits is not None
            else None
        )
        self.constraint_limits_final = (
            torch.as_tensor(
                constraint_limits_final if constraint_limits_final is not None else constraint_limits,
                dtype=torch.float32,
            )
            if constraint_limits_final is not None or constraint_limits is not None
            else None
        )
        self.constraint_limits_start = (
            torch.as_tensor(
                constraint_limits_start
                if constraint_limits_start is not None
                else self.constraint_limits_final,
                dtype=torch.float32,
            )
            if constraint_limits_start is not None or self.constraint_limits_final is not None
            else None
        )
        self.constraint_limits_current = (
            self.constraint_limits_start.clone()
            if self.constraint_limits_start is not None
            else (self.constraint_limits_final.clone() if self.constraint_limits_final is not None else None)
        )
        if self.constraint_limits is None and self.constraint_limits_final is not None:
            self.constraint_limits = self.constraint_limits_final.clone()
        self.adaptive_constraint_curriculum = bool(adaptive_constraint_curriculum)
        self.constraint_names = (
            [str(name) for name in constraint_names] if constraint_names is not None else None
        )
        self.constraint_curriculum_names = [
            str(name) for name in (constraint_curriculum_names or [])
        ]
        self.constraint_curriculum_ema_decay = min(
            max(float(constraint_curriculum_ema_decay), 0.0), 0.9999
        )
        self.constraint_curriculum_check_interval = max(int(constraint_curriculum_check_interval), 1)
        self.constraint_curriculum_alpha = float(constraint_curriculum_alpha)
        self.constraint_curriculum_shrink = min(max(float(constraint_curriculum_shrink), 0.0), 1.0)
        self._constraint_curriculum_ema: torch.Tensor | None = None
        self._constraint_curriculum_updates = 0
        self._constraint_curriculum_tighten_count = 0
        self.min_step_size = float(min_step_size)
        self.step_size_adaptive = bool(step_size_adaptive)
        self.step_size_up = float(step_size_up)
        self.step_size_down = float(step_size_down)
        self.step_size_min = float(step_size_min)
        self.step_size_max = float(step_size_max)
        self.target_accept_rate = float(target_accept_rate)
        self.step_size_cost_margin = float(step_size_cost_margin)
        self.feasible_first = bool(feasible_first)
        self.feasible_first_coef = float(feasible_first_coef)
        self.projection_scale_clip = float(projection_scale_clip)
        self.feasible_cost_margin = float(feasible_cost_margin)
        self.infeasible_improve_ratio = float(infeasible_improve_ratio)
        self.infeasible_improve_abs = float(infeasible_improve_abs)
        self.relax_cost_margin = float(relax_cost_margin)
        self._actor_params = self._get_actor_params()
        self._critic_params = list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters())
        self.critic_learning_rate = float(learning_rate)
        self.learning_rate = float(step_size)
        self.actor_optimizer = optim.Adam(self._actor_params, lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self._critic_params, lr=self.critic_learning_rate)
        self.optimizer = {"actor": self.actor_optimizer, "critic": self.critic_optimizer}
        self.train_metrics: dict[str, float] = {}

    def update(self):
        projection_batch = self._prepare_projection_batch()
        theta_anchor = self._actor_param_vector().detach().clone()
        a_mat, cost_surrogate_mean = self._compute_constraint_gradients(projection_batch)
        predictor_metrics = self._run_ppo_predictor()
        theta_predictor = self._actor_param_vector().detach().clone()
        correction_metrics = self._run_projection_corrector(
            projection_batch=projection_batch,
            theta_anchor=theta_anchor,
            theta_predictor=theta_predictor,
            a_mat=a_mat,
        )
        mean_value_loss, mean_cost_value_loss = self._update_value_functions()

        mean_cost_return = projection_batch["j_cost"].mean().item()
        mean_cost_margin = torch.min(projection_batch["d_tight"] - projection_batch["j_cost"]).item()
        sample_violation = (
            projection_batch["cost_terms_ret"] > projection_batch["d_limits"].unsqueeze(0)
        ).any(dim=1)
        mean_cost_violation = self._all_reduce_mean(sample_violation.float().mean()).item()
        current_max_violation = torch.max(projection_batch["j_cost"] - projection_batch["d_tight"]).item()
        viol_loss = self._positive_cost_penalty(
            cost_surrogate_mean,
            torch.max(projection_batch["j_cost"] - projection_batch["d_tight"]),
        )
        viol_loss = self._sanitize_tensor(
            viol_loss,
            nan=0.0,
            posinf=1.0e6,
            neginf=0.0,
            clamp=1.0e6,
        )
        self._step_constraint_scale(mean_cost_violation)
        self._adapt_step_size(
            accept_rate=correction_metrics["accept_rate"],
            mean_cost_margin=mean_cost_margin,
        )
        self._set_actor_learning_rate(self.step_size)
        curriculum_metrics = self._update_constraint_limit_curriculum(projection_batch["j_cost"])

        self.train_metrics = {
            "mean_cost_return": mean_cost_return,
            "cost_limit_margin": mean_cost_margin,
            "cost_violation_rate": mean_cost_violation,
            "viol_loss": viol_loss.item(),
            "k_value": self.k_value,
            "step_size": correction_metrics["effective_step_size"],
            "base_step_size": self.step_size,
            "effective_step_ratio": correction_metrics["effective_step_ratio"],
            "accept_rate": correction_metrics["accept_rate"],
            "reject_rate": 1.0 - correction_metrics["accept_rate"],
            "kl": correction_metrics["final_kl"],
            "active_constraints": correction_metrics["active_constraints"],
            "reject_kl_rate": correction_metrics["reject_kl_rate"],
            "infeasible_batch_rate": correction_metrics["infeasible_batch_rate"],
            "recovery_accept_rate": correction_metrics["recovery_accept_rate"],
            "current_max_violation": current_max_violation,
            "boundary_mode_rate": correction_metrics["boundary_mode_rate"],
            "recovery_mode_rate": correction_metrics["recovery_mode_rate"],
            "predictor_kl": predictor_metrics["mean_kl"],
            "predictor_stop_rate": predictor_metrics["stop_rate"],
        }
        self.train_metrics.update(curriculum_metrics)

        self.storage.clear()
        return {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": predictor_metrics["mean_surrogate"],
            "entropy": predictor_metrics["mean_entropy"],
            "viol": viol_loss.item(),
        }

    def _run_ppo_predictor(self) -> dict[str, float]:
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        kl_checks = 0
        updates = 0
        stop_count = 0
        stop_predictor = False
        self._set_actor_learning_rate(self.learning_rate)

        generator = self._mini_batch_generator()
        for (
            obs_batch,
            _critic_obs_batch,
            actions_batch,
            _target_values_batch,
            advantages_batch,
            _returns_batch,
            _cost_values_batch,
            _cost_returns_batch,
            _cost_advantages_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            *_extra_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            advantages_batch = self._sanitize_tensor(
                advantages_batch,
                nan=0.0,
                posinf=1.0e3,
                neginf=-1.0e3,
                clamp=1.0e3,
            )
            old_mu_batch = self._sanitize_tensor(
                old_mu_batch,
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            old_sigma_batch = self._sanitize_tensor(
                old_sigma_batch,
                nan=1.0e-6,
                posinf=1.0e2,
                neginf=1.0e-6,
                clamp=1.0e2,
            ).clamp_min(1.0e-6)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            with torch.inference_mode():
                kl_mean = self._all_reduce_mean(
                    torch.mean(self._safe_kl(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch))
                )
            mean_kl += kl_mean.item()
            kl_checks += 1

            if self.desired_kl is not None and self.schedule == "adaptive":
                if self.gpu_global_rank == 0:
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(self.step_size_min, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(self.step_size_max, self.learning_rate * 1.5)
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()
                self._set_actor_learning_rate(self.learning_rate)

            if torch.isfinite(kl_mean) and kl_mean.item() > self._kl_hard_limit():
                if self.gpu_global_rank == 0:
                    self.learning_rate = max(self.step_size_min, self.learning_rate / 2.0)
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()
                self._set_actor_learning_rate(self.learning_rate)
                stop_count += 1
                stop_predictor = True
                break

            if self.desired_kl is not None and torch.isfinite(kl_mean) and kl_mean.item() > self.desired_kl:
                stop_count += 1
                stop_predictor = True
                break

            ratio = self._safe_ratio(actions_log_prob_batch, old_actions_log_prob_batch).reshape(-1)
            adv_flat = advantages_batch.reshape(-1)
            surrogate = -adv_flat * ratio
            if self.use_clipped_surrogate:
                surrogate_clipped = -adv_flat * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            else:
                surrogate_loss = surrogate.mean()
            loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()
            loss = self._sanitize_tensor(
                loss,
                nan=0.0,
                posinf=1.0e6,
                neginf=-1.0e6,
                clamp=1.0e6,
            )
            if not torch.isfinite(loss):
                continue

            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._all_reduce_parameter_grads(self._actor_params)
            nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm)
            self.actor_optimizer.step()

            updates += 1
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        denom = max(updates, 1)
        return {
            "mean_surrogate": mean_surrogate_loss / denom,
            "mean_entropy": mean_entropy / denom,
            "mean_kl": mean_kl / max(kl_checks, 1),
            "stop_rate": float(stop_predictor or stop_count > 0),
        }

    def _update_value_functions(self) -> tuple[float, float]:
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        num_updates = self.num_learning_epochs * self.num_mini_batches
        generator = self._mini_batch_generator()

        for (
            _obs_batch,
            critic_obs_batch,
            _actions_batch,
            target_values_batch,
            _advantages_batch,
            returns_batch,
            cost_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
            _old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            hid_states_batch,
            masks_batch,
            *extra_batch,
        ) in generator:
            cost_term_returns_batch = extra_batch[0] if len(extra_batch) > 0 else None
            cost_term_advantages_batch = extra_batch[1] if len(extra_batch) > 1 else None
            cost_term_samples_batch = extra_batch[2] if len(extra_batch) > 2 else None
            cost_term_values_batch = extra_batch[3] if len(extra_batch) > 3 else None

            returns_batch = self._sanitize_tensor(
                returns_batch,
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            target_values_batch = self._sanitize_tensor(
                target_values_batch,
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            cost_values_batch = self._sanitize_tensor(
                cost_values_batch,
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            cost_returns_batch = self._sanitize_tensor(
                cost_returns_batch,
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            cost_terms_ret, _cost_terms_adv, _cost_terms_samples, cost_terms_val = self._prepare_cost_term_batches(
                cost_returns_batch=cost_returns_batch,
                cost_advantages_batch=cost_advantages_batch,
                cost_term_returns_batch=cost_term_returns_batch,
                cost_term_advantages_batch=cost_term_advantages_batch,
                cost_term_samples_batch=cost_term_samples_batch,
                cost_term_values_batch=cost_term_values_batch,
            )

            value_batch = self.policy.evaluate(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            cost_value_batch = self.policy.evaluate_cost(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[2],
            )
            cost_value_batch = self._sanitize_tensor(
                cost_value_batch,
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            if cost_value_batch.ndim == 1:
                cost_value_batch = cost_value_batch.unsqueeze(-1)
            elif cost_value_batch.ndim > 2:
                cost_value_batch = cost_value_batch.view(cost_value_batch.shape[0], -1)

            pred_cost_terms = self._match_cost_heads(cost_value_batch, cost_terms_ret.shape[1])
            old_cost_terms = self._match_cost_heads(cost_terms_val, cost_terms_ret.shape[1])

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            value_loss = self._sanitize_tensor(
                value_loss,
                nan=0.0,
                posinf=1.0e6,
                neginf=0.0,
                clamp=1.0e6,
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
                cost_value_loss,
                nan=0.0,
                posinf=1.0e6,
                neginf=0.0,
                clamp=1.0e6,
            )

            loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss
            loss = self._sanitize_tensor(
                loss,
                nan=0.0,
                posinf=1.0e6,
                neginf=0.0,
                clamp=1.0e6,
            )
            self.critic_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._all_reduce_parameter_grads(self._critic_params)
            nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)
            self.critic_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()

        return mean_value_loss / num_updates, mean_cost_value_loss / num_updates

    def _prepare_projection_batch(self) -> dict[str, torch.Tensor | None]:
        (
            obs_batch,
            _critic_obs_batch,
            actions_batch,
            _target_values_batch,
            _advantages_batch,
            _returns_batch,
            _cost_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            *extra_batch,
        ) = self._get_full_batch()

        cost_term_returns_batch = extra_batch[0] if len(extra_batch) > 0 else None
        cost_term_advantages_batch = extra_batch[1] if len(extra_batch) > 1 else None
        cost_term_samples_batch = extra_batch[2] if len(extra_batch) > 2 else None
        cost_term_values_batch = extra_batch[3] if len(extra_batch) > 3 else None

        cost_returns_batch = self._sanitize_tensor(
            cost_returns_batch,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        cost_advantages_batch = self._sanitize_tensor(
            cost_advantages_batch,
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
            clamp=1.0e3,
        )
        cost_terms_ret, cost_terms_adv, _cost_terms_samples, cost_terms_val = self._prepare_cost_term_batches(
            cost_returns_batch=cost_returns_batch,
            cost_advantages_batch=cost_advantages_batch,
            cost_term_returns_batch=cost_term_returns_batch,
            cost_term_advantages_batch=cost_term_advantages_batch,
            cost_term_samples_batch=cost_term_samples_batch,
            cost_term_values_batch=cost_term_values_batch,
        )
        cost_terms_adv = self._normalize_constraint_advantages(cost_terms_adv)
        j_cost = self._all_reduce_mean(cost_terms_ret.mean(dim=0))
        d_limits = self._resolve_constraint_limits(cost_terms_ret.shape[1], device=cost_terms_ret.device)
        d_tight = d_limits - self.epsilon_safe
        old_mu_batch = self._sanitize_tensor(
            old_mu_batch,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        old_sigma_batch = self._sanitize_tensor(
            old_sigma_batch,
            nan=1.0e-6,
            posinf=1.0e2,
            neginf=1.0e-6,
            clamp=1.0e2,
        ).clamp_min(1.0e-6)
        return {
            "obs": obs_batch,
            "actions": actions_batch,
            "old_logp": old_actions_log_prob_batch,
            "old_mu": old_mu_batch,
            "old_sigma": old_sigma_batch,
            "masks": masks_batch,
            "hid_actor": hid_states_batch[0],
            "cost_terms_ret": cost_terms_ret,
            "cost_terms_adv": cost_terms_adv,
            "cost_terms_val": cost_terms_val,
            "d_limits": d_limits,
            "d_tight": d_tight,
            "j_cost": j_cost,
        }

    def _compute_constraint_gradients(
        self,
        projection_batch: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cost_terms_adv = projection_batch["cost_terms_adv"]
        if cost_terms_adv is None or cost_terms_adv.numel() == 0:
            return torch.zeros((self._actor_param_vector().numel(), 0), device=self.device), torch.zeros(
                (),
                device=self.device,
            )

        self.policy.act(
            projection_batch["obs"],
            masks=projection_batch["masks"],
            hidden_states=projection_batch["hid_actor"],
        )
        actions_log_prob_batch = self.policy.get_actions_log_prob(projection_batch["actions"])
        ratio = self._safe_ratio(actions_log_prob_batch, projection_batch["old_logp"]).reshape(-1)
        cost_terms_adv = cost_terms_adv.reshape(-1, cost_terms_adv.shape[-1])

        cost_grad_lists: list[list[torch.Tensor]] = []
        cost_surrogates: list[torch.Tensor] = []
        num_constraints = cost_terms_adv.shape[1]
        for idx in range(num_constraints):
            cost_obj = torch.mean(ratio * cost_terms_adv[:, idx])
            cost_surrogates.append(cost_obj.detach())
            grads = torch.autograd.grad(
                cost_obj,
                self._actor_params,
                retain_graph=idx < (num_constraints - 1),
                allow_unused=True,
            )
            grads = [
                (grad if grad is not None else torch.zeros_like(param)).detach()
                for grad, param in zip(grads, self._actor_params)
            ]
            cost_grad_lists.append(grads)

        if self.is_multi_gpu:
            for grads in cost_grad_lists:
                self._all_reduce_grads(grads)

        a_mat = torch.stack([self._flatten_tensors(grads) for grads in cost_grad_lists], dim=1)
        cost_surrogate_mean = self._all_reduce_mean(torch.mean(torch.stack(cost_surrogates)))
        return a_mat, cost_surrogate_mean

    def _run_projection_corrector(
        self,
        projection_batch: dict[str, torch.Tensor | None],
        theta_anchor: torch.Tensor,
        theta_predictor: torch.Tensor,
        a_mat: torch.Tensor,
    ) -> dict[str, float]:
        b_budget = projection_batch["d_tight"] - projection_batch["j_cost"]
        theta_projected, active_count = self._project_safe_set(
            theta_predictor,
            theta_anchor,
            a_mat,
            b_budget,
        )

        kl_limit = self.delta_kl
        if kl_limit is None:
            kl_limit = self.desired_kl if self.desired_kl is not None else float("inf")

        eta = 1.0
        accepted = False
        reject_kl_count = 0
        attempted_recovery = False
        final_kl = torch.zeros((), device=self.device)

        total_checks = max(self.max_corrections, 1)
        for _ in range(total_checks):
            theta_candidate = theta_anchor + eta * (theta_projected - theta_anchor)
            self._set_actor_param_vector(theta_candidate)
            final_kl = self._evaluate_candidate(projection_batch)
            kl_ok = (not torch.isfinite(final_kl).item()) or final_kl.item() <= kl_limit
            if kl_ok:
                accepted = True
                break
            if not kl_ok:
                reject_kl_count += 1
            eta *= self.backtrack_coeff
            attempted_recovery = True

        if not accepted:
            eta = 0.0
            self._set_actor_param_vector(theta_anchor)
            final_kl = self._evaluate_candidate(projection_batch)
        else:
            self._set_actor_param_vector(theta_anchor + eta * (theta_projected - theta_anchor))

        theta_final = self._actor_param_vector().detach()
        effective_step_size = torch.norm(theta_final - theta_anchor).item()
        return {
            "accept_rate": float(accepted),
            "effective_step_ratio": eta,
            "effective_step_size": effective_step_size,
            "final_kl": final_kl.item(),
            "active_constraints": float(active_count),
            "reject_kl_rate": reject_kl_count / total_checks,
            "infeasible_batch_rate": float(not accepted),
            "recovery_accept_rate": float(accepted and attempted_recovery),
            "boundary_mode_rate": float(active_count > 0),
            "recovery_mode_rate": float(attempted_recovery),
        }

    def _get_full_batch(self):
        if self.policy.is_recurrent:
            return next(self.storage.recurrent_mini_batch_generator(1, 1))
        return next(self.storage.mini_batch_generator(1, 1))

    def _mini_batch_generator(self):
        if self.policy.is_recurrent:
            return self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        return self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    def _prepare_cost_term_batches(
        self,
        cost_returns_batch: torch.Tensor,
        cost_advantages_batch: torch.Tensor,
        cost_term_returns_batch: torch.Tensor | None,
        cost_term_advantages_batch: torch.Tensor | None,
        cost_term_samples_batch: torch.Tensor | None,
        cost_term_values_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cost_term_returns_batch is None or cost_term_advantages_batch is None:
            fallback_samples = (
                cost_term_samples_batch if cost_term_samples_batch is not None else cost_returns_batch
            )
            fallback_values = (
                cost_term_values_batch if cost_term_values_batch is not None else cost_returns_batch
            )
            if not torch.is_tensor(fallback_samples):
                fallback_samples = torch.as_tensor(fallback_samples, device=self.device)
            if not torch.is_tensor(fallback_values):
                fallback_values = torch.as_tensor(fallback_values, device=self.device)
            fallback_samples = self._sanitize_tensor(
                fallback_samples.to(self.device),
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            fallback_values = self._sanitize_tensor(
                fallback_values.to(self.device),
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            if fallback_samples.ndim == 1:
                fallback_samples = fallback_samples.unsqueeze(-1)
            if fallback_values.ndim == 1:
                fallback_values = fallback_values.unsqueeze(-1)
            return (
                cost_returns_batch.reshape(-1, 1),
                cost_advantages_batch.reshape(-1, 1),
                fallback_samples.reshape(-1, fallback_samples.shape[-1]),
                fallback_values.reshape(-1, fallback_values.shape[-1]),
            )

        if not torch.is_tensor(cost_term_returns_batch):
            cost_term_returns_batch = torch.as_tensor(cost_term_returns_batch, device=self.device)
        if not torch.is_tensor(cost_term_advantages_batch):
            cost_term_advantages_batch = torch.as_tensor(cost_term_advantages_batch, device=self.device)
        cost_term_returns_batch = self._sanitize_tensor(
            cost_term_returns_batch.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        cost_term_advantages_batch = self._sanitize_tensor(
            cost_term_advantages_batch.to(self.device),
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
            clamp=1.0e3,
        )
        if cost_term_returns_batch.ndim == 1:
            cost_term_returns_batch = cost_term_returns_batch.unsqueeze(-1)
        if cost_term_advantages_batch.ndim == 1:
            cost_term_advantages_batch = cost_term_advantages_batch.unsqueeze(-1)
        if cost_term_samples_batch is None:
            cost_term_samples_batch = cost_term_returns_batch
        if cost_term_values_batch is None:
            cost_term_values_batch = cost_term_returns_batch
        if not torch.is_tensor(cost_term_samples_batch):
            cost_term_samples_batch = torch.as_tensor(cost_term_samples_batch, device=self.device)
        if not torch.is_tensor(cost_term_values_batch):
            cost_term_values_batch = torch.as_tensor(cost_term_values_batch, device=self.device)
        cost_term_samples_batch = self._sanitize_tensor(
            cost_term_samples_batch.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        cost_term_values_batch = self._sanitize_tensor(
            cost_term_values_batch.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        if cost_term_samples_batch.ndim == 1:
            cost_term_samples_batch = cost_term_samples_batch.unsqueeze(-1)
        if cost_term_values_batch.ndim == 1:
            cost_term_values_batch = cost_term_values_batch.unsqueeze(-1)
        return (
            cost_term_returns_batch.reshape(-1, cost_term_returns_batch.shape[-1]),
            cost_term_advantages_batch.reshape(-1, cost_term_advantages_batch.shape[-1]),
            cost_term_samples_batch.reshape(-1, cost_term_samples_batch.shape[-1]),
            cost_term_values_batch.reshape(-1, cost_term_values_batch.shape[-1]),
        )

    @staticmethod
    def _match_cost_heads(cost_values: torch.Tensor, target_heads: int) -> torch.Tensor:
        if cost_values.ndim == 1:
            cost_values = cost_values.unsqueeze(-1)
        if cost_values.shape[-1] == target_heads:
            return cost_values
        if cost_values.shape[-1] == 1 and target_heads > 1:
            return cost_values.expand(-1, target_heads)
        if cost_values.shape[-1] > target_heads:
            return cost_values[:, :target_heads]
        pad = cost_values[:, -1:].expand(-1, target_heads - cost_values.shape[-1])
        return torch.cat([cost_values, pad], dim=-1)

    def _normalize_constraint_advantages(self, cost_terms_adv: torch.Tensor) -> torch.Tensor:
        if cost_terms_adv.numel() == 0:
            return cost_terms_adv
        mean = cost_terms_adv.mean(dim=0, keepdim=True)
        std = cost_terms_adv.std(dim=0, unbiased=False, keepdim=True)
        normalized = (cost_terms_adv - mean) / (std + 1e-8)
        return self._sanitize_tensor(
            normalized,
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
            clamp=1.0e3,
        )

    def _resolve_limit_tensor(
        self,
        raw_limits: torch.Tensor | None,
        num_constraints: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if raw_limits is None:
            return None
        d = raw_limits.to(device=device, dtype=torch.float32)
        if d.numel() == 1:
            return d.expand(num_constraints)
        if d.numel() != num_constraints:
            return torch.full((num_constraints,), float(d.flatten()[0].item()), device=device)
        return d.flatten()

    def _resolve_constraint_limits(self, num_constraints: int, device: torch.device) -> torch.Tensor:
        raw_limits = self.constraint_limits
        if self.adaptive_constraint_curriculum and self.constraint_limits_current is not None:
            raw_limits = self.constraint_limits_current
        resolved = self._resolve_limit_tensor(raw_limits, num_constraints, device)
        if resolved is None:
            return torch.full((num_constraints,), float(self.cost_limit), device=device)
        return resolved

    def _resolve_constraint_gate_indices(self, num_constraints: int) -> list[int]:
        if num_constraints <= 0:
            return []
        if not self.constraint_curriculum_names:
            return list(range(num_constraints))
        if not self.constraint_names:
            return []
        max_names = min(len(self.constraint_names), num_constraints)
        name_to_idx = {self.constraint_names[idx]: idx for idx in range(max_names)}
        indices: list[int] = []
        for name in self.constraint_curriculum_names:
            idx = name_to_idx.get(name)
            if idx is None or idx in indices:
                continue
            indices.append(idx)
        return indices

    def _constraint_curriculum_progress(
        self,
        current_limits: torch.Tensor,
        final_limits: torch.Tensor,
    ) -> float:
        start_limits = self._resolve_limit_tensor(
            self.constraint_limits_start,
            current_limits.numel(),
            current_limits.device,
        )
        if start_limits is None:
            start_limits = current_limits
        progress = torch.ones_like(current_limits)
        descending_mask = start_limits > (final_limits + 1.0e-8)
        if torch.any(descending_mask):
            progress[descending_mask] = torch.clamp(
                (start_limits[descending_mask] - current_limits[descending_mask])
                / (start_limits[descending_mask] - final_limits[descending_mask]),
                min=0.0,
                max=1.0,
            )
        static_mask = ~descending_mask
        if torch.any(static_mask):
            progress[static_mask] = (
                torch.abs(current_limits[static_mask] - final_limits[static_mask]) <= 1.0e-8
            ).float()
        return float(progress.mean().item())

    def _update_constraint_limit_curriculum(self, j_cost: torch.Tensor) -> dict[str, float]:
        if j_cost.ndim != 1:
            j_cost = j_cost.reshape(-1)
        j_cost = self._sanitize_tensor(
            j_cost.detach().to(self.device, dtype=torch.float32),
            nan=0.0,
            posinf=1.0e6,
            neginf=0.0,
            clamp=1.0e6,
        )
        num_constraints = int(j_cost.numel())
        current_limits = self._resolve_limit_tensor(
            self.constraint_limits_current if self.constraint_limits_current is not None else self.constraint_limits,
            num_constraints,
            j_cost.device,
        )
        if current_limits is None:
            current_limits = torch.full((num_constraints,), float(self.cost_limit), device=j_cost.device)
        final_limits = self._resolve_limit_tensor(
            self.constraint_limits_final if self.constraint_limits_final is not None else self.constraint_limits,
            num_constraints,
            j_cost.device,
        )
        if final_limits is None:
            final_limits = current_limits

        metrics = {
            "curriculum_enabled": float(self.adaptive_constraint_curriculum),
            "curriculum_update_count": float(self._constraint_curriculum_updates),
            "curriculum_gate_count": 0.0,
            "curriculum_gate_ready": 0.0,
            "curriculum_gate_max_ratio": 0.0,
            "curriculum_gate_min_margin": 0.0,
            "curriculum_limit_mean": float(current_limits.mean().item()) if num_constraints > 0 else 0.0,
            "curriculum_final_limit_mean": float(final_limits.mean().item()) if num_constraints > 0 else 0.0,
            "curriculum_progress": self._constraint_curriculum_progress(current_limits, final_limits)
            if num_constraints > 0
            else 1.0,
            "curriculum_tighten_triggered": 0.0,
            "curriculum_tighten_count": float(self._constraint_curriculum_tighten_count),
        }
        if not self.adaptive_constraint_curriculum or num_constraints == 0:
            return metrics

        if self.constraint_limits_current is None:
            self.constraint_limits_current = current_limits.detach().cpu()

        if self._constraint_curriculum_ema is None or self._constraint_curriculum_ema.numel() != num_constraints:
            ema_cost = j_cost.clone()
        else:
            ema_cost = self._constraint_curriculum_ema.to(device=j_cost.device, dtype=j_cost.dtype)
            ema_cost = self.constraint_curriculum_ema_decay * ema_cost + (
                1.0 - self.constraint_curriculum_ema_decay
            ) * j_cost
        self._constraint_curriculum_ema = ema_cost.detach().cpu()

        gate_indices = self._resolve_constraint_gate_indices(num_constraints)
        metrics["curriculum_gate_count"] = float(len(gate_indices))
        ready_to_tighten = False
        if gate_indices:
            gate_tensor = torch.as_tensor(gate_indices, device=j_cost.device, dtype=torch.long)
            gate_ema = ema_cost.index_select(0, gate_tensor)
            gate_limits = current_limits.index_select(0, gate_tensor)
            gate_ratio = gate_ema / torch.clamp(gate_limits, min=1.0e-8)
            gate_margin = gate_limits - gate_ema
            ready_to_tighten = bool(
                torch.all(gate_ema <= gate_limits * self.constraint_curriculum_alpha).item()
            )
            metrics["curriculum_gate_ready"] = float(ready_to_tighten)
            metrics["curriculum_gate_max_ratio"] = float(gate_ratio.max().item())
            metrics["curriculum_gate_min_margin"] = float(gate_margin.min().item())

        self._constraint_curriculum_updates += 1
        metrics["curriculum_update_count"] = float(self._constraint_curriculum_updates)

        should_check = (
            self._constraint_curriculum_updates % self.constraint_curriculum_check_interval == 0
        )
        has_tightening_room = bool(torch.any(current_limits > (final_limits + 1.0e-8)).item())
        if should_check and ready_to_tighten and has_tightening_room:
            tightened_limits = torch.minimum(
                current_limits,
                torch.maximum(final_limits, current_limits * self.constraint_curriculum_shrink),
            )
            if torch.any(tightened_limits < (current_limits - 1.0e-10)):
                self.constraint_limits_current = tightened_limits.detach().cpu()
                current_limits = tightened_limits
                self._constraint_curriculum_tighten_count += 1
                metrics["curriculum_tighten_triggered"] = 1.0

        metrics["curriculum_tighten_count"] = float(self._constraint_curriculum_tighten_count)
        metrics["curriculum_limit_mean"] = float(current_limits.mean().item())
        metrics["curriculum_progress"] = self._constraint_curriculum_progress(
            current_limits,
            final_limits,
        )
        return metrics

    def state_dict(self) -> dict:
        return {
            "step_size": self.step_size,
            "constraint_limits_current": (
                self.constraint_limits_current.detach().clone()
                if self.constraint_limits_current is not None
                else None
            ),
            "constraint_curriculum_ema": (
                self._constraint_curriculum_ema.detach().clone()
                if self._constraint_curriculum_ema is not None
                else None
            ),
            "constraint_curriculum_updates": self._constraint_curriculum_updates,
            "constraint_curriculum_tighten_count": self._constraint_curriculum_tighten_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        step_size = state_dict.get("step_size")
        if step_size is not None:
            self.step_size = float(step_size)
            self._set_actor_learning_rate(self.step_size)
        current_limits = state_dict.get("constraint_limits_current")
        if current_limits is not None:
            if not torch.is_tensor(current_limits):
                current_limits = torch.as_tensor(current_limits, dtype=torch.float32)
            self.constraint_limits_current = current_limits.detach().clone().cpu()
        ema_state = state_dict.get("constraint_curriculum_ema")
        if ema_state is not None:
            if not torch.is_tensor(ema_state):
                ema_state = torch.as_tensor(ema_state, dtype=torch.float32)
            self._constraint_curriculum_ema = ema_state.detach().clone().cpu()
        updates = state_dict.get("constraint_curriculum_updates")
        if updates is not None:
            self._constraint_curriculum_updates = int(updates)
        tighten_count = state_dict.get("constraint_curriculum_tighten_count")
        if tighten_count is not None:
            self._constraint_curriculum_tighten_count = int(tighten_count)

    def _project_safe_set(
        self,
        theta_prime: torch.Tensor,
        theta_anchor: torch.Tensor,
        a_mat: torch.Tensor,
        b_budget: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if a_mat.numel() == 0:
            return theta_prime, 0
        delta = theta_prime - theta_anchor
        violation = a_mat.transpose(0, 1).matmul(delta) - b_budget
        active_mask = violation > 0.0
        if not torch.any(active_mask):
            return theta_prime, 0
        a_active = a_mat[:, active_mask]
        v_active = violation[active_mask]
        q_mat = a_active.transpose(0, 1).matmul(a_active)
        lamb = self._solve_nonnegative_qp(q_mat, v_active)
        theta_proj = theta_prime - a_active.matmul(lamb)
        return theta_proj, int(active_mask.sum().item())

    def _solve_nonnegative_qp(self, q_mat: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        if q_mat.numel() == 0:
            return torch.zeros_like(v_vec)
        eye = torch.eye(q_mat.shape[0], device=q_mat.device, dtype=q_mat.dtype)
        q_reg = q_mat + self.projection_eps * eye
        max_eig = torch.linalg.eigvalsh(q_reg).max()
        step = 1.0 / (max_eig + self.projection_eps)
        lamb = torch.zeros_like(v_vec)
        for _ in range(self.softproj_max_iters):
            grad = q_reg.matmul(lamb) - v_vec
            new_lamb = torch.clamp(lamb - step * grad, min=0.0)
            if torch.max(torch.abs(new_lamb - lamb)) <= self.softproj_tol:
                lamb = new_lamb
                break
            lamb = new_lamb
        return lamb

    def _evaluate_candidate(
        self,
        projection_batch: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        with torch.inference_mode():
            self.policy.act(
                projection_batch["obs"],
                masks=projection_batch["masks"],
                hidden_states=projection_batch["hid_actor"],
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            kl = self._all_reduce_mean(
                torch.mean(
                    self._safe_kl(
                        mu_batch,
                        sigma_batch,
                        projection_batch["old_mu"],
                        projection_batch["old_sigma"],
                    )
                )
            )
        return kl

    def _set_actor_learning_rate(self, learning_rate: float):
        self.learning_rate = float(learning_rate)
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _adapt_step_size(self, accept_rate: float, mean_cost_margin: float):
        if not self.step_size_adaptive:
            self.step_size = self.learning_rate
            return
        if mean_cost_margin > self.step_size_cost_margin and accept_rate >= self.target_accept_rate:
            self.step_size = min(self.step_size_max, self.learning_rate * self.step_size_up)
        elif mean_cost_margin < 0.0 or accept_rate < self.target_accept_rate * 0.5:
            self.step_size = max(self.step_size_min, self.learning_rate * self.step_size_down)
        else:
            self.step_size = min(max(self.learning_rate, self.step_size_min), self.step_size_max)
        self.learning_rate = self.step_size

    def _step_constraint_scale(self, cost_violation: float | None = None):
        if cost_violation is None:
            super()._step_constraint_scale()
            return
        if cost_violation > self.k_violation_threshold:
            self.k_value = min(self.k_max, self.k_value * self.k_growth)
        else:
            self.k_value = max(self.k_min, self.k_value * self.k_decay)

    def _all_reduce_grads(self, grads: list[torch.Tensor]):
        for grad in grads:
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
            grad /= self.gpu_world_size

    def _all_reduce_parameter_grads(self, parameters: list[torch.nn.Parameter]):
        for param in parameters:
            if param.grad is None:
                continue
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
            param.grad /= self.gpu_world_size

    def _get_actor_params(self) -> list[torch.nn.Parameter]:
        params = [param for param in self.policy.actor.parameters() if param.requires_grad]
        std_param = getattr(self.policy, "std", None)
        log_std_param = getattr(self.policy, "log_std", None)
        if isinstance(std_param, torch.nn.Parameter) and std_param.requires_grad:
            params.append(std_param)
        if isinstance(log_std_param, torch.nn.Parameter) and log_std_param.requires_grad:
            params.append(log_std_param)
        return params

    def _flatten_tensors(self, tensor_list: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([tensor.reshape(-1) for tensor in tensor_list], dim=0)

    def _actor_param_vector(self) -> torch.Tensor:
        return self._flatten_tensors([param.data for param in self._actor_params])

    def _set_actor_param_vector(self, vector: torch.Tensor):
        offset = 0
        for param in self._actor_params:
            numel = param.numel()
            param.data.copy_(vector[offset : offset + numel].view_as(param))
            offset += numel
