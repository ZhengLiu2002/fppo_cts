# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from scripts.rsl_rl.algorithms.ppo import PPO


class CTS(PPO):
    """Concurrent teacher-student PPO with shared actor-critic and student latent alignment."""

    def __init__(
        self,
        policy,
        *,
        student_group_ratio: float = 0.25,
        reconstruction_learning_rate: float | None = None,
        num_reconstruction_epochs: int = 2,
        detach_student_encoder_during_rl: bool = True,
        **kwargs,
    ):
        super().__init__(policy=policy, **kwargs)
        self.student_group_ratio = float(student_group_ratio)
        self.num_reconstruction_epochs = max(int(num_reconstruction_epochs), 0)
        self.detach_student_encoder_during_rl = bool(detach_student_encoder_during_rl)
        ppo_params = list(self.policy.ppo_parameters())
        self.optimizer = optim.Adam(ppo_params, lr=self.learning_rate)
        student_params = list(self.policy.student_encoder_parameters())
        self.reconstruction_optimizer = None
        if student_params:
            self.reconstruction_optimizer = optim.Adam(
                student_params,
                lr=float(reconstruction_learning_rate or self.learning_rate),
            )
        self._action_dim = 0

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        super().init_storage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
        )
        self._action_dim = int(actions_shape[0])

    @staticmethod
    def _student_mask(actor_is_student: torch.Tensor) -> torch.Tensor:
        if actor_is_student.ndim > 1:
            actor_is_student = actor_is_student.view(actor_is_student.shape[0], -1)[:, 0]
        return actor_is_student.to(dtype=torch.bool)

    def _role_indices(self, actor_is_student: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        student_mask = self._student_mask(actor_is_student)
        student_idx = torch.nonzero(student_mask, as_tuple=False).flatten()
        teacher_idx = torch.nonzero(~student_mask, as_tuple=False).flatten()
        return teacher_idx, student_idx

    def _evaluate_group(
        self,
        obs_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor,
        actor_mode: str,
        *,
        actions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        detach_student = actor_mode == "student" and self.detach_student_encoder_during_rl
        latent = self.policy.actor_mode_latent(
            obs_batch,
            actor_mode,
            detach_student_encoder=detach_student,
        )
        self.policy.update_distribution_with_latent(obs_batch, latent)
        if actions is None:
            actions = self.policy.distribution.sample()
        log_prob = self.policy.get_actions_log_prob(actions)
        values = self.policy.evaluate(
            critic_obs_batch,
            latent=latent,
        )
        cost_pred = self.policy.evaluate_cost(
            critic_obs_batch,
            latent=latent,
        )
        cost_values, cost_term_values = self._split_cost_value_heads(cost_pred)
        return (
            actions,
            log_prob,
            values,
            cost_values,
            cost_term_values,
            self.policy.entropy,
        )

    def _evaluate_policy_batch(
        self,
        obs_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor,
        actor_is_student: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = obs_batch.shape[0]
        action_dim = int(actions.shape[-1]) if actions is not None else self._action_dim
        device = obs_batch.device
        dtype = obs_batch.dtype

        batched_actions = torch.zeros(batch_size, action_dim, device=device, dtype=dtype)
        log_prob = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        values = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        cost_values = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        mu = torch.zeros(batch_size, action_dim, device=device, dtype=dtype)
        sigma = torch.zeros(batch_size, action_dim, device=device, dtype=dtype)
        entropy = torch.zeros(batch_size, device=device, dtype=dtype)
        cost_term_values = None

        for actor_mode, batch_idx in (
            ("teacher", self._role_indices(actor_is_student)[0]),
            ("student", self._role_indices(actor_is_student)[1]),
        ):
            if batch_idx.numel() <= 0:
                continue
            group_actions = actions[batch_idx] if actions is not None else None
            (
                group_actions,
                group_log_prob,
                group_values,
                group_cost_values,
                group_cost_term_values,
                group_entropy,
            ) = self._evaluate_group(
                obs_batch[batch_idx],
                critic_obs_batch[batch_idx],
                actor_mode,
                actions=group_actions,
            )
            batched_actions[batch_idx] = group_actions
            log_prob[batch_idx] = group_log_prob.unsqueeze(-1)
            values[batch_idx] = group_values
            cost_values[batch_idx] = group_cost_values
            mu[batch_idx] = self.policy.action_mean
            sigma[batch_idx] = self.policy.action_std
            entropy[batch_idx] = group_entropy
            if group_cost_term_values is not None:
                if cost_term_values is None:
                    cost_term_values = torch.zeros(
                        batch_size,
                        group_cost_term_values.shape[-1],
                        device=device,
                        dtype=group_cost_term_values.dtype,
                    )
                cost_term_values[batch_idx] = group_cost_term_values

        if cost_term_values is None:
            cost_term_values = cost_values.clone()
        return batched_actions, log_prob, values, cost_values, cost_term_values, mu, sigma, entropy

    def act(self, obs, critic_obs, actor_is_student):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        (
            actions,
            actions_log_prob,
            values,
            cost_values,
            cost_term_values,
            action_mean,
            action_sigma,
            _entropy,
        ) = self._evaluate_policy_batch(
            obs,
            critic_obs,
            actor_is_student,
            actions=None,
        )
        self.transition.actions = actions.detach()
        self.transition.values = values.detach()
        self.transition.cost_values = cost_values.detach()
        self.transition.cost_term_values = cost_term_values.detach()
        self.transition.actions_log_prob = actions_log_prob.detach()
        self.transition.action_mean = action_mean.detach()
        self.transition.action_sigma = action_sigma.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.actor_is_student = actor_is_student.view(-1, 1).detach().bool()
        return self.transition.actions

    def compute_returns(self, last_obs, last_critic_obs, actor_is_student):
        (
            _actions,
            _actions_log_prob,
            last_values,
            last_cost_values,
            last_cost_term_values,
            _mu,
            _sigma,
            _entropy,
        ) = self._evaluate_policy_batch(
            last_obs,
            last_critic_obs,
            actor_is_student,
            actions=torch.zeros(
                last_obs.shape[0],
                self._action_dim,
                device=last_obs.device,
                dtype=last_obs.dtype,
            ),
        )
        self.storage.compute_returns(
            last_values.detach(),
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
            last_cost_values=last_cost_values.detach(),
            last_cost_term_values=last_cost_term_values.detach(),
            cost_gamma=self.cost_gamma,
            cost_lam=self.cost_lam,
            normalize_cost_advantage=self.normalize_cost_advantage
            and not self.normalize_advantage_per_mini_batch,
        )

    def _latent_alignment_epoch(self) -> float:
        if self.reconstruction_optimizer is None or self.num_reconstruction_epochs <= 0:
            return 0.0

        total_loss = 0.0
        num_updates = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_reconstruction_epochs,
        )
        for (
            obs_batch,
            _critic_obs_batch,
            _actions_batch,
            _target_values_batch,
            _advantages_batch,
            _returns_batch,
            _cost_values_batch,
            _cost_returns_batch,
            _cost_advantages_batch,
            _old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
            _cost_term_returns_batch,
            _cost_term_advantages_batch,
            _cost_term_rewards_batch,
            _cost_term_values_batch,
            actor_is_student_batch,
        ) in generator:
            if actor_is_student_batch is None:
                continue
            student_mask = self._student_mask(actor_is_student_batch)
            if not torch.any(student_mask):
                continue
            student_obs = obs_batch[student_mask]
            student_latent, teacher_latent = self.policy.latent_alignment_targets(student_obs)
            reconstruction_loss = torch.nn.functional.mse_loss(student_latent, teacher_latent)
            reconstruction_loss = self._sanitize_tensor(
                reconstruction_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )
            self.reconstruction_optimizer.zero_grad()
            reconstruction_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.student_encoder_parameters(), self.max_grad_norm)
            self.reconstruction_optimizer.step()
            total_loss += reconstruction_loss.item()
            num_updates += 1

        if num_updates <= 0:
            return 0.0
        return total_loss / float(num_updates)

    def update(self):  # noqa: C901
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
            actor_is_student_batch = extra_batch[4] if len(extra_batch) > 4 else None
            if actor_is_student_batch is None:
                actor_is_student_batch = torch.zeros(
                    obs_batch.shape[0],
                    1,
                    device=obs_batch.device,
                    dtype=torch.bool,
                )

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

            advantages_batch = self._sanitize_tensor(
                advantages_batch, nan=0.0, posinf=1.0e3, neginf=-1.0e3, clamp=1.0e3
            )
            cost_advantages_batch = self._sanitize_tensor(
                cost_advantages_batch, nan=0.0, posinf=1.0e3, neginf=-1.0e3, clamp=1.0e3
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

            (
                _current_actions,
                actions_log_prob_batch,
                value_batch,
                _cost_value_batch_agg,
                pred_cost_terms,
                mu_batch,
                sigma_batch,
                entropy_batch,
            ) = self._evaluate_policy_batch(
                obs_batch,
                critic_obs_batch,
                actor_is_student_batch,
                actions=actions_batch,
            )
            pred_cost_terms = self._sanitize_tensor(
                pred_cost_terms, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            old_cost_terms = self._match_cost_heads(cost_terms_val, cost_terms_ret.shape[1])

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
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            ratio_cost = torch.clamp(
                ratio,
                1.0 - self.cost_ratio_clip,
                1.0 + self.cost_ratio_clip,
            )
            cost_surrogates = self._constraint_surrogate_terms(cost_terms_adv, ratio_cost)
            viol_loss = self._positive_cost_penalty_per_constraint(
                cost_surrogates,
                constraint_stats["c_hat"],
            )
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
                cost_value_clipped = old_cost_terms + (pred_cost_terms - old_cost_terms).clamp(
                    -self.clip_param, self.clip_param
                )
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

            nn.utils.clip_grad_norm_(self.policy.ppo_parameters(), self.max_grad_norm)
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

        reconstruction_loss = self._latent_alignment_epoch()

        num_updates = self.num_learning_epochs * self.num_mini_batches
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
            "current_max_violation": mean_current_max_violation,
            "student_group_ratio": self.student_group_ratio,
        }

        loss_dict = {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "viol": mean_viol_loss,
            "reconstruction": reconstruction_loss,
        }
        return loss_dict

    def state_dict(self) -> dict:
        state = {
            "k_value": float(self.k_value),
        }
        if self.reconstruction_optimizer is not None:
            state["reconstruction_optimizer_state_dict"] = (
                self.reconstruction_optimizer.state_dict()
            )
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            return
        self.k_value = float(state_dict.get("k_value", self.k_value))
        if self.reconstruction_optimizer is not None:
            recon_state = state_dict.get("reconstruction_optimizer_state_dict")
            if recon_state is not None:
                self.reconstruction_optimizer.load_state_dict(recon_state)
