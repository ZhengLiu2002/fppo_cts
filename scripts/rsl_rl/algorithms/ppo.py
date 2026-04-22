# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .contracts import CTSRuntimeContract
from scripts.rsl_rl.modules.actor_critic_with_encoder import ActorCriticRMA as ActorCritic
from scripts.rsl_rl.storage.rollout_storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    cts_runtime_contract = CTSRuntimeContract()

    policy: ActorCritic
    """The actor critic module."""

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
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Cost advantage normalization
        normalize_cost_advantage: bool = False,
        # Positive-part cost violation regularization
        cost_limit: float = 0.0,
        cost_viol_loss_coef: float = 0.0,
        k_value: float = 1.0,
        k_growth: float = 1.0,
        k_max: float = 1.0,
        k_decay: float = 1.0,
        k_min: float = 0.0,
        k_violation_threshold: float = 0.02,
        velocity_estimation_loss_coef: float = 0.0,
        student_group_ratio: float = 0.25,
        reconstruction_learning_rate: float | None = None,
        num_reconstruction_epochs: int = 2,
        detach_student_encoder_during_rl: bool = True,
        roa_teacher_reg_coef_start: float = 0.0,
        roa_teacher_reg_coef_end: float = 0.0,
        roa_teacher_reg_warmup_updates: int = 5000,
        roa_teacher_reg_ramp_updates: int = 5000,
        roa_teacher_reg_scope: str = "teacher",
        roa_teacher_reg_loss: str = "mse",
        cost_ratio_clip: float | None = None,
        log_ratio_clip: float = 6.0,
        kl_hard_ratio: float = 4.0,
        kl_hard_abs: float = 0.05,
        constraint_limits: list[float] | tuple[float, ...] | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reconstruction_optimizer: torch.optim.Optimizer | None = None
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()
        self.training_type = "rl"
        self._action_dim = 0
        self._update_counter = 0

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = gamma if cost_gamma is None else cost_gamma
        self.cost_lam = lam if cost_lam is None else cost_lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.normalize_cost_advantage = normalize_cost_advantage
        self.cost_limit = cost_limit
        self.cost_viol_loss_coef = cost_viol_loss_coef
        self.k_value = float(k_value)
        self.k_growth = float(k_growth)
        self.k_max = float(k_max)
        self.k_decay = float(k_decay)
        self.k_min = float(k_min)
        self.k_violation_threshold = float(k_violation_threshold)
        self.velocity_estimation_loss_coef = float(velocity_estimation_loss_coef)
        self.student_group_ratio = float(student_group_ratio)
        self.reconstruction_learning_rate = (
            None if reconstruction_learning_rate is None else float(reconstruction_learning_rate)
        )
        self.num_reconstruction_epochs = max(int(num_reconstruction_epochs), 0)
        self.detach_student_encoder_during_rl = bool(detach_student_encoder_during_rl)
        self.roa_teacher_reg_coef_start = max(float(roa_teacher_reg_coef_start), 0.0)
        self.roa_teacher_reg_coef_end = max(float(roa_teacher_reg_coef_end), 0.0)
        self.roa_teacher_reg_warmup_updates = max(int(roa_teacher_reg_warmup_updates), 0)
        self.roa_teacher_reg_ramp_updates = max(int(roa_teacher_reg_ramp_updates), 0)
        self.roa_teacher_reg_scope = str(roa_teacher_reg_scope).strip().lower()
        if self.roa_teacher_reg_scope not in {"teacher", "all"}:
            raise ValueError(
                "Unsupported ROA teacher regularization scope: "
                f"{roa_teacher_reg_scope}. Expected 'teacher' or 'all'."
            )
        self.roa_teacher_reg_loss = str(roa_teacher_reg_loss).strip().lower()
        if self.roa_teacher_reg_loss not in {"mse", "l2"}:
            raise ValueError(
                "Unsupported ROA teacher regularization loss: "
                f"{roa_teacher_reg_loss}. Expected 'mse' or 'l2'."
            )
        self.cost_ratio_clip = float(cost_ratio_clip) if cost_ratio_clip is not None else clip_param
        self.log_ratio_clip = float(log_ratio_clip)
        self.kl_hard_ratio = float(kl_hard_ratio)
        self.kl_hard_abs = float(kl_hard_abs)
        self.constraint_limits = (
            torch.as_tensor(constraint_limits, dtype=torch.float32)
            if constraint_limits is not None
            else None
        )
        self.train_metrics: dict[str, float] = {}

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        self.training_type = str(training_type).strip().lower()
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            self.device,
        )
        self._action_dim = int(actions_shape[0])
        self.reconstruction_optimizer = None
        if self._cts_enabled():
            student_params = list(getattr(self.policy, "student_encoder_parameters", lambda: [])())
            if student_params and self.num_reconstruction_epochs > 0:
                recon_lr = (
                    self.reconstruction_learning_rate
                    if self.reconstruction_learning_rate is not None
                    else self.learning_rate
                )
                self.reconstruction_optimizer = optim.Adam(student_params, lr=float(recon_lr))

    def _all_reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        if self.is_multi_gpu:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value /= self.gpu_world_size
        return value

    @staticmethod
    def _sanitize_tensor(
        tensor: torch.Tensor,
        *,
        nan: float = 0.0,
        posinf: float = 1.0e6,
        neginf: float = -1.0e6,
        clamp: float | None = None,
    ) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
        if clamp is not None:
            tensor = torch.clamp(tensor, min=-clamp, max=clamp)
        return tensor

    def _safe_ratio(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
    ) -> torch.Tensor:
        actions_log_prob_batch = torch.squeeze(actions_log_prob_batch)
        old_actions_log_prob_batch = torch.squeeze(old_actions_log_prob_batch)
        log_ratio = actions_log_prob_batch - old_actions_log_prob_batch
        log_ratio = self._sanitize_tensor(
            log_ratio,
            nan=0.0,
            posinf=self.log_ratio_clip,
            neginf=-self.log_ratio_clip,
            clamp=self.log_ratio_clip,
        )
        return torch.exp(log_ratio)

    def _safe_kl(
        self,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ) -> torch.Tensor:
        sigma_batch = self._sanitize_tensor(sigma_batch, nan=1.0e-6, posinf=1.0, neginf=1.0e-6)
        old_sigma_batch = self._sanitize_tensor(
            old_sigma_batch, nan=1.0e-6, posinf=1.0, neginf=1.0e-6
        )
        sigma_batch = torch.clamp(sigma_batch, min=1.0e-6)
        old_sigma_batch = torch.clamp(old_sigma_batch, min=1.0e-6)
        mu_batch = self._sanitize_tensor(
            mu_batch, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
        )
        old_mu_batch = self._sanitize_tensor(
            old_mu_batch, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
        )
        kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            dim=-1,
        )
        return self._sanitize_tensor(kl, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6)

    def _batch_cost_stats(
        self, cost_returns_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cost_returns_batch = self._sanitize_tensor(
            cost_returns_batch, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
        )
        cost_return_mean = self._all_reduce_mean(cost_returns_batch.mean())
        cost_violation_rate = self._all_reduce_mean(
            (cost_returns_batch > self.cost_limit).float().mean()
        )
        cost_return_mean = self._sanitize_tensor(
            cost_return_mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
        )
        cost_violation_rate = self._sanitize_tensor(
            cost_violation_rate, nan=0.0, posinf=1.0, neginf=0.0, clamp=1.0
        )
        c_hat = cost_return_mean - self.cost_limit
        c_hat = self._sanitize_tensor(c_hat, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6)
        return cost_return_mean, cost_violation_rate, c_hat

    def _positive_cost_penalty(
        self, cost_surrogate: torch.Tensor, c_hat: torch.Tensor, detach_violation: bool = True
    ) -> torch.Tensor:
        if self.cost_viol_loss_coef <= 0.0 or self.k_value <= 0.0:
            return torch.zeros((), device=cost_surrogate.device, dtype=cost_surrogate.dtype)
        violation = c_hat.detach() if detach_violation else c_hat
        return self.cost_viol_loss_coef * self.k_value * torch.relu(cost_surrogate + violation)

    def _step_constraint_scale(self, cost_violation: float | None = None):
        if cost_violation is None:
            self.k_value = min(self.k_max, self.k_value * self.k_growth)
            return
        if cost_violation > self.k_violation_threshold:
            self.k_value = min(self.k_max, self.k_value * self.k_growth)
        else:
            self.k_value = max(self.k_min, self.k_value * self.k_decay)

    def _kl_hard_limit(self) -> float:
        if self.desired_kl is None:
            return self.kl_hard_abs
        return max(self.kl_hard_abs, float(self.desired_kl) * self.kl_hard_ratio)

    def _use_history_encoding(self) -> bool:
        actor = getattr(self.policy, "actor", None)
        return bool(getattr(actor, "num_hist", 0) > 0)

    def _cts_enabled(self) -> bool:
        return self.training_type == "cts"

    @staticmethod
    def _student_mask(actor_is_student: torch.Tensor) -> torch.Tensor:
        if actor_is_student.ndim > 1:
            actor_is_student = actor_is_student.view(actor_is_student.shape[0], -1)[:, 0]
        return actor_is_student.to(dtype=torch.bool)

    def _roa_teacher_reg_coefficient(self) -> float:
        if not self._cts_enabled():
            return 0.0
        start = float(getattr(self, "roa_teacher_reg_coef_start", 0.0) or 0.0)
        end = float(getattr(self, "roa_teacher_reg_coef_end", 0.0) or 0.0)
        if start <= 0.0 and end <= 0.0:
            return 0.0
        warmup = max(int(getattr(self, "roa_teacher_reg_warmup_updates", 0) or 0), 0)
        ramp = int(getattr(self, "roa_teacher_reg_ramp_updates", 0) or 0)
        update_index = max(int(getattr(self, "_update_counter", 0) or 0), 0)
        if ramp <= 0:
            return end if update_index >= warmup else start
        progress = min(max((update_index - warmup) / float(ramp), 0.0), 1.0)
        return start + (end - start) * progress

    def _teacher_latent_regularization_loss(
        self,
        obs_batch: torch.Tensor,
        actor_is_student_batch: torch.Tensor | None,
        *,
        coefficient: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        zero = torch.zeros((), device=obs_batch.device)
        coefficient = (
            self._roa_teacher_reg_coefficient()
            if coefficient is None
            else max(float(coefficient), 0.0)
        )
        if (
            coefficient <= 0.0
            or not self._cts_enabled()
            or not hasattr(self.policy, "teacher_latent")
            or not hasattr(self.policy, "student_latent")
        ):
            return zero, zero, 0

        if self.roa_teacher_reg_scope == "all":
            reg_idx = torch.arange(obs_batch.shape[0], device=obs_batch.device)
        else:
            actor_is_student = self._resolve_actor_is_student_batch(
                actor_is_student_batch,
                obs_batch.shape[0],
                obs_batch.device,
            )
            reg_idx = self._role_indices(actor_is_student)[0]
        if reg_idx.numel() <= 0:
            return zero, zero, 0

        reg_obs = obs_batch[reg_idx]
        teacher_latent = self.policy.teacher_latent(reg_obs)
        student_latent = self.policy.student_latent(reg_obs).detach()
        if self.roa_teacher_reg_loss == "l2":
            latent_reg_loss = torch.linalg.vector_norm(
                teacher_latent - student_latent, dim=-1
            ).mean()
        else:
            latent_reg_loss = torch.nn.functional.mse_loss(teacher_latent, student_latent)
        latent_reg_loss = self._sanitize_tensor(
            latent_reg_loss,
            nan=0.0,
            posinf=1.0e6,
            neginf=0.0,
            clamp=1.0e6,
        )
        return latent_reg_loss, latent_reg_loss * coefficient, int(reg_idx.numel())

    def _resolve_actor_is_student_batch(
        self,
        actor_is_student: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if actor_is_student is None:
            return torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
        actor_is_student = actor_is_student.to(device=device)
        if actor_is_student.numel() == 1:
            return actor_is_student.bool().expand(batch_size, 1)
        return actor_is_student.view(batch_size, -1).bool()

    def _role_indices(self, actor_is_student: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        student_mask = self._student_mask(actor_is_student)
        student_idx = torch.nonzero(student_mask, as_tuple=False).flatten()
        teacher_idx = torch.nonzero(~student_mask, as_tuple=False).flatten()
        return teacher_idx, student_idx

    def _evaluate_cts_group(
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
        velocity_feature = self.policy.actor_mode_velocity_feature(obs_batch, actor_mode)
        self.policy.update_distribution_with_latent(
            obs_batch,
            latent,
            velocity_feature=velocity_feature,
        )
        if actions is None:
            actions = self.policy.distribution.sample()
        log_prob = self.policy.get_actions_log_prob(actions)
        values = self.policy.evaluate(
            critic_obs_batch,
            observations=obs_batch,
            actor_mode=actor_mode,
            latent=latent,
            detach_student_encoder=detach_student,
        )
        cost_pred = self.policy.evaluate_cost(
            critic_obs_batch,
            observations=obs_batch,
            actor_mode=actor_mode,
            latent=latent,
            detach_student_encoder=detach_student,
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

    def _evaluate_cts_policy_batch(
        self,
        obs_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor,
        actor_is_student: torch.Tensor | None,
        *,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        actor_is_student = self._resolve_actor_is_student_batch(
            actor_is_student,
            obs_batch.shape[0],
            obs_batch.device,
        )
        batch_size = obs_batch.shape[0]
        action_dim = int(actions.shape[-1]) if actions is not None else self._action_dim
        device = obs_batch.device
        dtype = obs_batch.dtype

        batched_actions = torch.zeros(batch_size, action_dim, device=device, dtype=dtype)
        log_prob = torch.zeros(batch_size, device=device, dtype=dtype)
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
            ) = self._evaluate_cts_group(
                obs_batch[batch_idx],
                critic_obs_batch[batch_idx],
                actor_mode,
                actions=group_actions,
            )
            batched_actions[batch_idx] = group_actions
            log_prob[batch_idx] = group_log_prob
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

    def _evaluate_actor_batch(
        self,
        obs_batch: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
        actor_is_student: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cts_enabled():
            actor_is_student = self._resolve_actor_is_student_batch(
                actor_is_student,
                obs_batch.shape[0],
                obs_batch.device,
            )
            batch_size = obs_batch.shape[0]
            action_dim = int(actions.shape[-1]) if actions is not None else self._action_dim
            batched_actions = torch.zeros(
                batch_size, action_dim, device=obs_batch.device, dtype=obs_batch.dtype
            )
            log_prob = torch.zeros(batch_size, device=obs_batch.device, dtype=obs_batch.dtype)
            mu = torch.zeros(batch_size, action_dim, device=obs_batch.device, dtype=obs_batch.dtype)
            sigma = torch.zeros(
                batch_size, action_dim, device=obs_batch.device, dtype=obs_batch.dtype
            )
            entropy = torch.zeros(batch_size, device=obs_batch.device, dtype=obs_batch.dtype)

            for actor_mode, batch_idx in (
                ("teacher", self._role_indices(actor_is_student)[0]),
                ("student", self._role_indices(actor_is_student)[1]),
            ):
                if batch_idx.numel() <= 0:
                    continue
                detach_student = actor_mode == "student" and self.detach_student_encoder_during_rl
                latent = self.policy.actor_mode_latent(
                    obs_batch[batch_idx],
                    actor_mode,
                    detach_student_encoder=detach_student,
                )
                velocity_feature = self.policy.actor_mode_velocity_feature(
                    obs_batch[batch_idx],
                    actor_mode,
                )
                self.policy.update_distribution_with_latent(
                    obs_batch[batch_idx],
                    latent,
                    velocity_feature=velocity_feature,
                )
                group_actions = actions[batch_idx] if actions is not None else None
                if group_actions is None:
                    group_actions = self.policy.distribution.sample()
                batched_actions[batch_idx] = group_actions
                log_prob[batch_idx] = self.policy.get_actions_log_prob(group_actions)
                mu[batch_idx] = self.policy.action_mean
                sigma[batch_idx] = self.policy.action_std
                entropy[batch_idx] = self.policy.entropy
            return batched_actions, log_prob, mu, sigma, entropy

        self._policy_act(obs_batch, **kwargs)
        if actions is None:
            actions = self.policy.distribution.sample()
        return (
            actions,
            self.policy.get_actions_log_prob(actions),
            self.policy.action_mean,
            self.policy.action_std,
            self.policy.entropy,
        )

    def _evaluate_training_batch(
        self,
        obs_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor,
        *,
        actions: torch.Tensor,
        actor_is_student: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cts_enabled():
            (
                _current_actions,
                actions_log_prob_batch,
                value_batch,
                _cost_value_batch_agg,
                pred_cost_terms,
                mu_batch,
                sigma_batch,
                entropy_batch,
            ) = self._evaluate_cts_policy_batch(
                obs_batch,
                critic_obs_batch,
                actor_is_student,
                actions=actions,
            )
            return (
                actions_log_prob_batch,
                value_batch,
                pred_cost_terms,
                mu_batch,
                sigma_batch,
                entropy_batch,
            )

        self._policy_act(obs_batch, **kwargs)
        actions_log_prob_batch = self.policy.get_actions_log_prob(actions)
        value_batch = self.policy.evaluate(
            critic_obs_batch,
            masks=kwargs.get("masks"),
            hidden_states=kwargs.get("value_hidden_states"),
        )
        cost_value_batch = self.policy.evaluate_cost(
            critic_obs_batch,
            masks=kwargs.get("masks"),
            hidden_states=kwargs.get("cost_hidden_states"),
        )
        cost_value_batch = self._sanitize_tensor(
            cost_value_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
        )
        if cost_value_batch.ndim == 1:
            cost_value_batch = cost_value_batch.unsqueeze(-1)
        elif cost_value_batch.ndim > 2:
            cost_value_batch = cost_value_batch.view(cost_value_batch.shape[0], -1)
        return (
            actions_log_prob_batch,
            value_batch,
            cost_value_batch,
            self.policy.action_mean,
            self.policy.action_std,
            self.policy.entropy,
        )

    @staticmethod
    def _unpack_extra_batch(extra_batch: list[torch.Tensor]) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        cost_term_returns_batch = extra_batch[0] if len(extra_batch) > 0 else None
        cost_term_advantages_batch = extra_batch[1] if len(extra_batch) > 1 else None
        cost_term_rewards_batch = extra_batch[2] if len(extra_batch) > 2 else None
        cost_term_values_batch = extra_batch[3] if len(extra_batch) > 3 else None
        actor_is_student_batch = extra_batch[4] if len(extra_batch) > 4 else None
        return (
            cost_term_returns_batch,
            cost_term_advantages_batch,
            cost_term_rewards_batch,
            cost_term_values_batch,
            actor_is_student_batch,
        )

    def _policy_act(self, obs_batch: torch.Tensor, **kwargs):
        hist_encoding = kwargs.pop("hist_encoding", self._use_history_encoding())
        try:
            return self.policy.act(obs_batch, hist_encoding=hist_encoding, **kwargs)
        except TypeError as exc:
            if "hist_encoding" not in str(exc):
                raise
            return self.policy.act(obs_batch, **kwargs)

    def _velocity_estimation_loss(
        self,
        obs_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=obs_batch.device)
        velocity_estimation_loss_coef = float(
            getattr(self, "velocity_estimation_loss_coef", 0.0) or 0.0
        )
        if (
            velocity_estimation_loss_coef <= 0.0
            or critic_obs_batch is None
            or not self._use_history_encoding()
        ):
            return zero, zero
        if not hasattr(self.policy, "velocity_estimation"):
            return zero, zero
        prediction, target = self.policy.velocity_estimation(obs_batch, critic_obs_batch)
        if prediction is None or target is None:
            return zero, zero
        velocity_estimation_loss = torch.nn.functional.mse_loss(prediction, target.detach())
        velocity_estimation_loss = self._sanitize_tensor(
            velocity_estimation_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
        )
        return (
            velocity_estimation_loss,
            velocity_estimation_loss * velocity_estimation_loss_coef,
        )

    def _latent_alignment_epoch(self) -> float:
        if (
            not self._cts_enabled()
            or self.reconstruction_optimizer is None
            or self.num_reconstruction_epochs <= 0
        ):
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
            *extra_batch,
        ) in generator:
            _, _, _, _, actor_is_student_batch = self._unpack_extra_batch(extra_batch)
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
            student_params = list(self.policy.student_encoder_parameters())
            if student_params:
                nn.utils.clip_grad_norm_(student_params, self.max_grad_norm)
            self.reconstruction_optimizer.step()
            total_loss += reconstruction_loss.item()
            num_updates += 1

        if num_updates <= 0:
            return 0.0
        return total_loss / float(num_updates)

    def _split_cost_value_heads(
        self, cost_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split cost-value prediction into aggregate scalar and per-constraint heads."""
        if not torch.is_tensor(cost_values):
            cost_values = torch.as_tensor(cost_values, device=self.device)
        cost_values = cost_values.to(self.device)
        if cost_values.ndim == 1:
            cost_values = cost_values.unsqueeze(-1)
        elif cost_values.ndim > 2:
            cost_values = cost_values.view(cost_values.shape[0], -1)
        cost_values = self._sanitize_tensor(
            cost_values,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        aggregate = cost_values.sum(dim=-1, keepdim=True)
        return aggregate, cost_values

    def _prepare_cost_term_batches(
        self,
        cost_returns_batch: torch.Tensor,
        cost_advantages_batch: torch.Tensor,
        cost_term_returns_batch: torch.Tensor | None,
        cost_term_advantages_batch: torch.Tensor | None,
        cost_term_values_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare per-constraint tensors for multi-head cost critic training."""
        if cost_term_returns_batch is None or cost_term_advantages_batch is None:
            fallback_values = (
                cost_term_values_batch if cost_term_values_batch is not None else cost_returns_batch
            )
            if not torch.is_tensor(fallback_values):
                fallback_values = torch.as_tensor(fallback_values, device=self.device)
            fallback_values = self._sanitize_tensor(
                fallback_values.to(self.device),
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            if fallback_values.ndim == 1:
                fallback_values = fallback_values.unsqueeze(-1)
            return (
                cost_returns_batch.reshape(-1, 1),
                cost_advantages_batch.reshape(-1, 1),
                fallback_values.reshape(-1, fallback_values.shape[-1]),
            )

        if not torch.is_tensor(cost_term_returns_batch):
            cost_term_returns_batch = torch.as_tensor(cost_term_returns_batch, device=self.device)
        if not torch.is_tensor(cost_term_advantages_batch):
            cost_term_advantages_batch = torch.as_tensor(
                cost_term_advantages_batch, device=self.device
            )
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
        if cost_term_values_batch is None:
            cost_term_values_batch = cost_term_returns_batch
        if not torch.is_tensor(cost_term_values_batch):
            cost_term_values_batch = torch.as_tensor(cost_term_values_batch, device=self.device)
        cost_term_values_batch = self._sanitize_tensor(
            cost_term_values_batch.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        if cost_term_values_batch.ndim == 1:
            cost_term_values_batch = cost_term_values_batch.unsqueeze(-1)
        return (
            cost_term_returns_batch.reshape(-1, cost_term_returns_batch.shape[-1]),
            cost_term_advantages_batch.reshape(-1, cost_term_advantages_batch.shape[-1]),
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

    def _resolve_constraint_limits(
        self, num_constraints: int, device: torch.device
    ) -> torch.Tensor:
        if self.constraint_limits is None:
            return torch.full((num_constraints,), float(self.cost_limit), device=device)
        d = self.constraint_limits.to(device=device, dtype=torch.float32)
        if d.numel() == 1:
            return d.expand(num_constraints)
        if d.numel() != num_constraints:
            return torch.full((num_constraints,), float(d.flatten()[0].item()), device=device)
        return d.flatten()

    def _constraint_batch_stats(self, cost_terms_ret: torch.Tensor) -> dict[str, torch.Tensor]:
        if cost_terms_ret.ndim == 1:
            cost_terms_ret = cost_terms_ret.unsqueeze(-1)
        cost_terms_ret = self._sanitize_tensor(
            cost_terms_ret,
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
            clamp=1.0e6,
        )
        d_limits = self._resolve_constraint_limits(
            cost_terms_ret.shape[1], device=cost_terms_ret.device
        ).to(dtype=cost_terms_ret.dtype)
        j_cost = self._all_reduce_mean(cost_terms_ret.mean(dim=0))
        c_hat = self._sanitize_tensor(
            j_cost - d_limits,
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
            clamp=1.0e6,
        )
        aggregate_cost_return = self._sanitize_tensor(
            torch.sum(j_cost),
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
            clamp=1.0e6,
        )
        batch_cost_violation = self._all_reduce_mean(
            (cost_terms_ret > d_limits.unsqueeze(0)).any(dim=1).float().mean()
        )
        min_cost_margin = self._sanitize_tensor(
            torch.min(d_limits - j_cost),
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
            clamp=1.0e6,
        )
        max_c_hat = self._sanitize_tensor(
            torch.max(c_hat),
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
            clamp=1.0e6,
        )
        return {
            "d_limits": d_limits,
            "j_cost": j_cost,
            "c_hat": c_hat,
            "aggregate_cost_return": aggregate_cost_return,
            "batch_cost_violation": batch_cost_violation,
            "min_cost_margin": min_cost_margin,
            "max_c_hat": max_c_hat,
        }

    def _constraint_surrogate_terms(
        self,
        cost_advantages: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        if cost_advantages.ndim == 1:
            cost_advantages = cost_advantages.unsqueeze(-1)
        ratio = ratio.reshape(-1, 1).to(device=cost_advantages.device, dtype=cost_advantages.dtype)
        return torch.mean(cost_advantages * ratio, dim=0)

    def _constraint_weighted_advantages(
        self,
        cost_advantages: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if cost_advantages.ndim == 1:
            cost_advantages = cost_advantages.unsqueeze(-1)
        weights = weights.to(device=cost_advantages.device, dtype=cost_advantages.dtype)
        return cost_advantages.matmul(weights)

    def _positive_cost_penalty_per_constraint(
        self,
        cost_surrogate: torch.Tensor,
        c_hat: torch.Tensor,
        detach_violation: bool = True,
    ) -> torch.Tensor:
        if self.cost_viol_loss_coef <= 0.0 or self.k_value <= 0.0:
            return torch.zeros((), device=cost_surrogate.device, dtype=cost_surrogate.dtype)
        if cost_surrogate.ndim == 0:
            return self._positive_cost_penalty(
                cost_surrogate, c_hat, detach_violation=detach_violation
            )
        violation = c_hat.detach() if detach_violation else c_hat
        penalties = torch.relu(cost_surrogate + violation)
        return self.cost_viol_loss_coef * self.k_value * penalties.mean()

    def act(self, obs, critic_obs, hist_encoding=False):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        if self._cts_enabled():
            actor_is_student = hist_encoding
            (
                actions,
                actions_log_prob,
                values,
                cost_values,
                cost_term_values,
                action_mean,
                action_sigma,
                _entropy,
            ) = self._evaluate_cts_policy_batch(
                obs,
                critic_obs,
                actor_is_student,
                actions=None,
            )
            self.transition.actions = actions.detach()
            self.transition.values = values.detach()
            self.transition.cost_values = cost_values.detach()
            self.transition.cost_term_values = cost_term_values.detach()
            self.transition.actions_log_prob = actions_log_prob.unsqueeze(-1).detach()
            self.transition.action_mean = action_mean.detach()
            self.transition.action_sigma = action_sigma.detach()
            self.transition.actor_is_student = self._resolve_actor_is_student_batch(
                actor_is_student,
                obs.shape[0],
                obs.device,
            ).detach()
        else:
            self.transition.actions = self.policy.act(obs, hist_encoding).detach()
            self.transition.values = self.policy.evaluate(critic_obs).detach()
            cost_value_pred = self.policy.evaluate_cost(critic_obs).detach()
            self.transition.cost_values, self.transition.cost_term_values = (
                self._split_cost_value_heads(cost_value_pred)
            )
            self.transition.actions_log_prob = self.policy.get_actions_log_prob(
                self.transition.actions
            ).detach()
            self.transition.action_mean = self.policy.action_mean.detach()
            self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, infos, costs=None, cost_terms=None):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        if costs is None:
            costs = torch.zeros_like(rewards)
        self.transition.cost_rewards = costs.clone()
        self.transition.cost_term_rewards = None
        if cost_terms is not None:
            cost_term_values = (
                cost_terms.get("values", None) if isinstance(cost_terms, dict) else cost_terms
            )
            if cost_term_values is not None:
                if not torch.is_tensor(cost_term_values):
                    cost_term_values = torch.as_tensor(cost_term_values, device=self.device)
                cost_term_values = cost_term_values.to(self.device)
                if cost_term_values.ndim == 1:
                    cost_term_values = cost_term_values.unsqueeze(-1)
                elif cost_term_values.ndim > 2:
                    cost_term_values = cost_term_values.view(cost_term_values.shape[0], -1)
                self.transition.cost_term_rewards = cost_term_values.clone()
                if (
                    self.transition.cost_values is not None
                    and self.transition.cost_term_values is not None
                ):
                    pred_terms = self.transition.cost_term_values
                    target_heads = cost_term_values.shape[-1]
                    if pred_terms.shape[-1] > target_heads:
                        pred_terms = pred_terms[:, :target_heads]
                    elif pred_terms.shape[-1] < target_heads:
                        if pred_terms.shape[-1] == 1:
                            pred_terms = pred_terms.expand(-1, target_heads)
                        else:
                            pad = pred_terms[:, -1:].expand(-1, target_heads - pred_terms.shape[-1])
                            pred_terms = torch.cat([pred_terms, pad], dim=-1)
                    self.transition.cost_term_values = pred_terms.clone()
        if self.transition.cost_term_values is None and self.transition.cost_values is not None:
            self.transition.cost_term_values = self.transition.cost_values.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs, 1
            )
            self.transition.cost_rewards += self.cost_gamma * torch.squeeze(
                self.transition.cost_values * time_outs, 1
            )
            self.transition.cost_rewards = self._sanitize_tensor(
                self.transition.cost_rewards, nan=0.0, posinf=1.0e3, neginf=0.0, clamp=1.0e3
            )
            if self.transition.cost_term_rewards is not None:
                term_values = self.transition.cost_term_values
                if term_values is None:
                    term_values = self.transition.cost_values
                bootstrap = self.cost_gamma * term_values * time_outs
                self.transition.cost_term_rewards += bootstrap.expand_as(
                    self.transition.cost_term_rewards
                )
                self.transition.cost_term_rewards = self._sanitize_tensor(
                    self.transition.cost_term_rewards,
                    nan=0.0,
                    posinf=1.0e3,
                    neginf=0.0,
                    clamp=1.0e3,
                )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_obs, last_critic_obs=None, actor_is_student=None):
        if self._cts_enabled():
            if last_critic_obs is None:
                raise ValueError("CTS returns require both actor and critic observations.")
            (
                _actions,
                _actions_log_prob,
                last_values,
                last_cost_values,
                last_cost_term_values,
                _mu,
                _sigma,
                _entropy,
            ) = self._evaluate_cts_policy_batch(
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
            last_values = last_values.detach()
            last_cost_values = last_cost_values.detach()
            last_cost_term_values = last_cost_term_values.detach()
        else:
            critic_obs = last_obs if last_critic_obs is None else last_critic_obs
            last_values = self.policy.evaluate(critic_obs).detach()
            last_cost_pred = self.policy.evaluate_cost(critic_obs).detach()
            last_cost_values, last_cost_term_values = self._split_cost_value_heads(last_cost_pred)
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
            last_cost_values=last_cost_values,
            last_cost_term_values=last_cost_term_values,
            cost_gamma=self.cost_gamma,
            cost_lam=self.cost_lam,
            normalize_cost_advantage=self.normalize_cost_advantage
            and not self.normalize_advantage_per_mini_batch,
        )

    def update(self):  # noqa: C901
        self._update_counter += 1
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_viol_loss = 0.0
        mean_cost_return = 0.0
        mean_cost_violation = 0.0
        mean_cost_margin = 0.0
        mean_current_max_violation = 0.0
        mean_kl = 0.0
        mean_velocity_estimation_loss = 0.0
        mean_teacher_latent_reg_loss = 0.0
        mean_teacher_latent_reg_weighted = 0.0
        mean_teacher_latent_reg_row_ratio = 0.0
        skipped_updates = 0
        teacher_latent_reg_coef = self._roa_teacher_reg_coefficient()

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
            (
                cost_term_returns_batch,
                cost_term_advantages_batch,
                _cost_term_rewards_batch,
                cost_term_values_batch,
                actor_is_student_batch,
            ) = self._unpack_extra_batch(extra_batch)

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
                actions_log_prob_batch,
                value_batch,
                cost_value_batch,
                mu_batch,
                sigma_batch,
                entropy_batch,
            ) = self._evaluate_training_batch(
                obs_batch,
                critic_obs_batch,
                actions=actions_batch,
                actor_is_student=actor_is_student_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
                value_hidden_states=hid_states_batch[1],
                cost_hidden_states=hid_states_batch[2],
            )
            pred_cost_terms = self._match_cost_heads(cost_value_batch, cost_terms_ret.shape[1])
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
            velocity_estimation_loss, weighted_velocity_estimation_loss = (
                self._velocity_estimation_loss(obs_batch, critic_obs_batch)
            )
            teacher_latent_reg_loss, weighted_teacher_latent_reg_loss, teacher_reg_rows = (
                self._teacher_latent_regularization_loss(
                    obs_batch,
                    actor_is_student_batch,
                    coefficient=teacher_latent_reg_coef,
                )
            )

            loss = (
                surrogate_loss
                + viol_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                + weighted_velocity_estimation_loss
                + weighted_teacher_latent_reg_loss
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
            mean_velocity_estimation_loss += velocity_estimation_loss.item()
            mean_teacher_latent_reg_loss += teacher_latent_reg_loss.item()
            mean_teacher_latent_reg_weighted += weighted_teacher_latent_reg_loss.item()
            mean_teacher_latent_reg_row_ratio += teacher_reg_rows / float(
                max(obs_batch.shape[0], 1)
            )

        latent_alignment_loss = self._latent_alignment_epoch()
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
        mean_velocity_estimation_loss /= num_updates
        mean_teacher_latent_reg_loss /= num_updates
        mean_teacher_latent_reg_weighted /= num_updates
        mean_teacher_latent_reg_row_ratio /= num_updates
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
        }
        if self._cts_enabled():
            self.train_metrics["student_group_ratio"] = self.student_group_ratio
            self.train_metrics["teacher_latent_reg_coef"] = teacher_latent_reg_coef
            self.train_metrics["teacher_latent_reg_row_ratio"] = mean_teacher_latent_reg_row_ratio

        loss_dict = {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "viol": mean_viol_loss,
            "velocity_estimation": mean_velocity_estimation_loss,
            "teacher_latent_reg": mean_teacher_latent_reg_loss,
            "teacher_latent_reg_weighted": mean_teacher_latent_reg_weighted,
            "latent_alignment": latent_alignment_loss,
        }
        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [
            param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None
        ]
        if not grads:
            return
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

    def state_dict(self) -> dict:
        state = {
            "learning_rate": float(self.learning_rate),
        }
        if self.reconstruction_optimizer is not None:
            state["reconstruction_optimizer_state_dict"] = (
                self.reconstruction_optimizer.state_dict()
            )
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            return
        learning_rate = state_dict.get("learning_rate")
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
            optimizer = getattr(self, "optimizer", None)
            if hasattr(optimizer, "param_groups"):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
        if self.reconstruction_optimizer is not None:
            recon_state = state_dict.get("reconstruction_optimizer_state_dict")
            if recon_state is not None:
                self.reconstruction_optimizer.load_state_dict(recon_state)
