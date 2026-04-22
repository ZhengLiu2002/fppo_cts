# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.optim as optim

from .contracts import CTSRuntimeContract
from .ppo import PPO


class FPPO(PPO):
    """First-Order Projected PPO with robust margins and geometry-aware correction."""

    cts_runtime_contract = CTSRuntimeContract(inject_constraint_names=True)

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
        backtrack_coeff=0.5,
        max_backtracks: int = 10,
        projection_eps=1.0e-8,
        fisher_damping=1.0e-3,
        fisher_num_chunks: int = 4,
        fisher_min_diag: float = 1.0e-6,
        cost_confidence=1.0,
        gradient_confidence=0.2,
        curvature_proxy=0.0,
        gradient_uncertainty_mode: str = "shards",
        gradient_uncertainty_shards: int = 4,
        uncertainty_update_interval: int = 4,
        uncertainty_ema_decay: float = 0.9,
        predictor_kl_target: float | None = None,
        predictor_kl_hard_limit: float | None = None,
        predictor_adaptive_lr: bool = True,
        predictor_lr_min: float | None = None,
        predictor_lr_max: float | None = None,
        qp_max_iters: int = 64,
        qp_tol: float = 1.0e-6,
        exact_qp_max_constraints: int = 8,
        max_sigma_a: float | None = 2.0,
        max_margin_abs: float | None = None,
        max_margin_ratio: float | None = 0.5,
        projection_radius_cap: float | None = 1.0,
        projection_radius_mode: str = "kl",
        active_constraint_threshold: float = 0.0,
        constraint_advantage_key: str = "cost_terms_adv_norm",
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
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
        constraint_limits: list[float] | tuple[float, ...] | None = None,
        constraint_names: list[str] | tuple[str, ...] | None = None,
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
            normalize_cost_advantage=False,
            constraint_limits=constraint_limits,
            velocity_estimation_loss_coef=velocity_estimation_loss_coef,
            student_group_ratio=student_group_ratio,
            reconstruction_learning_rate=reconstruction_learning_rate,
            num_reconstruction_epochs=num_reconstruction_epochs,
            detach_student_encoder_during_rl=detach_student_encoder_during_rl,
            roa_teacher_reg_coef_start=roa_teacher_reg_coef_start,
            roa_teacher_reg_coef_end=roa_teacher_reg_coef_end,
            roa_teacher_reg_warmup_updates=roa_teacher_reg_warmup_updates,
            roa_teacher_reg_ramp_updates=roa_teacher_reg_ramp_updates,
            roa_teacher_reg_scope=roa_teacher_reg_scope,
            roa_teacher_reg_loss=roa_teacher_reg_loss,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.backtrack_coeff = float(backtrack_coeff)
        self.max_backtracks = max(int(max_backtracks), 0)
        self.projection_eps = max(float(projection_eps), 1.0e-12)
        self.fisher_damping = max(float(fisher_damping), 0.0)
        self.fisher_num_chunks = max(int(fisher_num_chunks), 1)
        self.fisher_min_diag = max(float(fisher_min_diag), self.projection_eps)
        self.cost_confidence = max(float(cost_confidence), 0.0)
        self.gradient_confidence = max(float(gradient_confidence), 0.0)
        self.curvature_proxy = max(float(curvature_proxy), 0.0)
        self.gradient_uncertainty_mode = str(gradient_uncertainty_mode).strip().lower()
        self.gradient_uncertainty_shards = max(int(gradient_uncertainty_shards), 1)
        self.uncertainty_update_interval = max(int(uncertainty_update_interval), 1)
        self.uncertainty_ema_decay = min(max(float(uncertainty_ema_decay), 0.0), 0.9999)
        self.predictor_kl_target = (
            float(predictor_kl_target)
            if predictor_kl_target is not None
            else max(self._target_kl() * 0.75, 1.0e-6)
        )
        self.predictor_kl_hard_limit = (
            float(predictor_kl_hard_limit)
            if predictor_kl_hard_limit is not None
            else max(self._target_kl() * 2.0, self.kl_hard_abs)
        )
        self.predictor_adaptive_lr = bool(predictor_adaptive_lr)
        self.predictor_lr = float(learning_rate)
        self.predictor_lr_min = (
            float(predictor_lr_min)
            if predictor_lr_min is not None
            else max(self.predictor_lr * 0.1, 1.0e-6)
        )
        self.predictor_lr_max = (
            float(predictor_lr_max)
            if predictor_lr_max is not None
            else max(self.predictor_lr * 2.0, self.predictor_lr_min)
        )
        self.qp_max_iters = max(int(qp_max_iters), 1)
        self.qp_tol = max(float(qp_tol), 1.0e-12)
        self.exact_qp_max_constraints = max(int(exact_qp_max_constraints), 1)
        self.max_sigma_a = None if max_sigma_a is None else max(float(max_sigma_a), 0.0)
        self.max_margin_abs = None if max_margin_abs is None else max(float(max_margin_abs), 0.0)
        self.max_margin_ratio = (
            None if max_margin_ratio is None else max(float(max_margin_ratio), 0.0)
        )
        self.projection_radius_cap = (
            None if projection_radius_cap is None else max(float(projection_radius_cap), 0.0)
        )
        self.projection_radius_mode = str(projection_radius_mode).strip().lower()
        self.active_constraint_threshold = max(float(active_constraint_threshold), 0.0)
        self.constraint_advantage_key = str(constraint_advantage_key).strip()
        if self.constraint_advantage_key not in {"cost_terms_adv_raw", "cost_terms_adv_norm"}:
            raise ValueError(
                "constraint_advantage_key must be one of "
                "{'cost_terms_adv_raw', 'cost_terms_adv_norm'}; "
                f"got {constraint_advantage_key!r}."
            )
        self.constraint_names = (
            [str(name) for name in constraint_names] if constraint_names is not None else None
        )

        self._actor_params = self._get_actor_params()
        self._critic_params = list(self.policy.critic.parameters()) + list(
            self.policy.cost_critic.parameters()
        )
        self.actor_optimizer = optim.Adam(self._actor_params, lr=self.predictor_lr)
        self.critic_optimizer = optim.Adam(self._critic_params, lr=float(learning_rate))
        self.optimizer = {"actor": self.actor_optimizer, "critic": self.critic_optimizer}
        self.learning_rate = self.predictor_lr
        self.train_metrics: dict[str, float] = {}

        self._update_counter = 0
        self._sigma_a_cache: torch.Tensor | None = None
        self._sigma_a_active_indices: torch.Tensor | None = None
        self._uncertainty_cache_age = 0

    def update(self):
        self._update_counter += 1
        projection_batch = self._prepare_projection_batch()
        theta_anchor = self._actor_param_vector().detach().clone()
        projection_state = self._build_projection_state(projection_batch)
        predictor_metrics = self._run_ppo_predictor()
        theta_predictor = self._actor_param_vector().detach().clone()
        corrector_metrics = self._run_projection_corrector(
            projection_batch=projection_batch,
            projection_state=projection_state,
            theta_anchor=theta_anchor,
            theta_predictor=theta_predictor,
        )
        mean_value_loss, mean_cost_value_loss = self._update_value_functions()
        latent_alignment_loss = self._latent_alignment_epoch()

        cost_terms_rollout = projection_batch["cost_terms_rollout"]
        d_limits = projection_state["d_limits"]
        j_cost = projection_state["j_cost"]
        sample_violation = (cost_terms_rollout > d_limits.unsqueeze(0)).any(dim=1)
        if j_cost.numel() == 0:
            cost_return = 0.0
            cost_margin = 0.0
            current_max_violation = 0.0
            margin_mean = 0.0
            margin_max = 0.0
            margin_to_slack_ratio = 0.0
            sigma_j_mean = 0.0
            sigma_a_mean = 0.0
        else:
            slack = torch.clamp(d_limits - j_cost, min=self.projection_eps)
            margin_to_slack = projection_state["margins"] / slack
            cost_return = float(j_cost.sum().item())
            cost_margin = float(torch.min(d_limits - j_cost).item())
            current_max_violation = float(torch.max(j_cost - d_limits).item())
            margin_mean = float(projection_state["margins"].mean().item())
            margin_max = float(projection_state["margins"].max().item())
            margin_to_slack_ratio = float(margin_to_slack.mean().item())
            sigma_j_mean = float(projection_state["sigma_j"].mean().item())
            sigma_a_mean = float(projection_state["sigma_a"].mean().item())

        constraint_grad_norms = projection_state["constraint_grad_norms"]
        fisher_diag = projection_state["fisher_diag"]
        self.train_metrics = {
            "mean_cost_return": cost_return,
            "reward_return_mean": float(projection_batch["reward_returns"].mean().item()),
            "cost_limit_margin": cost_margin,
            "cost_violation_rate": float(
                self._all_reduce_mean(sample_violation.float().mean()).item()
            ),
            "current_max_violation": current_max_violation,
            "active_constraints": corrector_metrics["active_constraints"],
            "accept_rate": corrector_metrics["accept_rate"],
            "corrector_accept_rate": corrector_metrics["corrector_accept_rate"],
            "fallback_to_predictor_rate": corrector_metrics["fallback_to_predictor"],
            "accepted_backtrack_factor": corrector_metrics["accepted_backtrack_factor"],
            "backtrack_steps": corrector_metrics["backtrack_steps"],
            "step_size": corrector_metrics["accepted_step_norm"],
            "nominal_step_norm": corrector_metrics["nominal_step_norm"],
            "projected_step_norm": corrector_metrics["projected_step_norm"],
            "accepted_step_norm": corrector_metrics["accepted_step_norm"],
            "effective_step_ratio": corrector_metrics["accepted_backtrack_factor"],
            "kl": corrector_metrics["final_kl"],
            "fallback_rate": 1.0 - corrector_metrics["accept_rate"],
            "predicted_violation_max": corrector_metrics["predicted_violation_max"],
            "qp_condition": corrector_metrics["qp_condition"],
            "projection_applied_rate": float(corrector_metrics["active_constraints"] > 0.0),
            "projection_radius_scale": corrector_metrics["projection_radius_scale"],
            "margin_mean": margin_mean,
            "margin_max": margin_max,
            "margin_to_slack_ratio": margin_to_slack_ratio,
            "sigma_j_mean": sigma_j_mean,
            "sigma_a_mean": sigma_a_mean,
            "predictor_kl": predictor_metrics["mean_kl"],
            "predictor_stop_rate": predictor_metrics["stop_rate"],
            "predictor_updates": predictor_metrics["updates"],
            "predictor_update_ratio": predictor_metrics["update_ratio"],
            "predictor_lr": self.predictor_lr,
            "uncertainty_cache_age": float(self._uncertainty_cache_age),
            "constraint_grad_norm_mean": float(
                constraint_grad_norms.mean().item() if constraint_grad_norms.numel() > 0 else 0.0
            ),
            "constraint_grad_norm_max": float(
                constraint_grad_norms.max().item() if constraint_grad_norms.numel() > 0 else 0.0
            ),
            "fisher_diag_mean": float(fisher_diag.mean().item()),
            "fisher_diag_min": float(fisher_diag.min().item()),
            "fisher_diag_max": float(fisher_diag.max().item()),
            "raw_cost_adv_std": float(projection_batch["cost_terms_adv_raw"].std().item()),
            "norm_cost_adv_std": float(projection_batch["cost_terms_adv_norm"].std().item()),
            "velocity_estimation": predictor_metrics["mean_reconstruction"],
            "teacher_latent_reg_coef": predictor_metrics["teacher_latent_reg_coef"],
            "teacher_latent_reg_row_ratio": predictor_metrics["teacher_latent_reg_row_ratio"],
            "latent_alignment": latent_alignment_loss,
        }
        if self._cts_enabled():
            self.train_metrics["student_group_ratio"] = self.student_group_ratio
        self.train_metrics.update(self._constraint_diagnostic_metrics(projection_state))

        self.storage.clear()
        return {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": predictor_metrics["mean_surrogate"],
            "entropy": predictor_metrics["mean_entropy"],
            "velocity_estimation": predictor_metrics["mean_reconstruction"],
            "teacher_latent_reg": predictor_metrics["mean_teacher_latent_reg"],
            "teacher_latent_reg_weighted": predictor_metrics["mean_teacher_latent_reg_weighted"],
            "latent_alignment": latent_alignment_loss,
        }

    def _build_projection_state(
        self,
        projection_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        d_limits = projection_batch["d_limits"]
        j_cost, sigma_j = self._estimate_constraint_cost_stats(
            projection_batch["cost_terms_rollout"]
        )
        num_constraints = int(j_cost.numel())
        param_count = self._actor_param_vector().numel()

        # Decide which constraints are near binding. Inactive constraints skip
        # the expensive Fisher / per-constraint gradient computation. Their
        # rows in ``a_mat`` stay zero so projection ignores them automatically.
        if num_constraints == 0:
            active_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
        elif self.active_constraint_threshold <= 0.0:
            active_mask = torch.ones(num_constraints, dtype=torch.bool, device=self.device)
        else:
            denom = d_limits.clamp_min(self.projection_eps)
            relevance = (j_cost + self.cost_confidence * sigma_j) / denom
            active_mask = relevance >= self.active_constraint_threshold

        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        num_active = int(active_indices.numel())

        a_mat = torch.zeros(param_count, num_constraints, device=self.device)
        sigma_a = torch.zeros(num_constraints, device=self.device)

        if num_active == 0 or num_constraints == 0:
            fisher_diag = torch.full(
                (param_count,), self.fisher_min_diag, device=self.device
            )
        else:
            fisher_diag = self._estimate_fisher_diagonal(projection_batch)
            a_active = self._compute_constraint_gradients(
                projection_batch,
                advantage_key=self.constraint_advantage_key,
                constraint_columns=active_indices,
            )
            if a_active.numel() > 0:
                a_mat[:, active_indices] = a_active
            sigma_active = self._estimate_gradient_uncertainty(
                projection_batch,
                a_active,
                fisher_diag,
                active_indices=active_indices,
            )
            if sigma_active.numel() > 0:
                sigma_a[active_indices] = sigma_active

        if self.max_sigma_a is not None:
            sigma_a = torch.clamp(sigma_a, max=self.max_sigma_a)
        margins = self._compute_robust_margin(
            sigma_j=sigma_j,
            sigma_a=sigma_a,
            j_cost=j_cost,
            d_limits=d_limits,
        )
        budgets = d_limits - j_cost - margins
        return {
            "a_mat": a_mat,
            "constraint_grad_norms": (
                torch.linalg.vector_norm(a_mat, dim=0)
                if a_mat.numel() > 0
                else torch.zeros(0, device=self.device)
            ),
            "fisher_diag": fisher_diag,
            "sigma_j": sigma_j,
            "sigma_a": sigma_a,
            "margins": margins,
            "budgets": budgets,
            "j_cost": j_cost,
            "d_limits": d_limits,
            "active_mask": active_mask,
            "active_count": float(num_active),
        }

    def _compute_robust_margin(
        self,
        sigma_j: torch.Tensor,
        sigma_a: torch.Tensor,
        j_cost: torch.Tensor | None = None,
        d_limits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        radius = math.sqrt(max(2.0 * self._target_kl(), 0.0))
        margins = (
            self.cost_confidence * sigma_j
            + self.gradient_confidence * sigma_a * radius
            + 0.5 * self.curvature_proxy * (radius**2)
        )
        if self.max_margin_abs is not None:
            margins = torch.clamp(margins, max=self.max_margin_abs)
        if self.max_margin_ratio is not None and j_cost is not None and d_limits is not None:
            slack = torch.clamp(d_limits - j_cost, min=self.projection_eps)
            margins = torch.minimum(margins, slack * self.max_margin_ratio)
        return margins

    def _target_kl(self) -> float:
        if self.desired_kl is None:
            return 0.01
        return max(float(self.desired_kl), 1.0e-8)

    def _constraint_diagnostic_metrics(
        self,
        projection_state: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        j_cost = projection_state["j_cost"]
        d_limits = projection_state["d_limits"]
        margins = projection_state["margins"]
        budgets = projection_state["budgets"]
        sigma_j = projection_state["sigma_j"]
        sigma_a = projection_state["sigma_a"]

        names = list(self.constraint_names) if self.constraint_names is not None else []
        metrics: dict[str, float] = {}
        for idx in range(int(j_cost.numel())):
            name = names[idx] if idx < len(names) else f"constraint_{idx}"
            name = str(name).replace(" ", "_")
            cost = float(j_cost[idx].item())
            limit = float(d_limits[idx].item())
            denom = max(abs(limit), 1.0e-8)
            metrics[f"constraint_cost/{name}"] = cost
            metrics[f"constraint_limit/{name}"] = limit
            metrics[f"constraint_ratio/{name}"] = cost / denom
            metrics[f"constraint_margin/{name}"] = limit - cost
            metrics[f"constraint_robust_margin/{name}"] = float(margins[idx].item())
            metrics[f"constraint_robust_budget/{name}"] = float(budgets[idx].item())
            metrics[f"constraint_sigma_j/{name}"] = float(sigma_j[idx].item())
            metrics[f"constraint_sigma_a/{name}"] = float(sigma_a[idx].item())
        return metrics

    def _run_ppo_predictor(self) -> dict[str, float]:
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        mean_reconstruction_loss = 0.0
        mean_teacher_latent_reg = 0.0
        mean_teacher_latent_reg_weighted = 0.0
        mean_teacher_latent_reg_row_ratio = 0.0
        kl_checks = 0
        updates = 0
        planned_updates = max(int(self.num_learning_epochs) * int(self.num_mini_batches), 1)
        stop_triggered = False
        teacher_latent_reg_coef = self._roa_teacher_reg_coefficient()

        generator = self._mini_batch_generator()
        for (
            obs_batch,
            critic_obs_batch,
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
            actor_is_student_batch = _extra_batch[4] if len(_extra_batch) > 4 else None
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
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

            (
                _current_actions,
                actions_log_prob_batch,
                mu_batch,
                sigma_batch,
                entropy_batch,
            ) = self._evaluate_actor_batch(
                obs_batch,
                actions=actions_batch,
                actor_is_student=actor_is_student_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )

            with torch.inference_mode():
                kl_mean = self._all_reduce_mean(
                    torch.mean(self._safe_kl(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch))
                )
            mean_kl += kl_mean.item()
            kl_checks += 1

            if self.predictor_adaptive_lr and torch.isfinite(kl_mean).item():
                if kl_mean.item() > self.predictor_kl_target * 1.5:
                    self._set_predictor_learning_rate(
                        max(self.predictor_lr_min, self.predictor_lr / 1.5)
                    )
                elif 0.0 < kl_mean.item() < self.predictor_kl_target * 0.5:
                    self._set_predictor_learning_rate(
                        min(self.predictor_lr_max, self.predictor_lr * 1.25)
                    )

            if torch.isfinite(kl_mean).item() and kl_mean.item() > self.predictor_kl_hard_limit:
                self._set_predictor_learning_rate(
                    max(self.predictor_lr_min, self.predictor_lr / 2.0)
                )
                stop_triggered = True
                break

            ratio = self._safe_ratio(actions_log_prob_batch, old_actions_log_prob_batch).reshape(-1)
            surrogate = -advantages_batch.reshape(-1) * ratio
            surrogate_clipped = -advantages_batch.reshape(-1) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            reconstruction_loss, weighted_reconstruction_loss = self._velocity_estimation_loss(
                obs_batch, critic_obs_batch
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
                + weighted_reconstruction_loss
                + weighted_teacher_latent_reg_loss
                - self.entropy_coef * entropy_batch.mean()
            )
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
            mean_reconstruction_loss += reconstruction_loss.item()
            mean_teacher_latent_reg += teacher_latent_reg_loss.item()
            mean_teacher_latent_reg_weighted += weighted_teacher_latent_reg_loss.item()
            mean_teacher_latent_reg_row_ratio += teacher_reg_rows / float(
                max(obs_batch.shape[0], 1)
            )

        denom = max(updates, 1)
        skipped_updates = planned_updates - updates
        if stop_triggered:
            skipped_updates = max(skipped_updates, 1)
        return {
            "mean_surrogate": mean_surrogate_loss / denom,
            "mean_entropy": mean_entropy / denom,
            "mean_kl": mean_kl / max(kl_checks, 1),
            "mean_reconstruction": mean_reconstruction_loss / denom,
            "mean_teacher_latent_reg": mean_teacher_latent_reg / denom,
            "mean_teacher_latent_reg_weighted": mean_teacher_latent_reg_weighted / denom,
            "teacher_latent_reg_coef": teacher_latent_reg_coef,
            "teacher_latent_reg_row_ratio": mean_teacher_latent_reg_row_ratio / denom,
            "updates": float(updates),
            "update_ratio": float(updates / planned_updates),
            "stop_rate": float(skipped_updates / planned_updates),
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
            cost_terms_ret, _cost_terms_adv, cost_terms_val = self._prepare_cost_term_batches(
                cost_returns_batch=cost_returns_batch,
                cost_advantages_batch=cost_advantages_batch,
                cost_term_returns_batch=cost_term_returns_batch,
                cost_term_advantages_batch=cost_term_advantages_batch,
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
                cost_value_clipped = old_cost_terms + (pred_cost_terms - old_cost_terms).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
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

    def _prepare_projection_batch(self) -> dict[str, torch.Tensor]:
        (
            obs_batch,
            _critic_obs_batch,
            actions_batch,
            _target_values_batch,
            _advantages_batch,
            returns_batch,
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
        cost_term_values_batch = extra_batch[3] if len(extra_batch) > 3 else None
        actor_is_student_batch = extra_batch[4] if len(extra_batch) > 4 else None

        cost_returns_batch = self._sanitize_tensor(
            cost_returns_batch,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        returns_batch = self._sanitize_tensor(
            returns_batch,
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
        cost_terms_ret, cost_terms_adv_raw, cost_terms_val = self._prepare_cost_term_batches(
            cost_returns_batch=cost_returns_batch,
            cost_advantages_batch=cost_advantages_batch,
            cost_term_returns_batch=cost_term_returns_batch,
            cost_term_advantages_batch=cost_term_advantages_batch,
            cost_term_values_batch=cost_term_values_batch,
        )
        cost_terms_adv_norm = self._normalize_constraint_advantages(cost_terms_adv_raw)
        cost_terms_rollout = self._constraint_rollout_matrix()
        d_limits = self._resolve_constraint_limits(
            cost_terms_rollout.shape[1], device=cost_terms_rollout.device
        ).to(dtype=cost_terms_rollout.dtype)
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
            "actor_is_student": actor_is_student_batch,
            "cost_terms_ret": cost_terms_ret,
            "cost_terms_adv_raw": cost_terms_adv_raw,
            "cost_terms_adv_norm": cost_terms_adv_norm,
            "cost_terms_rollout": cost_terms_rollout,
            "cost_terms_val": cost_terms_val,
            "d_limits": d_limits,
            "reward_returns": returns_batch,
        }

    def _constraint_rollout_matrix(self) -> torch.Tensor:
        rollout_tensor = getattr(self.storage, "cost_term_returns", None)
        if rollout_tensor is None:
            rollout_tensor = getattr(self.storage, "cost_term_rewards", None)
        if rollout_tensor is None:
            rollout_tensor = getattr(self.storage, "cost_returns", None)
        if rollout_tensor is None:
            return torch.zeros((self.storage.num_envs, 0), device=self.device)
        rollout_tensor = self._sanitize_tensor(
            rollout_tensor.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        if rollout_tensor.ndim == 2:
            rollout_tensor = rollout_tensor.unsqueeze(-1)
        return rollout_tensor.mean(dim=0)

    def _estimate_constraint_cost_stats(
        self,
        env_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_costs.numel() == 0:
            zeros = torch.zeros(0, device=self.device)
            return zeros, zeros
        j_cost = self._all_reduce_mean(env_costs.mean(dim=0))
        if env_costs.shape[0] <= 1:
            sigma_j = torch.zeros_like(j_cost)
        else:
            sigma_j = env_costs.std(dim=0, unbiased=False) / math.sqrt(float(env_costs.shape[0]))
            sigma_j = self._all_reduce_mean(sigma_j)
        sigma_j = self._sanitize_tensor(
            sigma_j,
            nan=0.0,
            posinf=1.0e6,
            neginf=0.0,
            clamp=1.0e6,
        )
        return j_cost, sigma_j

    def _compute_constraint_gradients(
        self,
        projection_batch: dict[str, torch.Tensor],
        indices: torch.Tensor | None = None,
        advantage_key: str = "cost_terms_adv_raw",
        constraint_columns: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cost_terms_adv = self._select_rows(projection_batch[advantage_key], indices)
        if cost_terms_adv is None or cost_terms_adv.numel() == 0:
            return torch.zeros((self._actor_param_vector().numel(), 0), device=self.device)
        if constraint_columns is not None:
            if constraint_columns.numel() == 0:
                return torch.zeros(
                    (self._actor_param_vector().numel(), 0), device=self.device
                )
            cost_terms_adv = cost_terms_adv.index_select(-1, constraint_columns.to(cost_terms_adv.device))

        obs = self._select_rows(projection_batch["obs"], indices)
        actions = self._select_rows(projection_batch["actions"], indices)
        old_logp = self._select_rows(projection_batch["old_logp"], indices)
        masks = self._select_rows(projection_batch["masks"], indices)
        actor_is_student = self._select_rows(projection_batch.get("actor_is_student"), indices)
        cost_terms_adv = cost_terms_adv.reshape(-1, cost_terms_adv.shape[-1])
        num_constraints = cost_terms_adv.shape[1]
        total_count = max(int(cost_terms_adv.shape[0]), 1)
        grad_accumulator = torch.zeros(
            self._actor_param_vector().numel(),
            num_constraints,
            device=self.device,
        )

        for chunk_indices in self._constraint_grad_chunks(total_count, cost_terms_adv.device):
            chunk_obs = obs.index_select(0, chunk_indices)
            chunk_actions = actions.index_select(0, chunk_indices)
            chunk_old_logp = old_logp.index_select(0, chunk_indices)
            chunk_masks = self._select_rows(masks, chunk_indices)
            chunk_actor_is_student = self._select_rows(actor_is_student, chunk_indices)
            chunk_cost_terms_adv = cost_terms_adv.index_select(0, chunk_indices)
            chunk_weight = float(chunk_indices.numel()) / float(total_count)

            (
                _current_actions,
                actions_log_prob_batch,
                _mu,
                _sigma,
                _entropy,
            ) = self._evaluate_actor_batch(
                chunk_obs,
                actions=chunk_actions,
                actor_is_student=chunk_actor_is_student,
                masks=chunk_masks,
                hidden_states=projection_batch["hid_actor"],
            )
            ratio = self._safe_ratio(actions_log_prob_batch, chunk_old_logp).reshape(-1)

            for idx in range(num_constraints):
                cost_surrogate = torch.mean(ratio * chunk_cost_terms_adv[:, idx])
                grads = torch.autograd.grad(
                    cost_surrogate,
                    self._actor_params,
                    retain_graph=idx < (num_constraints - 1),
                    allow_unused=True,
                )
                flat_grads = self._flatten_tensors(
                    [
                        grad.detach() if grad is not None else torch.zeros_like(param)
                        for grad, param in zip(grads, self._actor_params)
                    ]
                )
                grad_accumulator[:, idx] += flat_grads * chunk_weight

        a_mat = grad_accumulator
        if self.is_multi_gpu:
            a_mat = self._all_reduce_mean(a_mat)
        return a_mat

    def _estimate_fisher_diagonal(
        self,
        projection_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        obs = projection_batch["obs"]
        actions = projection_batch["actions"]
        masks = projection_batch["masks"]
        fisher_diag = torch.zeros(self._actor_param_vector().numel(), device=self.device)

        for indices in self._chunk_index_tensors(obs.shape[0], self.fisher_num_chunks, obs.device):
            chunk_obs = obs.index_select(0, indices)
            chunk_actions = actions.index_select(0, indices)
            chunk_masks = self._select_rows(masks, indices)
            chunk_actor_is_student = self._select_rows(
                projection_batch.get("actor_is_student"),
                indices,
            )
            (
                _current_actions,
                log_prob_batch,
                _mu,
                _sigma,
                _entropy,
            ) = self._evaluate_actor_batch(
                chunk_obs,
                actions=chunk_actions,
                actor_is_student=chunk_actor_is_student,
                masks=chunk_masks,
                hidden_states=projection_batch["hid_actor"],
            )
            log_prob = log_prob_batch.mean()
            grads = torch.autograd.grad(
                log_prob,
                self._actor_params,
                allow_unused=True,
            )
            flat_grads = self._flatten_tensors(
                [
                    grad.detach() if grad is not None else torch.zeros_like(param)
                    for grad, param in zip(grads, self._actor_params)
                ]
            )
            fisher_diag += flat_grads.pow(2) * (indices.numel() / max(obs.shape[0], 1))

        fisher_diag = self._sanitize_tensor(
            fisher_diag,
            nan=0.0,
            posinf=1.0e6,
            neginf=0.0,
            clamp=1.0e6,
        )
        fisher_diag = self._all_reduce_mean(fisher_diag)
        fisher_diag = torch.clamp(fisher_diag + self.fisher_damping, min=self.fisher_min_diag)
        return fisher_diag

    def _estimate_gradient_uncertainty(
        self,
        projection_batch: dict[str, torch.Tensor],
        a_mat: torch.Tensor,
        fisher_diag: torch.Tensor,
        active_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if a_mat.numel() == 0 or self.gradient_uncertainty_mode == "disabled":
            return torch.zeros(a_mat.shape[1], device=self.device)

        # Cache must be invalidated whenever the active constraint set changes,
        # because shard variances are tracked per active column only.
        cached_indices = self._sigma_a_active_indices
        cache_size_match = (
            self._sigma_a_cache is not None and self._sigma_a_cache.numel() == a_mat.shape[1]
        )
        if active_indices is None:
            cache_indices_match = cache_size_match
        else:
            cache_indices_match = (
                cached_indices is not None
                and cached_indices.numel() == active_indices.numel()
                and torch.equal(cached_indices.to(active_indices.device), active_indices)
            )
        cache_valid = cache_size_match and cache_indices_match
        refresh_now = (
            not cache_valid
            or self.gradient_uncertainty_mode != "shards"
            or self._update_counter == 1
            or (self._update_counter - 1) % self.uncertainty_update_interval == 0
        )
        if not refresh_now:
            self._uncertainty_cache_age += 1
            return self._sigma_a_cache.to(device=self.device)

        shard_grads = []
        for indices in self._chunk_index_tensors(
            projection_batch["obs"].shape[0],
            self.gradient_uncertainty_shards,
            projection_batch["obs"].device,
        ):
            shard_grads.append(
                self._compute_constraint_gradients(
                    projection_batch,
                    indices=indices,
                    advantage_key=self.constraint_advantage_key,
                    constraint_columns=active_indices,
                )
            )
        if len(shard_grads) <= 1:
            sigma_current = torch.zeros(a_mat.shape[1], device=self.device)
        else:
            shard_grad_stack = torch.stack(shard_grads, dim=0)
            diff = shard_grad_stack - a_mat.unsqueeze(0)
            inv_metric = torch.reciprocal(fisher_diag.clamp_min(self.projection_eps))
            dual_norm = torch.sqrt(
                torch.sum(diff.pow(2) * inv_metric.view(1, -1, 1), dim=1).clamp_min(0.0)
            )
            sigma_current = torch.sqrt(torch.mean(dual_norm.pow(2), dim=0))

        sigma_current = self._sanitize_tensor(
            sigma_current,
            nan=0.0,
            posinf=1.0e6,
            neginf=0.0,
            clamp=1.0e6,
        )
        if cache_valid:
            sigma_current = (
                self.uncertainty_ema_decay * self._sigma_a_cache.to(self.device)
                + (1.0 - self.uncertainty_ema_decay) * sigma_current
            )
        self._sigma_a_cache = sigma_current.detach().cpu()
        self._sigma_a_active_indices = (
            active_indices.detach().clone().cpu() if active_indices is not None else None
        )
        self._uncertainty_cache_age = 0
        return sigma_current

    def _run_projection_corrector(
        self,
        projection_batch: dict[str, torch.Tensor],
        projection_state: dict[str, torch.Tensor],
        theta_anchor: torch.Tensor,
        theta_predictor: torch.Tensor,
    ) -> dict[str, float]:
        delta_nom = theta_predictor - theta_anchor
        delta_proj, active_constraints, predicted_violation_max, qp_condition = self._project_step(
            delta_nom=delta_nom,
            a_mat=projection_state["a_mat"],
            budgets=projection_state["budgets"],
            fisher_diag=projection_state["fisher_diag"],
        )
        delta_proj, projection_radius_scale = self._apply_projection_radius_cap(
            delta_proj,
            projection_state["fisher_diag"],
        )

        accepted_factor = 0.0
        accepted_step = 0
        accepted_delta = torch.zeros_like(delta_proj)
        used_projected = False
        used_predictor_fallback = False
        final_kl = self._evaluate_candidate_kl(projection_batch, theta_anchor)
        for step in range(self.max_backtracks + 1):
            eta = self.backtrack_coeff**step
            theta_candidate = theta_anchor + eta * delta_proj
            kl_value = self._evaluate_candidate_kl(projection_batch, theta_candidate)
            if torch.isfinite(kl_value).item() and kl_value.item() <= self._target_kl():
                self._set_actor_param_vector(theta_candidate)
                accepted_factor = eta
                accepted_step = step
                accepted_delta = eta * delta_proj
                used_projected = True
                final_kl = kl_value
                break

        # If the projected (constraint-corrected) step never satisfies the KL
        # trust region, fall back to the un-projected predictor step. The
        # predictor itself enforced ``predictor_kl_hard_limit`` during its
        # update so backtracks here are typically light.
        if not used_projected:
            for step in range(self.max_backtracks + 1):
                eta = self.backtrack_coeff**step
                theta_candidate = theta_anchor + eta * delta_nom
                kl_value = self._evaluate_candidate_kl(projection_batch, theta_candidate)
                if torch.isfinite(kl_value).item() and kl_value.item() <= self._target_kl():
                    self._set_actor_param_vector(theta_candidate)
                    accepted_factor = eta
                    accepted_step = step
                    accepted_delta = eta * delta_nom
                    used_predictor_fallback = True
                    final_kl = kl_value
                    break

        if accepted_factor == 0.0:
            self._set_actor_param_vector(theta_anchor)
            final_kl = self._evaluate_candidate_kl(projection_batch, theta_anchor)

        any_accept = used_projected or used_predictor_fallback
        return {
            "accept_rate": float(any_accept),
            "corrector_accept_rate": float(used_projected),
            "fallback_to_predictor": float(used_predictor_fallback),
            "accepted_backtrack_factor": float(accepted_factor),
            "backtrack_steps": float(accepted_step),
            "final_kl": float(final_kl.item()),
            "active_constraints": float(active_constraints),
            "predicted_violation_max": float(predicted_violation_max),
            "qp_condition": float(qp_condition),
            "projection_radius_scale": float(projection_radius_scale),
            "nominal_step_norm": float(torch.linalg.vector_norm(delta_nom).item()),
            "projected_step_norm": float(torch.linalg.vector_norm(delta_proj).item()),
            "accepted_step_norm": float(torch.linalg.vector_norm(accepted_delta).item()),
        }

    def _project_step(
        self,
        delta_nom: torch.Tensor,
        a_mat: torch.Tensor,
        budgets: torch.Tensor,
        fisher_diag: torch.Tensor,
    ) -> tuple[torch.Tensor, int, float, float]:
        if a_mat.numel() == 0:
            return delta_nom, 0, 0.0, 1.0

        violation = a_mat.transpose(0, 1).matmul(delta_nom) - budgets
        active_mask = violation > 0.0
        predicted_violation = float(torch.clamp(violation.max(), min=0.0).item())
        if not torch.any(active_mask):
            return delta_nom, 0, predicted_violation, 1.0

        a_active = a_mat[:, active_mask]
        v_active = violation[active_mask]
        inv_metric = torch.reciprocal(fisher_diag.clamp_min(self.projection_eps))

        if a_active.shape[1] == 1:
            direction = inv_metric * a_active[:, 0]
            denom = torch.dot(a_active[:, 0], direction).clamp_min(self.projection_eps)
            multiplier = torch.clamp(v_active[0] / denom, min=0.0)
            return delta_nom - multiplier * direction, 1, predicted_violation, 1.0

        weighted_a = inv_metric.unsqueeze(1) * a_active
        q_mat = a_active.transpose(0, 1).matmul(weighted_a)
        lamb, qp_condition = self._solve_nonnegative_qp(q_mat, v_active)
        delta_proj = delta_nom - weighted_a.matmul(lamb)
        return delta_proj, int(active_mask.sum().item()), predicted_violation, qp_condition

    def _solve_nonnegative_qp(
        self,
        q_mat: torch.Tensor,
        v_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if q_mat.numel() == 0:
            return torch.zeros_like(v_vec), 1.0

        eye = torch.eye(q_mat.shape[0], device=q_mat.device, dtype=q_mat.dtype)
        q_reg = q_mat + self.projection_eps * eye
        eigvals = torch.linalg.eigvalsh(q_reg)
        min_eig = float(torch.clamp(eigvals.min(), min=self.projection_eps).item())
        max_eig = float(torch.clamp(eigvals.max(), min=self.projection_eps).item())
        condition = max_eig / min_eig

        if q_reg.shape[0] <= self.exact_qp_max_constraints:
            exact = self._solve_small_nonnegative_qp_exact(q_reg, v_vec)
            if exact is not None:
                return exact, condition

        step_size = 1.0 / max_eig
        lamb = torch.zeros_like(v_vec)
        for _ in range(self.qp_max_iters):
            grad = q_reg.matmul(lamb) - v_vec
            next_lamb = torch.clamp(lamb - step_size * grad, min=0.0)
            if torch.max(torch.abs(next_lamb - lamb)) <= self.qp_tol:
                lamb = next_lamb
                break
            lamb = next_lamb
        return lamb, condition

    def _solve_small_nonnegative_qp_exact(
        self,
        q_mat: torch.Tensor,
        v_vec: torch.Tensor,
    ) -> torch.Tensor | None:
        num_constraints = int(v_vec.numel())
        best_lambda: torch.Tensor | None = None
        best_objective: float | None = None
        tol = max(self.qp_tol * 10.0, 1.0e-8)

        for subset_size in range(1, num_constraints + 1):
            for subset in itertools.combinations(range(num_constraints), subset_size):
                idx = torch.as_tensor(subset, device=q_mat.device, dtype=torch.long)
                q_sub = q_mat.index_select(0, idx).index_select(1, idx)
                v_sub = v_vec.index_select(0, idx)
                try:
                    lambda_sub = torch.linalg.solve(q_sub, v_sub)
                except RuntimeError:
                    lambda_sub = torch.linalg.lstsq(q_sub, v_sub.unsqueeze(-1)).solution.squeeze(-1)
                if torch.any(lambda_sub <= tol):
                    continue

                candidate = torch.zeros_like(v_vec)
                candidate.index_copy_(0, idx, lambda_sub)
                grad = q_mat.matmul(candidate) - v_vec
                if torch.any(grad < -tol):
                    continue
                if torch.max(torch.abs(grad.index_select(0, idx))) > max(tol, 1.0e-5):
                    continue

                objective = float(
                    (0.5 * candidate.dot(q_mat.matmul(candidate)) - v_vec.dot(candidate)).item()
                )
                if best_objective is None or objective < best_objective:
                    best_objective = objective
                    best_lambda = candidate

        return best_lambda

    def _apply_projection_radius_cap(
        self,
        delta_proj: torch.Tensor,
        fisher_diag: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if self.projection_radius_cap is None or self.projection_radius_cap <= 0.0:
            return delta_proj, 1.0
        if self.projection_radius_mode == "none":
            return delta_proj, 1.0

        if self.projection_radius_mode == "kl":
            radius = self.projection_radius_cap * math.sqrt(2.0 * self._target_kl())
        elif self.projection_radius_mode == "absolute":
            radius = self.projection_radius_cap
        else:
            return delta_proj, 1.0

        metric_norm = torch.sqrt(torch.sum(delta_proj.pow(2) * fisher_diag).clamp_min(0.0))
        if not torch.isfinite(metric_norm).item() or metric_norm.item() <= radius:
            return delta_proj, 1.0
        scale = radius / max(metric_norm.item(), self.projection_eps)
        return delta_proj * scale, float(scale)

    def _evaluate_candidate_kl(
        self,
        projection_batch: dict[str, torch.Tensor],
        theta_candidate: torch.Tensor,
    ) -> torch.Tensor:
        self._set_actor_param_vector(theta_candidate)
        with torch.inference_mode():
            (
                _current_actions,
                _log_prob,
                mu_batch,
                sigma_batch,
                _entropy,
            ) = self._evaluate_actor_batch(
                projection_batch["obs"],
                actions=projection_batch["actions"],
                actor_is_student=projection_batch.get("actor_is_student"),
                masks=projection_batch["masks"],
                hidden_states=projection_batch["hid_actor"],
            )
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

    def _normalize_constraint_advantages(self, cost_terms_adv: torch.Tensor) -> torch.Tensor:
        if cost_terms_adv.numel() == 0:
            return cost_terms_adv
        mean = cost_terms_adv.mean(dim=0, keepdim=True)
        std = cost_terms_adv.std(dim=0, unbiased=False, keepdim=True)
        normalized = (cost_terms_adv - mean) / (std + 1.0e-8)
        return self._sanitize_tensor(
            normalized,
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
            clamp=1.0e3,
        )

    def _resolve_constraint_limits(
        self,
        num_constraints: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.constraint_limits is None:
            return torch.full((num_constraints,), float(self.cost_limit), device=device)
        limits = self.constraint_limits.to(device=device, dtype=torch.float32)
        if limits.numel() == 1:
            return limits.expand(num_constraints)
        if limits.numel() != num_constraints:
            return torch.full((num_constraints,), float(limits.flatten()[0].item()), device=device)
        return limits.flatten()

    @staticmethod
    def _select_rows(
        tensor: torch.Tensor | None,
        indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if tensor is None or indices is None:
            return tensor
        return tensor.index_select(0, indices)

    @staticmethod
    def _chunk_index_tensors(
        total: int,
        num_chunks: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if total <= 0:
            return []
        all_indices = torch.arange(total, device=device)
        return [
            chunk
            for chunk in torch.tensor_split(all_indices, min(num_chunks, total))
            if chunk.numel() > 0
        ]

    def _constraint_grad_chunks(
        self,
        total: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        # Constraint gradients build a full actor autograd graph; chunking them
        # keeps the CoordConv teacher path from exhausting GPU memory on large
        # CTS batches while preserving the exact batch-average gradient.
        target_chunk_size = 4096
        num_chunks = max(
            int(math.ceil(float(total) / float(target_chunk_size))),
            int(getattr(self, "fisher_num_chunks", 1) or 1),
            1,
        )
        return self._chunk_index_tensors(total, num_chunks, device)

    def _set_predictor_learning_rate(self, learning_rate: float) -> None:
        self.predictor_lr = float(learning_rate)
        self.learning_rate = self.predictor_lr
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.predictor_lr

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "predictor_lr": self.predictor_lr,
                "sigma_a_cache": (
                    self._sigma_a_cache.detach().clone()
                    if self._sigma_a_cache is not None
                    else None
                ),
                "sigma_a_active_indices": (
                    self._sigma_a_active_indices.detach().clone()
                    if self._sigma_a_active_indices is not None
                    else None
                ),
                "uncertainty_cache_age": self._uncertainty_cache_age,
                "update_counter": self._update_counter,
            }
        )
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        super().load_state_dict(state_dict)
        predictor_lr = state_dict.get("predictor_lr")
        if predictor_lr is not None:
            self._set_predictor_learning_rate(float(predictor_lr))
        sigma_a_cache = state_dict.get("sigma_a_cache")
        if sigma_a_cache is not None:
            if not torch.is_tensor(sigma_a_cache):
                sigma_a_cache = torch.as_tensor(sigma_a_cache, dtype=torch.float32)
            self._sigma_a_cache = sigma_a_cache.detach().clone().cpu()
        sigma_a_active_indices = state_dict.get("sigma_a_active_indices")
        if sigma_a_active_indices is not None:
            if not torch.is_tensor(sigma_a_active_indices):
                sigma_a_active_indices = torch.as_tensor(
                    sigma_a_active_indices, dtype=torch.long
                )
            self._sigma_a_active_indices = sigma_a_active_indices.detach().clone().cpu()
        uncertainty_cache_age = state_dict.get("uncertainty_cache_age")
        if uncertainty_cache_age is not None:
            self._uncertainty_cache_age = int(uncertainty_cache_age)
        update_counter = state_dict.get("update_counter")
        if update_counter is not None:
            self._update_counter = int(update_counter)

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
