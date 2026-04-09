# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from .ppo import PPO


class NP3O(PPO):
    """Neural Positive-Projection PPO style constrained update."""

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
        # NP3O parameters
        cost_limit=0.0,
        cost_viol_loss_coef=1.0,
        k_value=0.05,
        k_growth=1.0004,
        k_max=1.0,
        k_decay=1.0,
        k_min=0.0,
        k_violation_threshold=0.02,
        reconstruction_loss_coef: float = 0.0,
        cost_ratio_clip: float | None = None,
        log_ratio_clip: float = 6.0,
        kl_hard_ratio: float = 4.0,
        kl_hard_abs: float = 0.05,
        constraint_limits: list[float] | tuple[float, ...] | None = None,
        # Distributed training parameters
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
            reconstruction_loss_coef=reconstruction_loss_coef,
            cost_ratio_clip=cost_ratio_clip,
            log_ratio_clip=log_ratio_clip,
            kl_hard_ratio=kl_hard_ratio,
            kl_hard_abs=kl_hard_abs,
            constraint_limits=constraint_limits,
            multi_gpu_cfg=multi_gpu_cfg,
        )

    def update(self):  # noqa: C901
        return super().update()
