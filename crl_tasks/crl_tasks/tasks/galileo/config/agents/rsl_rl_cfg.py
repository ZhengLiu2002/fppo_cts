"""RSL-RL policy and algorithm configuration for Galileo CRL tasks."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CRLRslRlBaseCfg:
    """Shared observation layout for FPPO-style policies."""

    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
    # + joint_pos(12) + joint_vel(12) + actions(12) + commands(3)
    # + height_scan(182) = 230
    num_priv_hurdles: int = 0
    num_priv_explicit: int = 0
    num_priv_latent: int = 0
    history_latent_dim: int = 0
    history_reconstruction_dim: int = 0
    velocity_estimation_dim: int = 0
    velocity_estimator_channel_size: int = 0
    num_prop: int = 180
    num_scan: int = 0
    num_hist: int = 0


@configclass
class CRLRslRlStateHistEncoderCfg(CRLRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder"
    channel_size: int = 10


@configclass
class CRLRslRlActorCfg(CRLRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: CRLRslRlStateHistEncoderCfg = MISSING


@configclass
class CRLRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticRMA"
    num_cost_heads: int = 1
    num_prop: int = MISSING
    num_scan: int = 0
    num_priv_explicit: int = 0
    num_priv_latent: int = 0
    history_latent_dim: int = 0
    history_reconstruction_mode: str = "hidden_privileged"
    num_hist: int = 0
    normalize_latent: bool = False
    tanh_encoder_output: bool = False
    scan_encoder_type: str = "mlp"
    scan_grid_shape: tuple[int, int] | None = None
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    priv_terrain_scan_start: int = 0
    priv_terrain_scan_dim: int = 0
    priv_terrain_scan_shape: tuple[int, int] | None = None
    priv_terrain_encoder_output_dim: int | None = None
    cost_critic_hidden_dims: list[int] | None = None
    encode_scan_for_critic: bool = False
    critic_use_latent: bool = False
    critic_scan_encoder_dims: list[int] | None = None
    critic_num_prop: int | None = None
    critic_num_scan: int | None = None
    critic_num_priv_explicit: int | None = None
    critic_num_priv_latent: int | None = None
    critic_num_hist: int | None = None
    actor: CRLRslRlActorCfg = MISSING


@configclass
class CRLRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPO"
    velocity_estimation_loss_coef: float = 0.05
    # CTS framework controls shared by all optimizer variants.
    student_group_ratio: float = 0.25
    reconstruction_learning_rate: float = 3e-4
    num_reconstruction_epochs: int = 1
    detach_student_encoder_during_rl: bool = True
    roa_teacher_reg_coef_start: float = 0.0
    roa_teacher_reg_coef_end: float = 0.05
    roa_teacher_reg_warmup_updates: int = 5000
    roa_teacher_reg_ramp_updates: int = 5000
    roa_teacher_reg_scope: str = "teacher"
    roa_teacher_reg_loss: str = "mse"

    # Constrained RL extensions shared by non-FPPO algorithms.
    cost_value_loss_coef: float = 1.0
    cost_gamma: float | None = None
    cost_lam: float | None = None
    cost_limit: float = 0.0
    lagrangian_multiplier_init: float = 0.0
    lagrange_optimizer: str = "Adam"
    lagrange_lr: float = 1e-2
    lagrange_max: float = 100.0
    backtrack_coeff: float = 0.5
    max_backtracks: int = 10
    constraint_limits: dict[str, float] | list[float] | None = None
    normalize_cost_advantage: bool = False
    k_decay: float = 1.0
    k_min: float = 0.0
    k_violation_threshold: float = 0.02
    focops_eta: float | None = None
    focops_lambda: float | None = None
    cg_iters: int = 10
    cg_damping: float = 1e-2
    fvp_sample_freq: int = 1
    # NP3O-style positive-part cost shaping (shared across constrained algorithms)
    cost_viol_loss_coef: float = 0.0
    k_value: float = 1.0
    k_growth: float = 1.0
    k_max: float = 1.0


@configclass
class CRLRslRlFppoAlgorithmCfg(CRLRslRlPpoAlgorithmCfg):
    class_name: str = "FPPO"
    velocity_estimation_loss_coef: float = 0.05

    cost_value_loss_coef: float = 1.0
    cost_gamma: float | None = None
    cost_lam: float | None = None
    cost_limit: float = 0.0
    backtrack_coeff: float = 0.5
    max_backtracks: int = 10
    projection_eps: float = 1e-8
    fisher_damping: float = 1e-3
    fisher_num_chunks: int = 4
    fisher_min_diag: float = 1e-6
    cost_confidence: float = 1.0
    gradient_confidence: float = 0.2
    curvature_proxy: float = 0.0
    gradient_uncertainty_mode: str = "shards"
    gradient_uncertainty_shards: int = 4
    uncertainty_update_interval: int = 4
    uncertainty_ema_decay: float = 0.9
    predictor_kl_target: float | None = None
    predictor_kl_hard_limit: float | None = None
    predictor_adaptive_lr: bool = True
    predictor_lr_min: float | None = None
    predictor_lr_max: float | None = None
    qp_max_iters: int = 64
    qp_tol: float = 1e-6
    exact_qp_max_constraints: int = 8
    max_sigma_a: float | None = 2.0
    max_margin_abs: float | None = None
    max_margin_ratio: float | None = 0.5
    projection_radius_cap: float | None = 1.0
    projection_radius_mode: str = "kl"
    active_constraint_threshold: float = 0.0
    constraint_advantage_key: str = "cost_terms_adv_norm"
    constraint_limits: dict[str, float] | list[float] | None = None


@configclass
class CRLConstraintAdapterCfg:
    enabled: bool = False
    ema_beta: float = 0.9
    min_scale: float = 1e-3
    max_scale: float = 10.0
    clip: float = 5.0
    huber_delta: float = 0.2
    agg_tau: float = 0.5
    scale_by_gamma: bool = True
    cost_scale: float | None = None


@configclass
class CRLRslRlCTSAlgorithmCfg(CRLRslRlPpoAlgorithmCfg):
    class_name: str = "CTS"


@configclass
class CRLRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: CRLRslRlPpoActorCriticCfg = MISSING
    algorithm: CRLRslRlPpoAlgorithmCfg | CRLRslRlFppoAlgorithmCfg | CRLRslRlCTSAlgorithmCfg = (
        MISSING
    )
    constraint_adapter: CRLConstraintAdapterCfg = CRLConstraintAdapterCfg()
    framework_type: str | None = None
    force_student_history_rollout: bool = False


__all__ = [
    "CRLRslRlBaseCfg",
    "CRLRslRlStateHistEncoderCfg",
    "CRLRslRlActorCfg",
    "CRLRslRlPpoActorCriticCfg",
    "CRLRslRlPpoAlgorithmCfg",
    "CRLRslRlFppoAlgorithmCfg",
    "CRLConstraintAdapterCfg",
    "CRLRslRlCTSAlgorithmCfg",
    "CRLRslRlOnPolicyRunnerCfg",
]
