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
    # + height_scan(132) = 180
    num_priv_hurdles: int = 0
    num_priv_explicit: int = 0
    num_priv_latent: int = 0
    history_latent_dim: int = 0
    history_reconstruction_dim: int = 0
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
    num_hist: int = 0
    tanh_encoder_output: bool = False
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    cost_critic_hidden_dims: list[int] | None = None
    encode_scan_for_critic: bool = False
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
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float] = [0, 0.1, 2000, 3000]
    reconstruction_loss_coef: float = 0.0
    symmetry_cfg: dict | None = None

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
class CRLRslRlFppoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "FPPO"
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float] = [0, 0.1, 2000, 3000]
    reconstruction_loss_coef: float = 0.0
    symmetry_cfg: dict | None = None

    cost_value_loss_coef: float = 1.0
    step_size: float = 1e-3
    cost_gamma: float | None = None
    cost_lam: float | None = None
    cost_limit: float = 0.0
    backtrack_coeff: float = 0.5
    max_backtracks: int = 10
    projection_eps: float = 1e-8
    epsilon_safe: float = 0.0
    delta_kl: float | None = None
    predictor_desired_kl: float | None = None
    predictor_kl_hard_limit: float | None = None
    softproj_max_iters: int = 40
    softproj_tol: float = 1e-6
    constraint_limits: dict[str, float] | list[float] | None = None
    constraint_limits_start: dict[str, float] | list[float] | None = None
    constraint_limits_final: dict[str, float] | list[float] | None = None
    adaptive_constraint_curriculum: bool = False
    constraint_curriculum_names: list[str] | None = None
    constraint_curriculum_ema_decay: float = 0.95
    constraint_curriculum_check_interval: int = 20
    constraint_curriculum_alpha: float = 0.8
    constraint_curriculum_shrink: float = 0.97
    use_clipped_surrogate: bool = True
    normalize_cost_advantage: bool = False
    step_size_adaptive: bool = True
    cost_viol_loss_coef: float = 0.0
    k_value: float = 1.0
    k_growth: float = 1.0
    k_max: float = 1.0
    k_decay: float = 1.0
    k_min: float = 0.0
    k_violation_threshold: float = 0.02


@configclass
class CRLConstraintAdapterCfg:
    enabled: bool = False
    ema_beta: float = 0.99
    min_scale: float = 1e-3
    max_scale: float = 10.0
    clip: float = 5.0
    huber_delta: float = 0.1
    agg_tau: float = 0.5
    scale_by_gamma: bool = False
    cost_scale: float | None = None


@configclass
class CRLRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "Distillation"


@configclass
class CRLRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: CRLRslRlPpoActorCriticCfg = MISSING
    algorithm: CRLRslRlPpoAlgorithmCfg | CRLRslRlFppoAlgorithmCfg | CRLRslRlDistillationAlgorithmCfg = MISSING
    constraint_adapter: CRLConstraintAdapterCfg = CRLConstraintAdapterCfg()


__all__ = [
    "CRLRslRlBaseCfg",
    "CRLRslRlStateHistEncoderCfg",
    "CRLRslRlActorCfg",
    "CRLRslRlPpoActorCriticCfg",
    "CRLRslRlPpoAlgorithmCfg",
    "CRLRslRlFppoAlgorithmCfg",
    "CRLConstraintAdapterCfg",
    "CRLRslRlDistillationAlgorithmCfg",
    "CRLRslRlOnPolicyRunnerCfg",
]
