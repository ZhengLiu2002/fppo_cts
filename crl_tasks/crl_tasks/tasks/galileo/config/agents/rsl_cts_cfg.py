"""CTS-only benchmark runner configuration for Galileo CRL tasks."""

from __future__ import annotations

import inspect

from isaaclab.utils import configclass
from scripts.rsl_rl.algorithms.registry import get_algorithm_spec

from ..defaults import GalileoDefaults
from .rsl_rl_cfg import (
    CRLConstraintAdapterCfg,
    CRLRslRlActorCfg,
    CRLRslRlCTSAlgorithmCfg,
    CRLRslRlFppoAlgorithmCfg,
    CRLRslRlOnPolicyRunnerCfg,
    CRLRslRlPpoActorCriticCfg,
    CRLRslRlPpoAlgorithmCfg,
    CRLRslRlStateHistEncoderCfg,
)


def _filter_supported_cfg_params(cfg_cls, params: dict) -> dict:
    try:
        sig = inspect.signature(cfg_cls.__init__)
    except (TypeError, ValueError):
        return dict(params)
    allowed = {name for name in sig.parameters if name != "self"}
    return {key: value for key, value in params.items() if key in allowed}


_ALGORITHM_CFG_FAMILIES = {
    "ppo": CRLRslRlPpoAlgorithmCfg,
    "fppo": CRLRslRlFppoAlgorithmCfg,
    "cts": CRLRslRlCTSAlgorithmCfg,
}


def build_cts_policy_cfg() -> CRLRslRlPpoActorCriticCfg:
    return CRLRslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=1.0,
        activation="elu",
        num_prop=GalileoDefaults.obs.cts.actor_num_prop,
        num_scan=GalileoDefaults.obs.cts.actor_num_scan,
        num_priv_explicit=GalileoDefaults.obs.cts.actor_num_priv_explicit,
        num_priv_latent=GalileoDefaults.obs.cts.actor_num_priv_latent,
        history_latent_dim=GalileoDefaults.obs.cts.actor_history_latent_dim,
        num_hist=GalileoDefaults.obs.cts.actor_num_hist,
        normalize_latent=True,
        tanh_encoder_output=False,
        scan_encoder_dims=[],
        priv_encoder_dims=[256, 128],
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        cost_critic_hidden_dims=[512, 512, 256, 128],
        encode_scan_for_critic=True,
        critic_use_latent=False,
        critic_num_prop=GalileoDefaults.obs.cts.critic_num_prop,
        critic_num_scan=GalileoDefaults.obs.cts.critic_num_scan,
        critic_num_priv_explicit=GalileoDefaults.obs.cts.critic_num_priv_explicit,
        critic_num_priv_latent=GalileoDefaults.obs.cts.critic_num_priv_latent,
        critic_num_hist=GalileoDefaults.obs.cts.critic_num_hist,
        critic_scan_encoder_dims=[128, 64, 32],
        actor=CRLRslRlActorCfg(
            class_name="Actor",
            num_prop=GalileoDefaults.obs.cts.actor_num_prop,
            num_scan=GalileoDefaults.obs.cts.actor_num_scan,
            num_priv_explicit=GalileoDefaults.obs.cts.actor_num_priv_explicit,
            num_priv_latent=GalileoDefaults.obs.cts.actor_num_priv_latent,
            history_latent_dim=GalileoDefaults.obs.cts.actor_history_latent_dim,
            history_reconstruction_dim=0,
            velocity_estimation_dim=3,
            velocity_estimator_channel_size=64,
            num_hist=GalileoDefaults.obs.cts.actor_num_hist,
            state_history_encoder=CRLRslRlStateHistEncoderCfg(
                class_name="TCNHistoryEncoder",
                num_prop=GalileoDefaults.obs.cts.actor_num_prop,
                history_latent_dim=GalileoDefaults.obs.cts.actor_history_latent_dim,
                num_hist=GalileoDefaults.obs.cts.actor_num_hist,
                channel_size=64,
            ),
        ),
    )


def build_cts_algorithm_cfg(
    algo_key: str | None = None,
) -> CRLRslRlPpoAlgorithmCfg | CRLRslRlFppoAlgorithmCfg | CRLRslRlCTSAlgorithmCfg:
    algorithm_defaults = GalileoDefaults.algorithm
    selected_algo = (algo_key or getattr(algorithm_defaults, "name", "fppo")).strip().lower()
    alg_spec = get_algorithm_spec(selected_algo)
    cfg_cls = _ALGORITHM_CFG_FAMILIES.get(alg_spec.config_family)
    if cfg_cls is None:
        raise ValueError(
            f"Unsupported CTS config family '{alg_spec.config_family}' for algorithm '{selected_algo}'."
        )

    params: dict = {}
    params.update(getattr(algorithm_defaults, "base", {}))
    params.update(getattr(algorithm_defaults, "per_algo", {}).get(selected_algo, {}))
    return cfg_cls(
        class_name=alg_spec.class_name,
        **_filter_supported_cfg_params(cfg_cls, params),
    )


def build_cts_constraint_adapter_cfg(algo_key: str | None = None) -> CRLConstraintAdapterCfg:
    algorithm_defaults = GalileoDefaults.algorithm
    selected_algo = (algo_key or getattr(algorithm_defaults, "name", "fppo")).strip().lower()
    params: dict = {}
    params.update(getattr(algorithm_defaults, "constraint_adapter_base", {}))
    params.update(
        getattr(algorithm_defaults, "constraint_adapter_per_algo", {}).get(selected_algo, {})
    )
    return CRLConstraintAdapterCfg(**params)


@configclass
class GalileoCTSBenchmarkRunnerCfg(CRLRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 100000
    save_interval = 100
    experiment_name = "galileo_cts"
    empirical_normalization = False
    framework_type = "cts"
    force_student_history_rollout = True
    policy = build_cts_policy_cfg()
    algorithm = build_cts_algorithm_cfg()
    constraint_adapter = build_cts_constraint_adapter_cfg()


__all__ = [
    "GalileoCTSBenchmarkRunnerCfg",
    "build_cts_algorithm_cfg",
    "build_cts_constraint_adapter_cfg",
    "build_cts_policy_cfg",
]
