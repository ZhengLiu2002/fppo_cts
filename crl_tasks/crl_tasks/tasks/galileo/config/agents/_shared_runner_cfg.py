"""Shared helpers for Galileo RSL-RL runner configs."""

from __future__ import annotations

import inspect
from typing import Literal

from ..defaults import GalileoDefaults
from ..symmetry import build_symmetry_cfg
from .rsl_rl_cfg import (
    CRLConstraintAdapterCfg,
    CRLRslRlActorCfg,
    CRLRslRlDAggerAlgorithmCfg,
    CRLRslRlFppoAlgorithmCfg,
    CRLRslRlPpoActorCriticCfg,
    CRLRslRlPpoAlgorithmCfg,
    CRLRslRlStateHistEncoderCfg,
)

RunnerRole = Literal["teacher", "student"]

_ACTOR_HIDDEN_DIMS = [512, 512, 256, 128]
_CRITIC_HIDDEN_DIMS = [512, 512, 256, 128]
_SCAN_ENCODER_DIMS = [128, 64, 32]


def _filter_supported_cfg_params(cfg_cls, params: dict) -> dict:
    """Drop merged defaults that the target config class does not accept."""
    try:
        sig = inspect.signature(cfg_cls.__init__)
    except (TypeError, ValueError):
        return dict(params)
    allowed = {name for name in sig.parameters if name != "self"}
    return {key: value for key, value in params.items() if key in allowed}


def _merged_constraint_adapter_params(algo_key: str, role: RunnerRole) -> dict:
    params: dict = {}
    params.update(getattr(GalileoDefaults.algorithm, "constraint_adapter_base", {}))
    params.update(getattr(GalileoDefaults.algorithm, "constraint_adapter_per_algo", {}).get(algo_key, {}))
    if role == "teacher":
        params.update(getattr(GalileoDefaults.algorithm, "constraint_adapter_teacher_override", {}))
    else:
        params.update(getattr(GalileoDefaults.algorithm, "constraint_adapter_student_override", {}))
    return params


def _merged_algorithm_params(
    role: RunnerRole,
    algo_key_override: str | None = None,
) -> tuple[str, str, dict]:
    algo_key = algo_key_override or getattr(GalileoDefaults.algorithm, "name", "fppo")
    class_name = GalileoDefaults.algorithm.class_name_map.get(algo_key, algo_key)
    params: dict = {}
    params.update(getattr(GalileoDefaults.algorithm, "base", {}))
    params.update(getattr(GalileoDefaults.algorithm, "per_algo", {}).get(algo_key, {}))
    if role == "teacher":
        params.update(getattr(GalileoDefaults.algorithm, "teacher_override", {}))
    else:
        params.update(getattr(GalileoDefaults.algorithm, "student_override", {}))
    return algo_key, class_name, params


def build_algorithm_cfg(
    role: RunnerRole,
    algo_key: str | None = None,
) -> CRLRslRlPpoAlgorithmCfg | CRLRslRlFppoAlgorithmCfg | CRLRslRlDAggerAlgorithmCfg:
    """Build algorithm config from ``GalileoDefaults`` for the given role."""
    algo_key, class_name, params = _merged_algorithm_params(role, algo_key)
    symmetry_params: dict = {}
    symmetry_params.update(getattr(GalileoDefaults.algorithm, "symmetry_base", {}))
    if role == "teacher":
        symmetry_params.update(getattr(GalileoDefaults.algorithm, "symmetry_teacher_override", {}))
    else:
        symmetry_params.update(getattr(GalileoDefaults.algorithm, "symmetry_student_override", {}))
    symmetry_cfg = build_symmetry_cfg(role, symmetry_params)
    if symmetry_cfg is not None:
        params["symmetry_cfg"] = symmetry_cfg
    if algo_key == "fppo":
        cfg_cls = CRLRslRlFppoAlgorithmCfg
    elif algo_key == "dagger":
        cfg_cls = CRLRslRlDAggerAlgorithmCfg
    else:
        cfg_cls = CRLRslRlPpoAlgorithmCfg
    return cfg_cls(class_name=class_name, **_filter_supported_cfg_params(cfg_cls, params))


def build_constraint_adapter_cfg(role: RunnerRole) -> CRLConstraintAdapterCfg:
    """Build runner-side constraint preprocessing config."""
    algo_key, _class_name, _params = _merged_algorithm_params(role)
    adapter_params = _merged_constraint_adapter_params(algo_key, role)
    return CRLConstraintAdapterCfg(**adapter_params)


def build_obs_cfg(role: RunnerRole) -> tuple[dict[str, int], dict[str, int]]:
    """Build actor/critic observation layout from ``GalileoDefaults``."""
    obs_cfg = getattr(GalileoDefaults.obs, role)
    actor_cfg = {
        "num_prop": obs_cfg.actor_num_prop,
        "num_scan": obs_cfg.actor_num_scan,
        "num_priv_explicit": obs_cfg.actor_num_priv_explicit,
        "num_priv_latent": obs_cfg.actor_num_priv_latent,
        "history_latent_dim": getattr(
            obs_cfg, "actor_history_latent_dim", obs_cfg.actor_num_priv_latent
        ),
        "history_reconstruction_dim": max(obs_cfg.critic_num_prop - obs_cfg.actor_num_prop, 0)
        + obs_cfg.critic_num_scan
        + obs_cfg.critic_num_priv_explicit
        + obs_cfg.critic_num_priv_latent,
        "num_hist": obs_cfg.actor_num_hist,
    }
    critic_cfg = {
        "critic_num_prop": obs_cfg.critic_num_prop,
        "critic_num_scan": obs_cfg.critic_num_scan,
        "critic_num_priv_explicit": obs_cfg.critic_num_priv_explicit,
        "critic_num_priv_latent": obs_cfg.critic_num_priv_latent,
        "critic_num_hist": obs_cfg.critic_num_hist,
    }
    return actor_cfg, critic_cfg


def build_policy_cfg(role: RunnerRole) -> CRLRslRlPpoActorCriticCfg:
    """Build policy config from shared defaults and observation layout."""
    actor_obs, critic_obs = build_obs_cfg(role)
    algo_key = getattr(GalileoDefaults.algorithm, "name", "fppo")
    init_noise_std = 0.5 if algo_key == "fppo" else 1.0
    return CRLRslRlPpoActorCriticCfg(
        init_noise_std=init_noise_std,
        num_prop=actor_obs["num_prop"],
        num_scan=actor_obs["num_scan"],
        num_priv_explicit=actor_obs["num_priv_explicit"],
        num_priv_latent=actor_obs["num_priv_latent"],
        history_latent_dim=actor_obs["history_latent_dim"],
        num_hist=actor_obs["num_hist"],
        encode_scan_for_critic=(critic_obs["critic_num_scan"] > 0),
        critic_num_prop=critic_obs["critic_num_prop"],
        critic_num_scan=critic_obs["critic_num_scan"],
        critic_num_priv_explicit=critic_obs["critic_num_priv_explicit"],
        critic_num_priv_latent=critic_obs["critic_num_priv_latent"],
        critic_num_hist=critic_obs["critic_num_hist"],
        critic_scan_encoder_dims=_SCAN_ENCODER_DIMS if critic_obs["critic_num_scan"] > 0 else None,
        actor_hidden_dims=_ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=_CRITIC_HIDDEN_DIMS,
        scan_encoder_dims=_SCAN_ENCODER_DIMS,
        priv_encoder_dims=[],
        activation="elu",
        actor=CRLRslRlActorCfg(
            class_name="Actor",
            num_prop=actor_obs["num_prop"],
            num_scan=actor_obs["num_scan"],
            num_priv_explicit=actor_obs["num_priv_explicit"],
            num_priv_latent=actor_obs["num_priv_latent"],
            history_latent_dim=actor_obs["history_latent_dim"],
            history_reconstruction_dim=actor_obs["history_reconstruction_dim"],
            num_hist=actor_obs["num_hist"],
            state_history_encoder=CRLRslRlStateHistEncoderCfg(
                class_name="TCNHistoryEncoder",
                num_prop=actor_obs["num_prop"],
                history_latent_dim=actor_obs["history_latent_dim"],
                num_hist=actor_obs["num_hist"],
                channel_size=max(actor_obs["history_latent_dim"], 32),
            ),
        ),
    )


__all__ = ["build_algorithm_cfg", "build_constraint_adapter_cfg", "build_obs_cfg", "build_policy_cfg"]
