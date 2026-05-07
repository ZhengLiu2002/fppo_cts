"""Observation layout helpers for training/export consistency.

This module keeps the runtime observation manager, policy actor dimensions, and
deployment export metadata tied together.  The intent mirrors the compact
``ConfigSummary`` style used by the upstream Galileo training stack: record the
small set of layout facts that must agree, and fail early when they do not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__name__"):
        return value.__name__
    return str(value)


def stable_hash(value: Any, *, length: int = 16) -> str:
    payload = json.dumps(_to_serializable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _cfg_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _shape_dim(shape: Any) -> int:
    if shape is None:
        return 0
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, Sequence):
        dim = 1
        for item in shape:
            dim *= int(item)
        return int(dim)
    try:
        return int(shape)
    except Exception:
        return 0


def _normalize_scale(value: Any) -> Any:
    if value is None:
        return 1.0
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    try:
        return float(value)
    except Exception:
        return _to_serializable(value)


def _term_scale(term_cfg: Any) -> Any:
    scale = _cfg_attr(term_cfg, "scale")
    if scale is None:
        params = _cfg_attr(term_cfg, "params", {})
        if isinstance(params, Mapping) and "scale" in params:
            scale = params["scale"]
    return _normalize_scale(scale)


def _term_func_name(term_cfg: Any) -> str | None:
    func = _cfg_attr(term_cfg, "func")
    if func is None:
        return None
    return getattr(func, "__name__", str(func))


def _resolve_obs_manager(env: Any) -> Any | None:
    unwrapped = getattr(env, "unwrapped", env)
    return getattr(unwrapped, "observation_manager", None)


def extract_observation_layout(env: Any) -> dict[str, Any]:
    """Extract a compact layout from an Isaac Lab observation manager."""

    obs_mgr = _resolve_obs_manager(env)
    if obs_mgr is None:
        return {"groups": {}, "layout_hash": stable_hash({"groups": {}})}

    group_names = list(getattr(obs_mgr, "group_obs_dim", {}) or {})
    if not group_names:
        group_names = list(getattr(obs_mgr, "_group_obs_term_names", {}) or {})

    groups: dict[str, Any] = {}
    active_terms = getattr(obs_mgr, "active_terms", {}) or {}
    group_term_names_map = getattr(obs_mgr, "_group_obs_term_names", {}) or {}
    group_term_cfgs_map = getattr(obs_mgr, "_group_obs_term_cfgs", {}) or {}
    group_term_dims_map = getattr(obs_mgr, "group_obs_term_dim", {}) or {}
    group_dims_map = getattr(obs_mgr, "group_obs_dim", {}) or {}
    concat_map = getattr(obs_mgr, "group_obs_concatenate", {}) or {}

    for group_name in group_names:
        term_names = list(group_term_names_map.get(group_name) or active_terms.get(group_name) or [])
        term_cfgs = list(group_term_cfgs_map.get(group_name) or [])
        term_dims = list(group_term_dims_map.get(group_name) or [])
        terms = []
        for index, term_name in enumerate(term_names):
            term_cfg = term_cfgs[index] if index < len(term_cfgs) else None
            shape = term_dims[index] if index < len(term_dims) else None
            terms.append(
                {
                    "name": str(term_name),
                    "shape": _to_serializable(shape),
                    "dim": _shape_dim(shape),
                    "scale": _term_scale(term_cfg),
                    "func": _term_func_name(term_cfg),
                }
            )

        group_dim = group_dims_map.get(group_name)
        groups[str(group_name)] = {
            "dim": _shape_dim(group_dim),
            "shape": _to_serializable(group_dim),
            "concatenate": bool(concat_map.get(group_name, True)),
            "terms": terms,
        }

    payload = {"groups": groups}
    payload["layout_hash"] = stable_hash(payload)
    return payload


def _actor_layout(actor_critic: Any | None) -> dict[str, Any]:
    actor = getattr(actor_critic, "actor", actor_critic)
    if actor is None:
        return {}
    keys = (
        "in_features",
        "student_in_features",
        "num_prop",
        "num_scan",
        "num_hist",
        "num_priv_explicit",
        "num_priv_latent",
        "history_latent_dim",
        "num_actions",
    )
    layout = {"class_name": type(actor).__name__}
    for key in keys:
        value = getattr(actor, key, None)
        if value is not None:
            try:
                layout[key] = int(value)
            except Exception:
                layout[key] = _to_serializable(value)
    return layout


def build_export_layout_signature(policy_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deployment-facing part of a policy export config."""

    input_names = list(policy_cfg.get("input_names") or [])
    input_obs_size_map = dict(policy_cfg.get("input_obs_size_map") or {})
    obs_history_length = dict(policy_cfg.get("obs_history_length") or {})
    export_input_dims = dict(policy_cfg.get("export_input_dims") or {})
    if not export_input_dims:
        export_input_dims = {
            name: int(input_obs_size_map[name]) * int(obs_history_length.get(name, 1))
            for name in input_names
            if name in input_obs_size_map
        }
    return {
        "input_names": input_names,
        "input_obs_names_map": _to_serializable(policy_cfg.get("input_obs_names_map") or {}),
        "input_obs_scales_map": _to_serializable(policy_cfg.get("input_obs_scales_map") or {}),
        "input_obs_size_map": _to_serializable(input_obs_size_map),
        "obs_history_length": _to_serializable(obs_history_length),
        "export_input_dims": _to_serializable(export_input_dims),
        "export_input_order": list(policy_cfg.get("export_input_order") or input_names),
        "joint_names": list(policy_cfg.get("joint_names") or []),
    }


def _git_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]

    def _run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = _run_git(["rev-parse", "HEAD"])
    status = _run_git(["status", "--short"])
    return {
        "repo_root": str(repo_root),
        "git_commit": commit,
        "git_dirty": bool(status),
    }


def add_policy_layout_metadata(
    policy_cfg: dict[str, Any],
    *,
    env: Any | None = None,
    env_cfg: Any | None = None,
    agent_cfg: Any | None = None,
    actor_critic: Any | None = None,
) -> dict[str, Any]:
    """Attach stable layout metadata to a policy export dictionary."""

    export_layout = build_export_layout_signature(policy_cfg)
    observation_layout = extract_observation_layout(env) if env is not None else None
    actor_layout = _actor_layout(actor_critic)
    config_summary = _cfg_attr(env_cfg, "config_summary")
    metadata = {
        **_git_metadata(),
        "env_class": type(getattr(env, "unwrapped", env)).__name__ if env is not None else None,
        "env_cfg_class": type(env_cfg).__name__ if env_cfg is not None else None,
        "agent_cfg_class": type(agent_cfg).__name__ if agent_cfg is not None else None,
        "config_summary": getattr(config_summary, "__name__", type(config_summary).__name__)
        if config_summary is not None
        else None,
        "actor": actor_layout,
    }
    layout_payload = {
        "export": export_layout,
        "observation_layout_hash": observation_layout.get("layout_hash")
        if isinstance(observation_layout, Mapping)
        else None,
        "actor": actor_layout,
    }
    policy_cfg["export_input_dims"] = export_layout["export_input_dims"]
    policy_cfg["layout_hash"] = stable_hash(layout_payload)
    policy_cfg["observation_layout_hash"] = (
        observation_layout.get("layout_hash") if isinstance(observation_layout, Mapping) else None
    )
    if observation_layout is not None:
        policy_cfg["observation_layout"] = observation_layout
    policy_cfg["runtime_metadata"] = metadata
    return policy_cfg


def validate_policy_layout(
    policy_cfg: Mapping[str, Any],
    *,
    env: Any | None = None,
    actor_critic: Any | None = None,
    strict: bool = True,
) -> list[str]:
    """Validate the runtime policy layout against env and actor dimensions.

    Returns a list of human-readable issues.  When ``strict`` is true, a
    ``RuntimeError`` is raised if any issue is found.
    """

    issues: list[str] = []
    input_names = list(policy_cfg.get("input_names") or [])
    size_map = dict(policy_cfg.get("input_obs_size_map") or {})
    history_map = dict(policy_cfg.get("obs_history_length") or {})
    export_dims = dict(policy_cfg.get("export_input_dims") or {})

    for input_name in input_names:
        if input_name not in size_map:
            issues.append(f"Missing input_obs_size_map entry for '{input_name}'.")
            continue
        size = int(size_map[input_name])
        history = int(history_map.get(input_name, 1))
        expected_dim = size * history
        actual_dim = int(export_dims.get(input_name, expected_dim))
        if actual_dim != expected_dim:
            issues.append(
                f"Export dim mismatch for '{input_name}': {actual_dim} != {size} * {history}."
            )

    export_total = sum(int(export_dims.get(name, 0)) for name in input_names)
    actor = getattr(actor_critic, "actor", actor_critic)
    if actor is not None:
        actor_in = getattr(actor, "in_features", None)
        student_in = getattr(actor, "student_in_features", None)
        accepted = {int(v) for v in (actor_in, student_in) if v is not None and int(v) > 0}
        if accepted and export_total not in accepted:
            issues.append(
                f"Export input total {export_total} does not match actor dims {sorted(accepted)}."
            )

    if env is not None and actor is not None:
        obs_layout = extract_observation_layout(env)
        policy_dim = _cfg_attr(_cfg_attr(obs_layout, "groups", {}), "policy", {}).get("dim")
        actor_in = getattr(actor, "in_features", None)
        student_in = getattr(actor, "student_in_features", None)
        accepted = {int(v) for v in (actor_in, student_in) if v is not None and int(v) > 0}
        if policy_dim and accepted and int(policy_dim) not in accepted:
            issues.append(
                f"Runtime policy obs dim {policy_dim} does not match actor dims {sorted(accepted)}."
            )

    if strict and issues:
        joined = "\n".join(f"- {issue}" for issue in issues)
        raise RuntimeError(f"Policy observation layout validation failed:\n{joined}")
    return issues
