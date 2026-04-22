# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import re

import numpy as np
import torch


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.array(
            [v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for v in value],
            dtype=object,
        )
    return np.array(value)


def _looks_like_pattern(name: str) -> bool:
    name = str(name)
    if ".*" in name:
        return True
    return any(ch in name for ch in ("*", "?", "[", "]", "|", "^", "$"))


def _infer_actor_input_dim(actor: torch.nn.Module) -> int | None:
    """Infer actor input dim by taking the maximum in_features across Linear layers.

    Some actors contain auxiliary encoders whose first Linear has smaller in_features
    (e.g., scan encoder), so returning the first Linear underestimates the true
    observation size. Using the max is safer for ONNX export.
    """
    max_in = None
    if hasattr(actor, "in_features"):
        try:
            max_in = int(getattr(actor, "in_features"))
        except Exception:
            max_in = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            val = int(module.in_features)
            if (max_in is None) or (val > max_in):
                max_in = val
    return max_in


def _call_actor(actor: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    try:
        return actor(obs, hist_encoding=True)
    except TypeError:
        try:
            return actor(obs)
        except TypeError:
            return actor(obs, True)


def _extract_action_joint_names(env_cfg):
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return None
    joint_pos_cfg = getattr(actions_cfg, "joint_pos", None)
    if joint_pos_cfg is None:
        return None
    joint_names = getattr(joint_pos_cfg, "joint_names", None)
    if not joint_names:
        return None
    if any(_looks_like_pattern(name) for name in joint_names):
        return None
    return list(joint_names)


def _resolve_obs_group(obs_mgr):
    group_names = list(getattr(obs_mgr, "group_obs_dim", {}) or {})
    if not group_names:
        group_names = list(getattr(obs_mgr, "_group_obs_term_names", {}) or {})
    for name in ("actor_obs", "policy"):
        if name in group_names:
            return name
    return group_names[0] if group_names else None


def _extract_obs_terms(env):
    obs_mgr = env.unwrapped.observation_manager
    group = _resolve_obs_group(obs_mgr)
    if group is None:
        return []
    return list(getattr(obs_mgr, "_group_obs_term_names", {}).get(group, []))


def _extract_obs_scales(env):
    obs_mgr = env.unwrapped.observation_manager
    group = _resolve_obs_group(obs_mgr)
    if group is None:
        return {}
    term_names = list(getattr(obs_mgr, "_group_obs_term_names", {}).get(group, []))
    term_cfgs = list(getattr(obs_mgr, "_group_obs_term_cfgs", {}).get(group, []))
    scales = {}
    for name, cfg in zip(term_names, term_cfgs):
        scale = getattr(cfg, "scale", None)
        if scale is None:
            params = getattr(cfg, "params", None)
            if isinstance(params, dict) and "scale" in params:
                scale = params["scale"]
        if scale is None:
            scale = 1.0
        scales[name] = scale
    return scales


def _extract_obs_term_cfgs(env):
    obs_mgr = env.unwrapped.observation_manager
    group = _resolve_obs_group(obs_mgr)
    if group is None:
        return {}
    term_names = list(getattr(obs_mgr, "_group_obs_term_names", {}).get(group, []))
    term_cfgs = list(getattr(obs_mgr, "_group_obs_term_cfgs", {}).get(group, []))
    return {name: cfg for name, cfg in zip(term_names, term_cfgs)}


def _extract_obs_terms_from_cfg(env_cfg) -> list[tuple[str, object]]:
    observations_cfg = getattr(env_cfg, "observations", None)
    policy_cfg = getattr(observations_cfg, "policy", None)
    if policy_cfg is None:
        return []

    if isinstance(policy_cfg, dict):
        items = policy_cfg.items()
    else:
        items = getattr(policy_cfg, "__dict__", {}).items()

    term_items: list[tuple[str, object]] = []
    for name, cfg in items:
        if str(name).startswith("_"):
            continue
        if hasattr(cfg, "func"):
            term_items.append((str(name), cfg))
    return term_items


def _normalize_scale(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        return [_safe_float(v, 1.0) for v in val]
    return _safe_float(val, 1.0)


def _extract_term_scale(term_cfg):
    scale = getattr(term_cfg, "scale", None)
    if scale is None:
        params = getattr(term_cfg, "params", None)
        if isinstance(params, dict) and "scale" in params:
            scale = params["scale"]
    if scale is None:
        scale = 1.0
    return scale


_HISTORY_KEY_EXPORT_CANDIDATES = {
    "base_lin_vel": ["base_lin_vel_nad", "base_lin_vel_gt", "base_lin_vel"],
    "base_ang_vel": ["base_ang_vel_nad", "base_ang_vel_gt", "base_ang_vel"],
    "projected_gravity": [
        "projected_gravity_nad",
        "projected_gravity_gt",
        "projected_gravity",
    ],
    "joint_pos": ["joint_pos_rel_nad", "joint_pos_rel_gt", "joint_pos"],
    "joint_vel": ["joint_vel_rel_nad", "joint_vel_rel_gt", "joint_vel"],
    "last_action": ["actions_gt", "actions_nad", "actions"],
    "commands": ["base_commands_gt", "base_commands_nad", "velocity_commands"],
}

_HISTORY_KEY_ORDER = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "last_action",
    "commands",
]


def _resolve_history_export_name(current_obs_names: list[str], canonical_name: str) -> str:
    candidates = _HISTORY_KEY_EXPORT_CANDIDATES.get(canonical_name, [canonical_name])
    current_set = set(current_obs_names)
    for candidate in candidates:
        if candidate in current_set:
            return candidate
    return candidates[0]


def _build_history_input_spec(
    history_cfg,
    current_obs_names: list[str],
    current_obs_scales: dict[str, float | list[float]],
    input_name: str = "proprio_history",
) -> dict | None:
    params = getattr(history_cfg, "params", None)
    if not isinstance(params, dict):
        return None

    history_length = int(params.get("history_length", 1) or 1)
    if history_length <= 0:
        return None

    include_base_lin_vel = bool(params.get("include_base_lin_vel", True))
    scale_cfg = dict(params.get("scales", {}) or {})

    obs_names: list[str] = []
    obs_scales: dict[str, float | list[float]] = {}
    for key in _HISTORY_KEY_ORDER:
        if key == "base_lin_vel" and not include_base_lin_vel:
            continue
        export_name = _resolve_history_export_name(current_obs_names, key)
        obs_names.append(export_name)
        if key in scale_cfg:
            obs_scales[export_name] = _normalize_scale(scale_cfg[key])
        else:
            obs_scales[export_name] = current_obs_scales.get(export_name, 1.0)

    return {
        "input_name": input_name,
        "obs_names": obs_names,
        "obs_scales": obs_scales,
        "history_length": history_length,
    }


def _build_export_input_layout(
    env,
    env_cfg,
    actor_critic: object | None = None,
) -> dict:
    raw_obs_names = _extract_obs_terms(env)
    obs_scales = _extract_obs_scales(env)
    obs_term_cfgs = _extract_obs_term_cfgs(env)

    summary = getattr(env.unwrapped.cfg, "config_summary", None) or getattr(
        env_cfg, "config_summary", None
    )
    obs_summary = getattr(summary, "observation", None)
    obs_name_map = getattr(obs_summary, "export_name_map", None)
    if not isinstance(obs_name_map, dict):
        obs_name_map = {}

    mapped_obs_names: list[str] = []
    mapped_obs_scales: dict[str, float | list[float]] = {}
    raw_to_mapped: dict[str, str] = {}
    for raw_name in raw_obs_names:
        mapped_name = obs_name_map.get(raw_name, raw_name)
        raw_to_mapped[raw_name] = mapped_name
        mapped_obs_names.append(mapped_name)
        mapped_scale = _normalize_scale(obs_scales.get(raw_name, 1.0))
        if mapped_name not in mapped_obs_scales or not str(raw_name).startswith("teacher_"):
            mapped_obs_scales[mapped_name] = mapped_scale

    actor_module = getattr(actor_critic, "actor", None)
    actor_num_prop = int(getattr(actor_module, "num_prop", 0) or 0)
    actor_num_hist = int(getattr(actor_module, "num_hist", 0) or 0)
    actor_num_scan = int(getattr(actor_module, "num_scan", 0) or 0)
    actor_num_priv_explicit = int(getattr(actor_module, "num_priv_explicit", 0) or 0)
    actor_num_priv_latent = int(getattr(actor_module, "num_priv_latent", 0) or 0)

    history_raw_name = next((name for name in raw_obs_names if "history" in name.lower()), None)
    history_cfg = obs_term_cfgs.get(history_raw_name) if history_raw_name is not None else None
    can_split_history = bool(
        history_raw_name is not None
        and history_cfg is not None
        and actor_num_hist > 0
        and actor_num_prop > 0
        and actor_num_scan == 0
        and actor_num_priv_explicit == 0
        and actor_num_priv_latent == 0
    )
    can_split_cts_history = bool(
        history_raw_name is not None
        and history_cfg is not None
        and actor_num_hist > 0
        and actor_num_prop > 0
        and any(name.startswith("teacher_") for name in raw_obs_names)
    )

    if can_split_history or can_split_cts_history:
        current_raw_names = [
            name
            for name in raw_obs_names
            if name != history_raw_name and not name.startswith("teacher_")
        ]
        current_obs_names = [raw_to_mapped[name] for name in current_raw_names]
        current_obs_scales = {
            raw_to_mapped[name]: _normalize_scale(obs_scales.get(name, 1.0))
            for name in current_raw_names
        }
        history_input_name = raw_to_mapped.get(history_raw_name, history_raw_name)
        history_spec = _build_history_input_spec(
            history_cfg,
            current_obs_names=current_obs_names,
            current_obs_scales=current_obs_scales,
            input_name=history_input_name,
        )
        if history_spec is None:
            raise RuntimeError(
                f"Unable to infer history export spec for observation term '{history_raw_name}'."
            )
        return {
            "input_names": ["actor_obs", history_spec["input_name"]],
            "input_obs_names_map": {
                "actor_obs": current_obs_names,
                history_spec["input_name"]: history_spec["obs_names"],
            },
            "input_obs_scales_map": {
                "actor_obs": current_obs_scales,
                history_spec["input_name"]: history_spec["obs_scales"],
            },
            "input_obs_size_map": {
                "actor_obs": actor_num_prop,
                history_spec["input_name"]: actor_num_prop,
            },
            "obs_history_length": {
                "actor_obs": 1,
                history_spec["input_name"]: int(history_spec["history_length"]),
            },
            "export_input_order": ["actor_obs", history_spec["input_name"]],
        }

    if history_raw_name is not None and actor_num_hist > 0:
        raise RuntimeError(
            "History-based policy export is only implemented for proprio-history student actors "
            "(no scan / no privileged inputs)."
        )

    actor_obs_names = mapped_obs_names
    input_actor_obs_scales = {name: mapped_obs_scales.get(name, 1.0) for name in actor_obs_names}
    obs_mgr = env.unwrapped.observation_manager
    group = _resolve_obs_group(obs_mgr)
    group_dims = getattr(obs_mgr, "group_obs_dim", {})
    actor_obs_dim = None
    if group is not None and group in group_dims:
        dim = group_dims[group]
        actor_obs_dim = int(dim[0] if isinstance(dim, (list, tuple)) else dim)
    if actor_obs_dim is None or actor_obs_dim <= 0:
        actor_obs_dim = int(actor_num_prop or len(actor_obs_names))

    return {
        "input_names": ["actor_obs"],
        "input_obs_names_map": {"actor_obs": actor_obs_names},
        "input_obs_scales_map": {"actor_obs": input_actor_obs_scales},
        "input_obs_size_map": {"actor_obs": int(actor_obs_dim)},
        "obs_history_length": {"actor_obs": 1},
        "export_input_order": ["actor_obs"],
    }


def _build_export_input_layout_from_cfg(
    env_cfg,
    actor_critic: object | None = None,
) -> dict | None:
    term_items = _extract_obs_terms_from_cfg(env_cfg)
    if not term_items:
        return None

    raw_obs_names = [name for name, _cfg in term_items]
    obs_scales = {name: _extract_term_scale(cfg) for name, cfg in term_items}
    obs_term_cfgs = dict(term_items)

    summary = getattr(env_cfg, "config_summary", None)
    obs_summary = getattr(summary, "observation", None)
    obs_name_map = getattr(obs_summary, "export_name_map", None)
    if not isinstance(obs_name_map, dict):
        obs_name_map = {}

    mapped_obs_names: list[str] = []
    mapped_obs_scales: dict[str, float | list[float]] = {}
    raw_to_mapped: dict[str, str] = {}
    for raw_name in raw_obs_names:
        mapped_name = obs_name_map.get(raw_name, raw_name)
        raw_to_mapped[raw_name] = mapped_name
        mapped_obs_names.append(mapped_name)
        mapped_scale = _normalize_scale(obs_scales.get(raw_name, 1.0))
        if mapped_name not in mapped_obs_scales or not str(raw_name).startswith("teacher_"):
            mapped_obs_scales[mapped_name] = mapped_scale

    actor_module = getattr(actor_critic, "actor", None)
    actor_num_prop = int(getattr(actor_module, "num_prop", 0) or 0)
    actor_num_hist = int(getattr(actor_module, "num_hist", 0) or 0)
    actor_num_scan = int(getattr(actor_module, "num_scan", 0) or 0)
    actor_num_priv_explicit = int(getattr(actor_module, "num_priv_explicit", 0) or 0)
    actor_num_priv_latent = int(getattr(actor_module, "num_priv_latent", 0) or 0)

    history_raw_name = next((name for name in raw_obs_names if "history" in name.lower()), None)
    history_cfg = obs_term_cfgs.get(history_raw_name) if history_raw_name is not None else None
    can_split_history = bool(
        history_raw_name is not None
        and history_cfg is not None
        and actor_num_hist > 0
        and actor_num_prop > 0
        and actor_num_scan == 0
        and actor_num_priv_explicit == 0
        and actor_num_priv_latent == 0
    )
    can_split_cts_history = bool(
        history_raw_name is not None
        and history_cfg is not None
        and actor_num_hist > 0
        and actor_num_prop > 0
        and any(name.startswith("teacher_") for name in raw_obs_names)
    )

    if can_split_history or can_split_cts_history:
        current_raw_names = [
            name
            for name in raw_obs_names
            if name != history_raw_name and not name.startswith("teacher_")
        ]
        current_obs_names = [raw_to_mapped[name] for name in current_raw_names]
        current_obs_scales = {
            raw_to_mapped[name]: _normalize_scale(obs_scales.get(name, 1.0))
            for name in current_raw_names
        }
        history_input_name = raw_to_mapped.get(history_raw_name, history_raw_name)
        history_spec = _build_history_input_spec(
            history_cfg,
            current_obs_names=current_obs_names,
            current_obs_scales=current_obs_scales,
            input_name=history_input_name,
        )
        if history_spec is None:
            raise RuntimeError(
                f"Unable to infer history export spec for observation term '{history_raw_name}'."
            )
        return {
            "input_names": ["actor_obs", history_spec["input_name"]],
            "input_obs_names_map": {
                "actor_obs": current_obs_names,
                history_spec["input_name"]: history_spec["obs_names"],
            },
            "input_obs_scales_map": {
                "actor_obs": current_obs_scales,
                history_spec["input_name"]: history_spec["obs_scales"],
            },
            "input_obs_size_map": {
                "actor_obs": actor_num_prop,
                history_spec["input_name"]: actor_num_prop,
            },
            "obs_history_length": {
                "actor_obs": 1,
                history_spec["input_name"]: int(history_spec["history_length"]),
            },
            "export_input_order": ["actor_obs", history_spec["input_name"]],
        }

    if history_raw_name is not None and actor_num_hist > 0:
        raise RuntimeError(
            "History-based policy export is only implemented for proprio-history student actors "
            "(no scan / no privileged inputs)."
        )

    actor_obs_names = mapped_obs_names
    input_actor_obs_scales = {name: mapped_obs_scales.get(name, 1.0) for name in actor_obs_names}
    actor_obs_dim = int(
        getattr(actor_module, "student_in_features", 0)
        or getattr(actor_module, "in_features", 0)
        or actor_num_prop
        or len(actor_obs_names)
    )
    return {
        "input_names": ["actor_obs"],
        "input_obs_names_map": {"actor_obs": actor_obs_names},
        "input_obs_scales_map": {"actor_obs": input_actor_obs_scales},
        "input_obs_size_map": {"actor_obs": actor_obs_dim},
        "obs_history_length": {"actor_obs": 1},
        "export_input_order": ["actor_obs"],
    }


def _synchronize_policy_export_cfg(policy_cfg_dict: dict) -> dict:
    input_names = list(policy_cfg_dict.get("input_names") or [])
    if not input_names:
        input_names = ["actor_obs"]
        policy_cfg_dict["input_names"] = input_names

    input_obs_names_map = copy.deepcopy(policy_cfg_dict.get("input_obs_names_map") or {})
    if not input_obs_names_map and "input_actor_obs_names" in policy_cfg_dict:
        input_obs_names_map["actor_obs"] = copy.deepcopy(policy_cfg_dict["input_actor_obs_names"])
    policy_cfg_dict["input_obs_names_map"] = input_obs_names_map

    input_obs_scales_map = copy.deepcopy(policy_cfg_dict.get("input_obs_scales_map") or {})
    if not input_obs_scales_map and "input_actor_obs_scales" in policy_cfg_dict:
        input_obs_scales_map["actor_obs"] = copy.deepcopy(policy_cfg_dict["input_actor_obs_scales"])
    policy_cfg_dict["input_obs_scales_map"] = input_obs_scales_map

    input_obs_size_map = copy.deepcopy(policy_cfg_dict.get("input_obs_size_map") or {})
    policy_cfg_dict["input_obs_size_map"] = input_obs_size_map

    obs_history_length = copy.deepcopy(policy_cfg_dict.get("obs_history_length") or {})
    if not obs_history_length:
        obs_history_length = {name: 1 for name in input_names}
    policy_cfg_dict["obs_history_length"] = obs_history_length

    if "output_names" not in policy_cfg_dict or not policy_cfg_dict["output_names"]:
        policy_cfg_dict["output_names"] = ["actions"]

    export_input_dims = copy.deepcopy(policy_cfg_dict.get("export_input_dims") or {})
    if not export_input_dims:
        export_input_dims = {
            name: int(input_obs_size_map[name] * obs_history_length.get(name, 1))
            for name in input_names
            if name in input_obs_size_map
        }
    policy_cfg_dict["export_input_dims"] = export_input_dims

    policy_cfg_dict["input_actor_obs_names"] = copy.deepcopy(
        policy_cfg_dict["input_obs_names_map"].get("actor_obs", [])
    )
    policy_cfg_dict["input_actor_obs_scales"] = copy.deepcopy(
        policy_cfg_dict["input_obs_scales_map"].get("actor_obs", {})
    )
    policy_cfg_dict["export_input_order"] = list(
        policy_cfg_dict.get("export_input_order") or input_names
    )
    return policy_cfg_dict


def _infer_command_max_velocity(command_cfg):
    explicit_max_vel = getattr(command_cfg, "max_velocity", None)
    if explicit_max_vel is not None:
        return [float(v) for v in explicit_max_vel]
    max_vel = [0.0, 0.0, 0.0]
    for rng in _iter_command_range_cfgs(command_cfg):
        if isinstance(rng, dict):
            lin_x = rng.get("lin_vel_x")
            lin_y = rng.get("lin_vel_y")
            ang_z = rng.get("ang_vel_z")
        else:
            lin_x = getattr(rng, "lin_vel_x", None)
            lin_y = getattr(rng, "lin_vel_y", None)
            ang_z = getattr(rng, "ang_vel_z", None)
        if lin_x is not None:
            max_vel[0] = max(max_vel[0], max(abs(lin_x[0]), abs(lin_x[1])))
        if lin_y is not None:
            max_vel[1] = max(max_vel[1], max(abs(lin_y[0]), abs(lin_y[1])))
        if ang_z is not None:
            max_vel[2] = max(max_vel[2], max(abs(ang_z[0]), abs(ang_z[1])))
    max_lin_x_level = getattr(command_cfg, "max_lin_x_level", None)
    max_ang_z_level = getattr(command_cfg, "max_ang_z_level", None)
    if max_lin_x_level is not None:
        max_vel[0] = max(max_vel[0], abs(max_lin_x_level))
    if max_ang_z_level is not None:
        max_vel[2] = max(max_vel[2], abs(max_ang_z_level))
    return [float(v if v > 0.0 else 1.0) for v in max_vel]


def _iter_command_range_cfgs(command_cfg):
    ranges = getattr(command_cfg, "ranges", None)
    if ranges is None:
        return []
    if isinstance(ranges, dict):
        if any(key in ranges for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            return [ranges]
        return list(ranges.values())
    return [ranges]


def _infer_command_direction_scales(command_cfg):
    if command_cfg is None:
        return 1.0, 1.0, 1.0, 1.0

    forward = getattr(command_cfg, "velocity_x_forward_scale", None)
    backward = getattr(command_cfg, "velocity_x_backward_scale", None)
    lateral = getattr(command_cfg, "velocity_y_scale", None)
    yaw = getattr(command_cfg, "velocity_yaw_scale", None)
    if None not in (forward, backward, lateral, yaw):
        return (
            _safe_float(forward, 1.0),
            _safe_float(backward, 1.0),
            _safe_float(lateral, 1.0),
            _safe_float(yaw, 1.0),
        )

    inferred_forward = 0.0
    inferred_backward = 0.0
    inferred_lateral = 0.0
    inferred_yaw = 0.0
    for rng in _iter_command_range_cfgs(command_cfg):
        if isinstance(rng, dict):
            lin_x = rng.get("lin_vel_x")
            lin_y = rng.get("lin_vel_y")
            ang_z = rng.get("ang_vel_z")
        else:
            lin_x = getattr(rng, "lin_vel_x", None)
            lin_y = getattr(rng, "lin_vel_y", None)
            ang_z = getattr(rng, "ang_vel_z", None)
        if lin_x is not None:
            inferred_forward = max(inferred_forward, max(0.0, _safe_float(lin_x[1], 0.0)))
            inferred_backward = max(inferred_backward, abs(min(0.0, _safe_float(lin_x[0], 0.0))))
        if lin_y is not None:
            inferred_lateral = max(
                inferred_lateral,
                abs(_safe_float(lin_y[0], 0.0)),
                abs(_safe_float(lin_y[1], 0.0)),
            )
        if ang_z is not None:
            inferred_yaw = max(
                inferred_yaw,
                abs(_safe_float(ang_z[0], 0.0)),
                abs(_safe_float(ang_z[1], 0.0)),
            )

    max_lin_x_level = getattr(command_cfg, "max_lin_x_level", None)
    max_ang_z_level = getattr(command_cfg, "max_ang_z_level", None)
    if max_lin_x_level is not None:
        inferred_forward = max(inferred_forward, abs(_safe_float(max_lin_x_level, 0.0)))
    if max_ang_z_level is not None:
        inferred_yaw = max(inferred_yaw, abs(_safe_float(max_ang_z_level, 0.0)))

    return (
        _safe_float(forward, inferred_forward or 1.0),
        _safe_float(backward, inferred_backward or 1.0),
        _safe_float(lateral, inferred_lateral or 1.0),
        _safe_float(yaw, inferred_yaw or 1.0),
    )


def _resolve_joint_cfg_value(joint_cfg, joint_name):
    if isinstance(joint_cfg, dict):
        if joint_name in joint_cfg:
            return joint_cfg[joint_name]
        for pattern, value in joint_cfg.items():
            if _looks_like_pattern(pattern) and re.fullmatch(str(pattern), joint_name):
                return value
        return None
    if isinstance(joint_cfg, (list, tuple, np.ndarray)):
        return None
    return joint_cfg


def _resolve_joint_cfg_vector(joint_cfg, joint_names):
    if joint_cfg is None:
        return None
    if isinstance(joint_cfg, (list, tuple, np.ndarray)):
        if len(joint_cfg) != len(joint_names):
            return None
        return list(joint_cfg)

    values = []
    for joint_name in joint_names:
        value = _resolve_joint_cfg_value(joint_cfg, joint_name)
        if value is None:
            return None
        values.append(value)
    return values


def _extract_joint_defaults_from_cfg(env_cfg, joint_names):
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    init_state = getattr(robot_cfg, "init_state", None)
    joint_pos_cfg = getattr(init_state, "joint_pos", None)
    defaults = _resolve_joint_cfg_vector(joint_pos_cfg, joint_names)
    if defaults is None:
        return None
    return [float(f"{_safe_float(value, 0.0):.4f}") for value in defaults]


def _extract_actuator_vectors_from_cfg(
    env_cfg,
    joint_names,
    fallback_kp=90.0,
    fallback_kd=3.0,
    fallback_torque=130.0,
):
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    actuators = getattr(robot_cfg, "actuators", None)
    actuator = None
    if isinstance(actuators, dict):
        actuator = actuators.get("base_legs")
        if actuator is None and actuators:
            actuator = next(iter(actuators.values()))
    if actuator is None:
        return None

    stiffness_cfg = getattr(actuator, "stiffness", None)
    damping_cfg = getattr(actuator, "damping", None)
    effort_limit_cfg = (
        getattr(actuator, "effort_limit", None)
        or getattr(actuator, "effort_limit_sim", None)
        or getattr(actuator, "saturation_effort", None)
    )

    stiffness = _resolve_joint_cfg_vector(stiffness_cfg, joint_names)
    damping = _resolve_joint_cfg_vector(damping_cfg, joint_names)
    effort_limit = _resolve_joint_cfg_vector(effort_limit_cfg, joint_names)

    kp_list = (
        [_safe_float(value, fallback_kp) for value in stiffness]
        if stiffness is not None
        else [fallback_kp for _ in joint_names]
    )
    kd_list = (
        [_safe_float(value, fallback_kd) for value in damping]
        if damping is not None
        else [fallback_kd for _ in joint_names]
    )
    torque_list = (
        [_safe_float(value, fallback_torque) for value in effort_limit]
        if effort_limit is not None
        else [fallback_torque for _ in joint_names]
    )
    return kp_list, kd_list, torque_list


def _extract_joint_defaults(env, joint_names):
    joint_name_to_idx = {
        name: idx for idx, name in enumerate(env.unwrapped.scene.articulations["robot"].joint_names)
    }
    default_joint_pos = (
        env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()
    )
    return [float(f"{default_joint_pos[joint_name_to_idx[name]]:.4f}") for name in joint_names]


def _extract_actuator_vectors(
    env, joint_names, fallback_kp=90.0, fallback_kd=3.0, fallback_torque=130.0
):
    actuators = getattr(env.unwrapped.scene.articulations["robot"], "actuators", {})
    actuator = None
    if isinstance(actuators, dict):
        actuator = actuators.get("base_legs")
        if actuator is None and actuators:
            actuator = next(iter(actuators.values()))
    kp_list = [fallback_kp for _ in joint_names]
    kd_list = [fallback_kd for _ in joint_names]
    torque_list = [fallback_torque for _ in joint_names]
    if actuator is None:
        return kp_list, kd_list, torque_list
    act_joint_names = getattr(actuator, "joint_names", None)
    stiffness = getattr(actuator, "stiffness", None)
    damping = getattr(actuator, "damping", None)
    effort_limit = getattr(actuator, "effort_limit", None)
    if stiffness is not None:
        stiffness = _to_numpy(stiffness).reshape(-1)
    if damping is not None:
        damping = _to_numpy(damping).reshape(-1)
    if effort_limit is not None:
        effort_limit = _to_numpy(effort_limit).reshape(-1)
    if act_joint_names and stiffness is not None and len(act_joint_names) == len(stiffness):
        kp_map = {
            name: _safe_float(val, fallback_kp) for name, val in zip(act_joint_names, stiffness)
        }
        kp_list = [kp_map.get(name, fallback_kp) for name in joint_names]
    elif stiffness is not None and len(stiffness) == len(joint_names):
        kp_list = [_safe_float(val, fallback_kp) for val in stiffness]
    if act_joint_names and damping is not None and len(act_joint_names) == len(damping):
        kd_map = {
            name: _safe_float(val, fallback_kd) for name, val in zip(act_joint_names, damping)
        }
        kd_list = [kd_map.get(name, fallback_kd) for name in joint_names]
    elif damping is not None and len(damping) == len(joint_names):
        kd_list = [_safe_float(val, fallback_kd) for val in damping]
    if act_joint_names and effort_limit is not None and len(act_joint_names) == len(effort_limit):
        tq_map = {
            name: _safe_float(val, fallback_torque)
            for name, val in zip(act_joint_names, effort_limit)
        }
        torque_list = [tq_map.get(name, fallback_torque) for name in joint_names]
    elif effort_limit is not None and len(effort_limit) == len(joint_names):
        torque_list = [_safe_float(val, fallback_torque) for val in effort_limit]
    return kp_list, kd_list, torque_list


def export_policy_as_jit(
    actor_critic: object, normalizer: object | None, path: str, filename="policy.pt"
):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_policy_as_onnx_dual_input(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    actor_obs_dim: int | None = None,
    verbose: bool = False,
):
    """Export policy into an ONNX file with a single actor_obs input."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporterDualInput(
        actor_critic,
        normalizer,
        actor_obs_dim=actor_obs_dim,
        verbose=verbose,
    )
    policy_exporter.export(path, filename)


def export_policy_as_onnx_grouped_inputs(
    actor_critic: object,
    path: str,
    input_groups: dict[str, int] | list[tuple[str, int]],
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose: bool = False,
):
    """Export policy into an ONNX file with explicit input groups.

    The grouped inputs are concatenated in-order before normalizer / actor inference.
    This is useful for deployment stacks that reconstruct current proprio and
    history buffers separately but the training-time actor consumes a single
    flattened observation vector.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporterGroupedInputs(
        actor_critic,
        normalizer,
        input_groups=input_groups,
        verbose=verbose,
    )
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            )
            self.register_buffer(
                "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            )
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return _call_actor(self.actor, x)

    def forward(self, x):
        return _call_actor(self.actor, self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return _call_actor(self.actor, x), h, c

    def forward(self, x):
        return _call_actor(self.actor, self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            actor_in = _infer_actor_input_dim(self.actor)
            if actor_in is None:
                raise RuntimeError("Unable to infer actor input dimension for ONNX export.")
            obs = torch.zeros(1, actor_in)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class _OnnxPolicyExporterGroupedInputs(torch.nn.Module):
    """Exporter that exposes one or more named input groups."""

    def __init__(
        self,
        actor_critic: object,
        normalizer: object | None,
        input_groups: dict[str, int] | list[tuple[str, int]],
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        if getattr(actor_critic, "is_recurrent", False):
            raise RuntimeError("Grouped-input exporter does not support recurrent policies.")
        self.actor = copy.deepcopy(actor_critic.actor)
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        if isinstance(input_groups, dict):
            input_groups = list(input_groups.items())
        normalized_groups: list[tuple[str, int]] = []
        for name, dim in input_groups:
            try:
                group_dim = int(dim)
            except Exception as exc:
                raise RuntimeError(f"Invalid input dim for group '{name}': {dim}") from exc
            if group_dim <= 0:
                raise RuntimeError(
                    f"Invalid non-positive input dim for group '{name}': {group_dim}"
                )
            normalized_groups.append((str(name), group_dim))
        if not normalized_groups:
            raise RuntimeError("Grouped-input exporter requires at least one input group.")

        inferred = _infer_actor_input_dim(self.actor)
        student_inferred = int(getattr(self.actor, "student_in_features", 0) or 0)
        total_input_dim = sum(dim for _name, dim in normalized_groups)
        if inferred is None and total_input_dim <= 0:
            raise RuntimeError(
                "Unable to infer grouped-input export size. Please provide explicit input_groups."
            )
        if (
            inferred is not None
            and inferred != total_input_dim
            and student_inferred != total_input_dim
        ):
            raise RuntimeError(
                f"Grouped-input export dims {total_input_dim} do not match actor input dim {inferred}."
            )

        self.input_names = [name for name, _dim in normalized_groups]
        self.input_dims = [dim for _name, dim in normalized_groups]
        self.expected_obs_dim = (
            total_input_dim
            if student_inferred == total_input_dim
            else (inferred or total_input_dim)
        )

    @staticmethod
    def _pad_or_trim(obs: torch.Tensor, expected: int | None) -> torch.Tensor:
        """Ensure obs has expected feature dim without tracing-time python branching."""
        if expected is None or expected <= 0:
            return obs
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return obs[:, :expected]

        current = obs.shape[1]
        if current == expected:
            return obs
        if current > expected:
            return obs[:, :expected]
        pad = expected - current
        if pad <= 0:
            return obs
        zeros = torch.zeros(obs.shape[0], pad, device=obs.device, dtype=obs.dtype)
        return torch.cat([obs, zeros], dim=1)

    def forward(self, *inputs):
        if len(inputs) != len(self.input_dims):
            raise RuntimeError(
                f"Expected {len(self.input_dims)} grouped inputs, received {len(inputs)}."
            )
        obs_parts = [
            self._pad_or_trim(obs, expected_dim)
            for obs, expected_dim in zip(inputs, self.input_dims)
        ]
        obs = torch.cat(obs_parts, dim=1)
        return _call_actor(self.actor, self.normalizer(obs))

    def export(self, path, filename):
        self.to("cpu")
        export_inputs = tuple(torch.zeros(1, dim) for dim in self.input_dims)
        torch.onnx.export(
            self,
            export_inputs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=self.input_names,
            output_names=["actions"],
            dynamic_axes={},
        )


class _OnnxPolicyExporterDualInput(_OnnxPolicyExporterGroupedInputs):
    """Compatibility wrapper that exposes a single actor_obs input."""

    def __init__(
        self,
        actor_critic: object,
        normalizer: object | None,
        actor_obs_dim: int | None,
        verbose: bool = False,
    ):
        inferred = actor_obs_dim
        if inferred is not None:
            try:
                inferred = int(inferred)
            except Exception:
                inferred = None
        if inferred is None or inferred <= 0:
            inferred = _infer_actor_input_dim(getattr(actor_critic, "actor", actor_critic))
        if inferred is None or inferred <= 0:
            raise RuntimeError("Unable to infer actor_obs_dim for ONNX export.")
        super().__init__(
            actor_critic,
            normalizer,
            input_groups=[("actor_obs", inferred)],
            verbose=verbose,
        )


def export_inference_cfg(
    env,
    env_cfg,
    path,
    agent_cfg=None,
    actor_critic: object | None = None,
):
    policy_cfg_dict = {}
    live_env_cfg = getattr(env.unwrapped, "cfg", None) or env_cfg

    sim_cfg = getattr(live_env_cfg, "sim", None)
    decimation = getattr(live_env_cfg, "decimation", None)
    sim_dt = getattr(sim_cfg, "dt", None)
    if decimation is not None and sim_dt is not None:
        policy_cfg_dict["dt"] = _safe_float(decimation, 0.0) * _safe_float(sim_dt, 0.0)
    else:
        physics_dt = getattr(env.unwrapped, "physics_dt", None)
        if physics_dt is not None:
            policy_cfg_dict["dt"] = _safe_float(
                getattr(live_env_cfg, "decimation", 1.0), 1.0
            ) * _safe_float(physics_dt, 0.0)
        else:
            policy_cfg_dict["dt"] = (
                getattr(env.unwrapped, "step_dt", None)
                or getattr(getattr(env.unwrapped, "sim", None), "dt", None)
                or 0.0
            )

    joint_names = _extract_action_joint_names(live_env_cfg)
    if not joint_names:
        joint_names = _extract_action_joint_names(env_cfg)
    if not joint_names:
        joint_names = list(env.unwrapped.scene.articulations["robot"].joint_names)
    policy_cfg_dict["joint_names"] = joint_names
    default_joint_pos = _extract_joint_defaults_from_cfg(live_env_cfg, joint_names)
    if default_joint_pos is None:
        default_joint_pos = _extract_joint_defaults_from_cfg(env_cfg, joint_names)
    if default_joint_pos is None:
        default_joint_pos = _extract_joint_defaults(env, joint_names)
    policy_cfg_dict["default_joint_pos"] = default_joint_pos

    policy_cfg_dict["output_names"] = ["actions"]

    summary = getattr(live_env_cfg, "config_summary", None) or getattr(
        env.unwrapped.cfg, "config_summary", None
    )
    env_summary = getattr(summary, "env", None)
    action_summary = getattr(summary, "action", None)
    export_layout = _build_export_input_layout_from_cfg(live_env_cfg, actor_critic=actor_critic)
    if export_layout is None:
        export_layout = _build_export_input_layout(env, live_env_cfg, actor_critic=actor_critic)
    policy_cfg_dict["input_names"] = list(export_layout["input_names"])
    policy_cfg_dict["input_obs_names_map"] = copy.deepcopy(export_layout["input_obs_names_map"])
    policy_cfg_dict["input_obs_scales_map"] = copy.deepcopy(export_layout["input_obs_scales_map"])
    policy_cfg_dict["input_obs_size_map"] = copy.deepcopy(export_layout["input_obs_size_map"])
    policy_cfg_dict["obs_history_length"] = copy.deepcopy(export_layout["obs_history_length"])
    policy_cfg_dict["export_input_order"] = list(export_layout["export_input_order"])
    policy_cfg_dict["input_actor_obs_names"] = copy.deepcopy(
        policy_cfg_dict["input_obs_names_map"].get("actor_obs", [])
    )
    policy_cfg_dict["input_actor_obs_scales"] = copy.deepcopy(
        policy_cfg_dict["input_obs_scales_map"].get("actor_obs", {})
    )

    action_scale = getattr(action_summary, "scale", None)
    if action_scale is None:
        action_scale = getattr(getattr(live_env_cfg, "actions", None), "joint_pos", None)
        if action_scale is None:
            action_scale = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        action_scale = getattr(action_scale, "scale", 1.0)
    policy_cfg_dict["action_scale"] = _safe_float(action_scale, 1.0)

    clip_actions = getattr(env_summary, "clip_actions", None)
    if clip_actions is None and agent_cfg is not None:
        clip_actions = getattr(agent_cfg, "clip_actions", None)
    if clip_actions is None:
        clip_actions = 100.0
    policy_cfg_dict["clip_actions"] = _safe_float(clip_actions, 100.0)

    clip_obs = getattr(env_summary, "clip_obs", None)
    if clip_obs is None:
        clip_obs = 100.0
    policy_cfg_dict["clip_obs"] = _safe_float(clip_obs, 100.0)

    actuator_vectors = _extract_actuator_vectors_from_cfg(live_env_cfg, joint_names)
    if actuator_vectors is None:
        actuator_vectors = _extract_actuator_vectors_from_cfg(env_cfg, joint_names)
    if actuator_vectors is None:
        actuator_vectors = _extract_actuator_vectors(env, joint_names)
    kp, kd, max_torques = actuator_vectors
    policy_cfg_dict["joint_kp"] = [float(f"{x:.4f}") for x in kp]
    policy_cfg_dict["joint_kd"] = [float(f"{x:.4f}") for x in kd]
    policy_cfg_dict["max_torques"] = [float(f"{x:.4f}") for x in max_torques]

    command_cfg = getattr(summary, "command", None)
    if command_cfg is None:
        command_cfg = getattr(getattr(live_env_cfg, "commands", None), "base_velocity", None)
    if command_cfg is None:
        command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if command_cfg is None:
        command_cfg = getattr(env.unwrapped.cfg, "commands", None)
    (
        velocity_x_forward_scale,
        velocity_x_backward_scale,
        velocity_y_scale,
        velocity_yaw_scale,
    ) = _infer_command_direction_scales(command_cfg)
    policy_cfg_dict["velocity_x_forward_scale"] = _safe_float(velocity_x_forward_scale, 1.0)
    policy_cfg_dict["velocity_x_backward_scale"] = _safe_float(velocity_x_backward_scale, 1.0)
    policy_cfg_dict["velocity_y_scale"] = _safe_float(velocity_y_scale, 1.0)
    policy_cfg_dict["velocity_yaw_scale"] = _safe_float(velocity_yaw_scale, 1.0)
    policy_cfg_dict["max_velocity"] = (
        _infer_command_max_velocity(command_cfg) if command_cfg is not None else [1.0, 1.0, 1.0]
    )
    policy_cfg_dict["max_acceleration"] = [1.5, 1.5, 6.0]
    policy_cfg_dict["max_jerk"] = [5.0, 5.0, 30.0]
    policy_cfg_dict["threshold"] = {"limit_lower": 0.0, "limit_upper": 0.0, "damping": 5.0}
    policy_cfg_dict = _synchronize_policy_export_cfg(policy_cfg_dict)

    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_obs_names_map:", policy_cfg_dict["input_obs_names_map"])
    print("input_obs_scales_map:", policy_cfg_dict["input_obs_scales_map"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    print("max_torques:", policy_cfg_dict["max_torques"])
    print("velocity_x_forward_scale:", policy_cfg_dict["velocity_x_forward_scale"])
    print("velocity_x_backward_scale:", policy_cfg_dict["velocity_x_backward_scale"])
    print("velocity_y_scale:", policy_cfg_dict["velocity_y_scale"])
    print("velocity_yaw_scale:", policy_cfg_dict["velocity_yaw_scale"])
    print("max_velocity:", policy_cfg_dict["max_velocity"])
    print("max_acceleration:", policy_cfg_dict["max_acceleration"])
    print("max_jerk:", policy_cfg_dict["max_jerk"])
    print("threshold:", policy_cfg_dict["threshold"])
    export_inference_cfg_to_yaml(policy_cfg_dict, path)
    return policy_cfg_dict


def export_inference_cfg_to_yaml(config_dict, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    config_dict = _synchronize_policy_export_cfg(copy.deepcopy(config_dict))
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    content += "input_obs_names_map:\n  {\n"
    for input_name in config_dict["input_names"]:
        obs_names = config_dict["input_obs_names_map"].get(input_name, [])
        content += f"    {input_name}: ["
        content += ", ".join(f'"{o}"' for o in obs_names)
        content += "],\n"
    content += "  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    content += "input_obs_scales_map:\n  {\n"
    for input_name in config_dict["input_names"]:
        scales = config_dict["input_obs_scales_map"].get(input_name, {})
        obs_list = config_dict["input_obs_names_map"].get(input_name, [])
        parts = []
        for obs in obs_list:
            val = scales.get(obs, 1.0)
            if isinstance(val, (list, tuple, np.ndarray)):
                sval = "[" + ", ".join(f"{x}" for x in val) + "]"
            else:
                sval = f"{val}"
            parts.append(f"{obs}: {sval}")
        content += f"    {input_name}: {{ "
        content += ", ".join(parts)
        content += " },\n"
    content += "  }\n"

    content += "input_obs_size_map:\n  {\n"
    for input_name in config_dict["input_names"]:
        dim = config_dict["input_obs_size_map"][input_name]
        content += f"    {input_name}: {dim},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(
        f"{input_name}: {config_dict['obs_history_length'][input_name]}"
        for input_name in config_dict["input_names"]
    )
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"
    content += f"velocity_x_forward_scale: {config_dict['velocity_x_forward_scale']}\n"
    content += f"velocity_x_backward_scale: {config_dict['velocity_x_backward_scale']}\n"
    content += f"velocity_y_scale: {config_dict['velocity_y_scale']}\n"
    content += f"velocity_yaw_scale: {config_dict['velocity_yaw_scale']}\n"
    content += "max_velocity: ["
    content += ", ".join(f"{float(v)}" for v in config_dict["max_velocity"])
    content += "]\n"
    content += "max_acceleration: ["
    content += ", ".join(f"{float(v)}" for v in config_dict["max_acceleration"])
    content += "]\n"
    content += "max_jerk: ["
    content += ", ".join(f"{float(v)}" for v in config_dict["max_jerk"])
    content += "]\n"
    content += "threshold:\n"
    content += f"  limit_lower: {config_dict['threshold']['limit_lower']}\n"
    content += f"  limit_upper: {config_dict['threshold']['limit_upper']}\n"
    content += f"  damping: {config_dict['threshold']['damping']}\n"
    content += f"max_torques: {config_dict['max_torques']}\n"
    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)
