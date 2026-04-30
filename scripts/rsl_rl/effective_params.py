"""Effective Galileo configuration summary helpers.

The raw Isaac Lab YAML dumps are complete but hard to skim.  These helpers
write a compact "what actually mattered" artifact beside each training run and
are also reused by the post-reset effective-parameter dump script.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if hasattr(value, "to_dict"):
        return _to_serializable(value.to_dict())
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__name__"):
        return value.__name__
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _to_serializable(vars(value))
    return str(value)


def _cfg_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _class_public_attrs(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    entries: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        value = getattr(obj, name)
        if callable(value) or isinstance(value, type):
            continue
        entries[name] = _to_serializable(value)
    return entries


def _event_param(events: Any, event_name: str, param_name: str, default: Any = None) -> Any:
    event_cfg = _cfg_attr(events, event_name)
    params = _cfg_attr(event_cfg, "params", {})
    if isinstance(params, Mapping):
        return params.get(param_name, default)
    return default


def _term_dump(term: Any) -> dict[str, Any] | None:
    if term is None:
        return None
    if not any(hasattr(term, key) for key in ("func", "weight", "params")):
        return None
    func = _cfg_attr(term, "func")
    return {
        "func": getattr(func, "__name__", str(func)) if func is not None else None,
        "weight": _cfg_attr(term, "weight"),
        "params": _to_serializable(_cfg_attr(term, "params", {})),
    }


def _manager_terms_dump(block: Any) -> dict[str, Any]:
    terms: dict[str, Any] = {}
    if block is None:
        return terms
    for name in dir(block):
        if name.startswith("_"):
            continue
        term = getattr(block, name)
        if callable(term) or isinstance(term, type):
            continue
        dumped = _term_dump(term)
        if dumped is not None:
            terms[name] = dumped
    return terms


def _terrain_subterrains_dump(terrain_generator: Any) -> dict[str, Any]:
    sub_terrains = _cfg_attr(terrain_generator, "sub_terrains", {}) or {}
    dumped: dict[str, Any] = {}
    for name, terrain_cfg in sub_terrains.items():
        entry = {
            "type": type(terrain_cfg).__name__,
            "proportion": _cfg_attr(terrain_cfg, "proportion"),
        }
        for key in (
            "step_height_range",
            "step_width",
            "slope_range",
            "platform_width",
            "border_width",
            "holes",
            "noise_range",
            "noise_step",
            "grid_width",
            "grid_height_range",
        ):
            value = _cfg_attr(terrain_cfg, key)
            if value is not None:
                entry[key] = _to_serializable(value)
        dumped[str(name)] = entry
    return dumped


def _command_cfg_dump(command_cfg: Any, config_summary: Any) -> dict[str, Any]:
    command_summary = getattr(config_summary, "command", None)
    ranges = getattr(command_summary, "ranges", {})
    default_ranges = _cfg_attr(command_cfg, "ranges")
    clips = _cfg_attr(command_cfg, "clips")
    return {
        "resampling_time_range": _cfg_attr(command_cfg, "resampling_time_range"),
        "levels": {
            "lin_x": _cfg_attr(command_cfg, "lin_x_level"),
            "lin_y": _cfg_attr(command_cfg, "lin_y_level"),
            "ang_z": _cfg_attr(command_cfg, "ang_z_level"),
        },
        "max_levels": {
            "lin_x": _cfg_attr(command_cfg, "max_lin_x_level"),
            "lin_y": _cfg_attr(command_cfg, "max_lin_y_level"),
            "ang_z": _cfg_attr(command_cfg, "max_ang_z_level"),
        },
        "level_steps": {
            "lin_x": _cfg_attr(command_cfg, "lin_x_level_step"),
            "lin_y": _cfg_attr(command_cfg, "lin_y_level_step"),
            "ang_z": _cfg_attr(command_cfg, "ang_z_level_step"),
        },
        "heading_control_stiffness": _cfg_attr(command_cfg, "heading_control_stiffness"),
        "velocity_scales": {
            "x_forward": _cfg_attr(command_cfg, "velocity_x_forward_scale"),
            "x_backward": _cfg_attr(command_cfg, "velocity_x_backward_scale"),
            "y": _cfg_attr(command_cfg, "velocity_y_scale"),
            "yaw": _cfg_attr(command_cfg, "velocity_yaw_scale"),
        },
        "max_velocity": _cfg_attr(command_cfg, "max_velocity"),
        "deadzone_clips": {
            "lin_vel_clip": _cfg_attr(clips, "lin_vel_clip"),
            "ang_vel_clip": _cfg_attr(clips, "ang_vel_clip"),
            "small_commands_to_zero": _cfg_attr(command_cfg, "small_commands_to_zero"),
        },
        "default_ranges": _to_serializable(default_ranges),
        "terrain_mode_groups": _to_serializable(getattr(command_summary, "terrain_mode_groups", {})),
        "terrain_ranges": _to_serializable(ranges),
    }


def _agent_dump(agent_cfg: Any) -> dict[str, Any]:
    if agent_cfg is None:
        return {}
    algorithm_cfg = _cfg_attr(agent_cfg, "algorithm")
    policy_cfg = _cfg_attr(agent_cfg, "policy")
    return {
        "experiment_name": _cfg_attr(agent_cfg, "experiment_name"),
        "run_name": _cfg_attr(agent_cfg, "run_name"),
        "seed": _cfg_attr(agent_cfg, "seed"),
        "device": _cfg_attr(agent_cfg, "device"),
        "num_steps_per_env": _cfg_attr(agent_cfg, "num_steps_per_env"),
        "max_iterations": _cfg_attr(agent_cfg, "max_iterations"),
        "framework_type": _cfg_attr(agent_cfg, "framework_type"),
        "algorithm": _to_serializable(algorithm_cfg),
        "policy": _to_serializable(policy_cfg),
    }


def build_effective_config_summary(env_cfg: Any, agent_cfg: Any | None = None) -> dict[str, Any]:
    """Build a compact, JSON-safe summary of the active Galileo configuration."""

    config_summary = _cfg_attr(env_cfg, "config_summary")
    scene = _cfg_attr(env_cfg, "scene")
    terrain = _cfg_attr(scene, "terrain")
    terrain_generator = _cfg_attr(terrain, "terrain_generator")
    commands = _cfg_attr(env_cfg, "commands")
    base_velocity = _cfg_attr(commands, "base_velocity")
    actions = _cfg_attr(env_cfg, "actions")
    joint_pos_action = _cfg_attr(actions, "joint_pos")
    sim = _cfg_attr(env_cfg, "sim")

    summary = {
        "general": {
            "decimation": _cfg_attr(env_cfg, "decimation"),
            "episode_length_s": _cfg_attr(env_cfg, "episode_length_s"),
            "num_envs": _cfg_attr(scene, "num_envs"),
            "env_spacing": _cfg_attr(scene, "env_spacing"),
            "seed": _cfg_attr(env_cfg, "seed"),
        },
        "sim": {
            "dt": _cfg_attr(sim, "dt"),
            "render_interval": _cfg_attr(sim, "render_interval"),
            "device": _cfg_attr(sim, "device"),
            "gravity": _cfg_attr(sim, "gravity"),
            "use_fabric": _cfg_attr(sim, "use_fabric"),
        },
        "terrain": {
            "profile": _cfg_attr(env_cfg, "terrain_profile"),
            "debug_single_family": _cfg_attr(env_cfg, "terrain_debug_single_family"),
            "type": _cfg_attr(terrain, "terrain_type"),
            "max_init_terrain_level": _cfg_attr(terrain, "max_init_terrain_level"),
            "generator": {
                "size": _cfg_attr(terrain_generator, "size"),
                "border_width": _cfg_attr(terrain_generator, "border_width"),
                "num_rows": _cfg_attr(terrain_generator, "num_rows"),
                "num_cols": _cfg_attr(terrain_generator, "num_cols"),
                "horizontal_scale": _cfg_attr(terrain_generator, "horizontal_scale"),
                "vertical_scale": _cfg_attr(terrain_generator, "vertical_scale"),
                "slope_threshold": _cfg_attr(terrain_generator, "slope_threshold"),
                "curriculum": _cfg_attr(terrain_generator, "curriculum"),
                "difficulty_range": _cfg_attr(terrain_generator, "difficulty_range"),
                "sub_terrains": _terrain_subterrains_dump(terrain_generator),
            },
            "plane_split": {
                "names": getattr(getattr(config_summary, "terrain", None), "flat_subterrain_names", ()),
                "proportions": getattr(
                    getattr(config_summary, "terrain", None), "flat_subterrain_proportions", {}
                ),
            },
        },
        "command": _command_cfg_dump(base_velocity, config_summary),
        "action": {
            "type": type(joint_pos_action).__name__ if joint_pos_action is not None else None,
            "scale": _cfg_attr(joint_pos_action, "scale"),
            "use_default_offset": _cfg_attr(joint_pos_action, "use_default_offset"),
            "action_delay_steps": _cfg_attr(joint_pos_action, "action_delay_steps"),
            "delay_update_global_steps": _cfg_attr(joint_pos_action, "delay_update_global_steps"),
            "history_length": _cfg_attr(joint_pos_action, "history_length"),
            "use_delay": _cfg_attr(joint_pos_action, "use_delay"),
            "clip": _cfg_attr(joint_pos_action, "clip"),
        },
        "observations": {
            "cts_layout": _class_public_attrs(getattr(getattr(config_summary, "obs", None), "cts", None)),
        },
        "randomization_ranges": {
            "base_mass": _event_param(env_cfg.events, "randomize_base_mass", "mass_distribution_params"),
            "base_mass_operation": _event_param(env_cfg.events, "randomize_base_mass", "operation"),
            "base_com": _event_param(env_cfg.events, "randomize_base_com", "com_range"),
            "friction": _event_param(env_cfg.events, "physics_material", "friction_range"),
            "restitution": _event_param(env_cfg.events, "physics_material", "restitution_range"),
            "reset_base_pose": _event_param(env_cfg.events, "reset_base_pose", "pose_range"),
            "reset_base_velocity": _event_param(env_cfg.events, "reset_base_pose", "velocity_range"),
            "reset_leg_joints": _event_param(env_cfg.events, "reset_leg_joints", "position_range"),
            "actuator_kp": _event_param(
                env_cfg.events, "randomize_actuator_kp_kd_gains", "stiffness_distribution_params"
            ),
            "actuator_kd": _event_param(
                env_cfg.events, "randomize_actuator_kp_kd_gains", "damping_distribution_params"
            ),
            "push_velocity": _event_param(env_cfg.events, "push_robot_vel", "velocity_range"),
            "push_torque": _event_param(env_cfg.events, "push_robot_torque", "torque_range"),
        },
        "curriculum": _manager_terms_dump(_cfg_attr(env_cfg, "curriculum")),
        "rewards": _manager_terms_dump(_cfg_attr(env_cfg, "rewards")),
        "costs": _manager_terms_dump(_cfg_attr(env_cfg, "costs")),
        "agent": _agent_dump(agent_cfg),
    }
    return _to_serializable(summary)


def write_effective_config_summary(
    log_dir: str | Path,
    env_cfg: Any,
    agent_cfg: Any | None = None,
    *,
    filename: str = "effective_summary.json",
) -> Path:
    """Write the compact effective summary under ``<log_dir>/params``."""

    target_path = Path(log_dir) / "params" / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(build_effective_config_summary(env_cfg, agent_cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target_path
