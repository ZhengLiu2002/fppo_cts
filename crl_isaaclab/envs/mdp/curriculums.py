# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for CRL tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_ratio: float = 0.5,
    move_down_ratio: float = 0.5,
    progress_mode: str = "displacement",
    command_name: str = "base_velocity",
    min_active_command_speed: float = 0.0,
    min_active_ratio: float = 0.0,
    freeze_inactive_envs: bool = True,
) -> torch.Tensor:
    """Update terrain levels based on distance walked under velocity commands.

    - If the robot walks farther than terrain.size * move_up_ratio -> move up.
    - If it walks much less than commanded distance * move_down_ratio -> move down.
    - ``command_projected`` mode integrates progress along the commanded planar direction
      across the whole episode, which is more robust when commands resample multiple times.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    env_id_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    terrain_span = terrain.cfg.terrain_generator.size[0] * move_up_ratio

    if progress_mode == "command_projected":
        command_term = env.command_manager.get_term(command_name)
        progress_stats = getattr(command_term, "get_episode_planar_progress", None)
        if callable(progress_stats):
            episode_progress = progress_stats(env_id_tensor)
            command_distance = episode_progress["command_distance"]
            projected_distance = episode_progress["projected_distance"]
            active_time = episode_progress["active_time"]

            active = command_distance > 1.0e-6
            required_active_time = max(float(min_active_ratio), 0.0) * env.max_episode_length_s
            if required_active_time > 0.0:
                active &= active_time >= required_active_time

            required_command_distance = max(float(min_active_command_speed), 0.0) * required_active_time
            if required_command_distance > 0.0:
                active &= command_distance >= required_command_distance

            move_up = active & (projected_distance > terrain_span)
            move_down = active & ~move_up & (projected_distance < command_distance * move_down_ratio)
            if not freeze_inactive_envs:
                move_down |= ~active

            terrain.update_env_origins(env_id_tensor, move_up, move_down)
            return torch.mean(terrain.terrain_levels.float())

    command = env.command_manager.get_command(command_name)
    distance = torch.norm(
        asset.data.root_pos_w[env_id_tensor, :2] - env.scene.env_origins[env_id_tensor, :2], dim=1
    )
    move_up = distance > terrain_span
    move_down = (
        distance
        < torch.norm(command[env_id_tensor, :2], dim=1) * env.max_episode_length_s * move_down_ratio
    )
    move_down *= ~move_up

    terrain.update_env_origins(env_id_tensor, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def _clamp_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _interp_scalar(start: float, end: float, progress: float) -> float:
    return float(start + (end - start) * progress)


def _interp_structure(start: Any, end: Any, progress: float) -> Any:
    """Linearly interpolate nested numeric ranges used by event configs."""
    if isinstance(end, dict):
        start = start if isinstance(start, dict) else {}
        return {
            key: _interp_structure(start.get(key, 0.0), value, progress)
            for key, value in end.items()
        }
    if isinstance(end, tuple):
        if not isinstance(start, tuple):
            start = tuple(0.0 for _ in end)
        return tuple(_interp_structure(s, e, progress) for s, e in zip(start, end, strict=True))
    if isinstance(end, list):
        if not isinstance(start, list):
            start = [0.0 for _ in end]
        return [_interp_structure(s, e, progress) for s, e in zip(start, end, strict=True)]
    return _interp_scalar(float(start), float(end), progress)


def _set_event_param(env: ManagerBasedRLEnv, term_name: str, param_name: str, value: Any) -> None:
    try:
        term_cfg = env.event_manager.get_term_cfg(term_name)
    except Exception:
        return
    if term_cfg is None:
        return
    term_cfg.params[param_name] = value


def _get_event_param(env: ManagerBasedRLEnv, term_name: str, param_name: str) -> Any | None:
    try:
        term_cfg = env.event_manager.get_term_cfg(term_name)
    except Exception:
        return None
    if term_cfg is None:
        return None
    return term_cfg.params.get(param_name)


def _cache_event_targets(
    env: ManagerBasedRLEnv,
    specs: Sequence[tuple[str, str, Any]],
) -> dict[tuple[str, str], Any]:
    cache_attr = "_domain_randomization_curriculum_targets"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached

    targets: dict[tuple[str, str], Any] = {}
    for term_name, param_name, _ in specs:
        value = _get_event_param(env, term_name, param_name)
        if value is not None:
            targets[(term_name, param_name)] = value
    setattr(env, cache_attr, targets)
    return targets


def _domain_randomization_specs() -> tuple[tuple[str, str, Any], ...]:
    zero_pose = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    zero_velocity = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    return (
        ("randomize_base_mass", "mass_distribution_params", (0.0, 0.0)),
        ("randomize_base_com", "com_range", {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}),
        ("physics_material", "friction_range", (1.0, 1.0)),
        ("physics_material", "restitution_range", (0.0, 0.0)),
        ("reset_base_pose", "pose_range", zero_pose),
        ("reset_base_pose", "velocity_range", zero_velocity),
        ("reset_leg_joints", "position_range", (0.0, 0.0)),
        ("randomize_actuator_kp_kd_gains", "stiffness_distribution_params", (1.0, 1.0)),
        ("randomize_actuator_kp_kd_gains", "damping_distribution_params", (1.0, 1.0)),
        ("push_robot_vel", "velocity_range", zero_velocity),
        ("push_robot_torque", "force_range", (0.0, 0.0)),
        ("push_robot_torque", "torque_range", (0.0, 0.0)),
    )


def _apply_domain_randomization_targets(
    env: ManagerBasedRLEnv,
    *,
    progress: float,
) -> bool:
    specs = _domain_randomization_specs()
    targets = _cache_event_targets(env, specs)
    if not targets:
        return False
    for term_name, param_name, easy_value in specs:
        target_value = targets.get((term_name, param_name))
        if target_value is None:
            continue
        _set_event_param(
            env,
            term_name,
            param_name,
            _interp_structure(easy_value, target_value, progress),
        )
    return True


def initialize_domain_randomization_curriculum(env: ManagerBasedRLEnv) -> bool:
    """Initialize DR events to the current curriculum level before the first reset.

    This is primarily used right after environment creation so a preset can keep
    full DR target ranges in config while still starting from deterministic
    level-0 dynamics before training/evaluation begins.
    """

    curriculum_cfg = getattr(getattr(env, "cfg", None), "curriculum", None)
    term_cfg = getattr(curriculum_cfg, "domain_randomization_scale", None)
    if term_cfg is None:
        return False

    params = dict(getattr(term_cfg, "params", {}) or {})
    max_level = max(float(params.get("max_level", 1.0)), 1.0e-6)
    warmup_steps = max(int(params.get("warmup_steps", 0)), 0)

    level_attr = "_domain_randomization_curriculum_level"
    last_step_attr = "_domain_randomization_curriculum_last_step"
    level = max(0.0, min(float(getattr(env, level_attr, 0.0)), max_level))
    setattr(env, level_attr, level)
    if not hasattr(env, last_step_attr):
        setattr(env, last_step_attr, warmup_steps)

    return _apply_domain_randomization_targets(env, progress=_clamp_unit(level / max_level))


def _command_level(env: ManagerBasedRLEnv, name: str) -> float:
    command = env.command_manager.get_term("base_velocity")
    return float(getattr(command.cfg, name, 0.0))


def _terrain_level_mean(env: ManagerBasedRLEnv) -> float:
    terrain_levels = getattr(getattr(env.scene, "terrain", None), "terrain_levels", None)
    if terrain_levels is None:
        return 0.0
    return float(torch.mean(terrain_levels.float()).item())


def _terrain_type_stage(env: ManagerBasedRLEnv) -> int:
    return int(getattr(env, "_terrain_type_curriculum_stage", 0))


def _stage_threshold(
    value: float | Sequence[float | None] | None,
    *,
    stage: int,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return None
        selected = value[min(max(int(stage), 0), len(value) - 1)]
        return None if selected is None else float(selected)
    return float(value)


def _terrain_type_columns_by_name(env: ManagerBasedRLEnv) -> dict[str, list[int]]:
    cache_attr = "_terrain_type_columns_by_name"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached

    terrain = getattr(env.scene, "terrain", None)
    terrain_gen = getattr(terrain, "terrain_generator_class", None)
    terrain_names = getattr(terrain_gen, "terrain_names", None)
    mapping: dict[str, list[int]] = {}
    if terrain_names is not None:
        names = np.asarray(terrain_names)
        if names.ndim == 3 and names.shape[-1] == 1:
            names = names[..., 0]
        if names.ndim >= 2 and names.shape[0] > 0:
            col_names = [str(name) for name in names[0].tolist()]
        elif names.ndim == 1:
            col_names = [str(name) for name in names.tolist()]
        else:
            col_names = []
        for col_idx, name in enumerate(col_names):
            mapping.setdefault(name, []).append(col_idx)

    setattr(env, cache_attr, mapping)
    return mapping


def _terrain_type_name_by_column(env: ManagerBasedRLEnv) -> dict[int, str]:
    cache_attr = "_terrain_type_name_by_column"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached

    inverse_mapping: dict[int, str] = {}
    for terrain_name, columns in _terrain_type_columns_by_name(env).items():
        for column in columns:
            inverse_mapping[int(column)] = str(terrain_name)

    setattr(env, cache_attr, inverse_mapping)
    return inverse_mapping


def _allowed_terrain_type_columns(
    env: ManagerBasedRLEnv,
    stage_names: Sequence[Sequence[str]] | None,
    stage_index: int,
) -> list[int]:
    terrain = getattr(env.scene, "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_origins is None:
        return []

    num_cols = int(terrain_origins.shape[1])
    if not stage_names:
        return list(range(num_cols))

    stage_index = max(0, min(int(stage_index), len(stage_names) - 1))
    columns_by_name = _terrain_type_columns_by_name(env)
    allowed: list[int] = []
    for terrain_name in stage_names[stage_index]:
        allowed.extend(columns_by_name.get(str(terrain_name), ()))

    allowed = sorted(set(int(col) for col in allowed))
    if allowed:
        return allowed
    return list(range(num_cols))


def _tracking_gate(
    env: ManagerBasedRLEnv,
    *,
    component: str,
    error_threshold: float | None,
    min_command_speed: float,
    min_active_ratio: float,
) -> bool:
    if error_threshold is None:
        return True
    command_term = env.command_manager.get_term("base_velocity")
    asset_name = getattr(command_term.cfg, "asset_name", "robot")
    asset: Articulation = env.scene[asset_name]
    cmd = env.command_manager.get_command("base_velocity")

    if component == "lin_xy":
        cmd_speed = torch.norm(cmd[:, :2], dim=1)
        active = cmd_speed > max(float(min_command_speed), 0.0)
        if float(active.float().mean().item()) < max(float(min_active_ratio), 0.0):
            return False
        if not torch.any(active):
            return False
        error = torch.norm(cmd[active, :2] - asset.data.root_lin_vel_b[active, :2], dim=1)
    elif component == "lin_y":
        cmd_speed = torch.abs(cmd[:, 1])
        active = cmd_speed > max(float(min_command_speed), 0.0)
        if float(active.float().mean().item()) < max(float(min_active_ratio), 0.0):
            return False
        if not torch.any(active):
            return False
        error = torch.abs(cmd[active, 1] - asset.data.root_lin_vel_b[active, 1])
    else:
        cmd_speed = torch.abs(cmd[:, 2])
        active = cmd_speed > max(float(min_command_speed), 0.0)
        if float(active.float().mean().item()) < max(float(min_active_ratio), 0.0):
            return False
        if not torch.any(active):
            return False
        error = torch.abs(cmd[active, 2] - asset.data.root_ang_vel_b[active, 2])

    return float(error.mean().item()) <= float(error_threshold)


def lin_vel_x_command_threshold(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    episodes_per_level: int = 8,
    warmup_steps: int = 0,
    min_progress_steps: int | None = None,
    terrain_level_threshold: float | None = None,
    error_threshold: float | None = None,
    min_command_speed: float = 0.2,
    min_active_ratio: float = 0.05,
    min_terrain_type_stage: int = 0,
) -> torch.Tensor:
    """Curriculum for widening lin_vel_x command range.

    Supports two modes:
    1) legacy time-based stepping when ``min_progress_steps`` is None
    2) gated stepping with warmup, minimum update interval, and tracking checks
    """
    command = env.command_manager.get_term("base_velocity")
    max_episode_length = env.max_episode_length
    lin_x_level = float(getattr(command.cfg, "lin_x_level", 0.0))
    max_lin_x_level = float(getattr(command.cfg, "max_lin_x_level", 1.0))
    lin_x_level_step = float(getattr(command.cfg, "lin_x_level_step", 1.0))
    lin_x_level_step = max(lin_x_level_step, 0.0)

    lin_x_level = max(0.0, min(lin_x_level, max_lin_x_level))
    should_advance = False
    if min_progress_steps is None:
        should_advance = env.common_step_counter > (
            (lin_x_level + 1.0) * max_episode_length * episodes_per_level
        )
    else:
        min_progress_steps = max(int(min_progress_steps), 1)
        warmup_steps = max(int(warmup_steps), 0)
        last_step_attr = "_lin_x_curriculum_last_step"
        last_step = int(getattr(env, last_step_attr, warmup_steps))
        should_advance = (
            env.common_step_counter >= warmup_steps
            and (env.common_step_counter - last_step) >= min_progress_steps
        )
        if should_advance and _terrain_type_stage(env) < int(min_terrain_type_stage):
            should_advance = False
        if should_advance and terrain_level_threshold is not None:
            terrain_levels = getattr(getattr(env.scene, "terrain", None), "terrain_levels", None)
            if terrain_levels is not None:
                terrain_level_mean = float(torch.mean(terrain_levels.float()).item())
                should_advance = terrain_level_mean >= float(terrain_level_threshold)
        if should_advance:
            should_advance = _tracking_gate(
                env,
                component="lin_xy",
                error_threshold=error_threshold,
                min_command_speed=min_command_speed,
                min_active_ratio=min_active_ratio,
            )

    if should_advance and (lin_x_level < max_lin_x_level):
        if lin_x_level_step > 0.0:
            lin_x_level = min(lin_x_level + lin_x_level_step, max_lin_x_level)
        else:
            lin_x_level = max_lin_x_level
        command.cfg.lin_x_level = lin_x_level
        if min_progress_steps is not None:
            setattr(env, "_lin_x_curriculum_last_step", int(env.common_step_counter))

    # Always apply the current curriculum level to the command ranges.
    denom = max(float(max_lin_x_level), 1.0e-6)
    ranges = command.cfg.ranges
    if (
        hasattr(ranges, "start_curriculum_lin_x")
        and ranges.start_curriculum_lin_x is not None
        and hasattr(ranges, "max_curriculum_lin_x")
        and ranges.max_curriculum_lin_x is not None
    ):
        start_min, start_max = ranges.start_curriculum_lin_x
        max_min, max_max = ranges.max_curriculum_lin_x
        step0 = (max_min - start_min) / denom
        step1 = (max_max - start_max) / denom
        ranges.lin_vel_x = (start_min + step0 * lin_x_level, start_max + step1 * lin_x_level)

    return torch.tensor(lin_x_level, device=env.device)


def lin_vel_y_command_threshold(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    episodes_per_level: int = 8,
    warmup_steps: int = 0,
    min_progress_steps: int | None = None,
    terrain_level_threshold: float | None = None,
    error_threshold: float | None = None,
    min_command_speed: float = 0.1,
    min_active_ratio: float = 0.05,
    min_lin_x_level: float = 0.0,
) -> torch.Tensor:
    """Curriculum for widening lateral linear velocity commands."""
    command = env.command_manager.get_term("base_velocity")
    max_episode_length = env.max_episode_length
    lin_y_level = float(getattr(command.cfg, "lin_y_level", 0.0))
    max_lin_y_level = float(getattr(command.cfg, "max_lin_y_level", 1.0))
    lin_y_level_step = max(float(getattr(command.cfg, "lin_y_level_step", 1.0)), 0.0)
    lin_x_level = float(getattr(command.cfg, "lin_x_level", 0.0))

    lin_y_level = max(0.0, min(lin_y_level, max_lin_y_level))
    if min_progress_steps is None:
        should_advance = env.common_step_counter > (
            (lin_y_level + 1.0) * max_episode_length * episodes_per_level
        )
    else:
        min_progress_steps = max(int(min_progress_steps), 1)
        warmup_steps = max(int(warmup_steps), 0)
        last_step_attr = "_lin_y_curriculum_last_step"
        last_step = int(getattr(env, last_step_attr, warmup_steps))
        should_advance = (
            env.common_step_counter >= warmup_steps
            and (env.common_step_counter - last_step) >= min_progress_steps
        )
        if should_advance and lin_x_level < float(min_lin_x_level):
            should_advance = False
        if should_advance and terrain_level_threshold is not None:
            should_advance = _terrain_level_mean(env) >= float(terrain_level_threshold)
        if should_advance:
            should_advance = _tracking_gate(
                env,
                component="lin_y",
                error_threshold=error_threshold,
                min_command_speed=min_command_speed,
                min_active_ratio=min_active_ratio,
            )

    if should_advance and (lin_y_level < max_lin_y_level):
        if lin_y_level_step > 0.0:
            lin_y_level = min(lin_y_level + lin_y_level_step, max_lin_y_level)
        else:
            lin_y_level = max_lin_y_level
        command.cfg.lin_y_level = lin_y_level
        if min_progress_steps is not None:
            setattr(env, "_lin_y_curriculum_last_step", int(env.common_step_counter))

    denom = max(float(max_lin_y_level), 1.0e-6)
    ranges = command.cfg.ranges
    if (
        hasattr(ranges, "start_curriculum_lin_y")
        and ranges.start_curriculum_lin_y is not None
        and hasattr(ranges, "max_curriculum_lin_y")
        and ranges.max_curriculum_lin_y is not None
    ):
        start_min, start_max = ranges.start_curriculum_lin_y
        max_min, max_max = ranges.max_curriculum_lin_y
        step0 = (max_min - start_min) / denom
        step1 = (max_max - start_max) / denom
        ranges.lin_vel_y = (start_min + step0 * lin_y_level, start_max + step1 * lin_y_level)

    return torch.tensor(lin_y_level, device=env.device)


def ang_vel_z_command_threshold(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    episodes_per_level: int = 8,
    warmup_steps: int = 0,
    min_progress_steps: int | None = None,
    terrain_level_threshold: float | None = None,
    error_threshold: float | None = None,
    min_command_speed: float = 0.1,
    min_active_ratio: float = 0.05,
    min_lin_x_level: float = 0.0,
) -> torch.Tensor:
    """Curriculum for widening ang_vel_z command range."""
    command = env.command_manager.get_term("base_velocity")
    max_episode_length = env.max_episode_length
    ang_z_level = float(getattr(command.cfg, "ang_z_level", 0.0))
    lin_x_level = float(getattr(command.cfg, "lin_x_level", 0.0))
    max_ang_z_level = float(getattr(command.cfg, "max_ang_z_level", 1.0))
    ang_z_level_step = float(getattr(command.cfg, "ang_z_level_step", 1.0))
    ang_z_level_step = max(ang_z_level_step, 0.0)

    ang_z_level = max(0.0, min(ang_z_level, max_ang_z_level))
    should_advance = False
    if min_progress_steps is None:
        should_advance = env.common_step_counter > (
            (ang_z_level + 1.0) * max_episode_length * episodes_per_level
        )
    else:
        min_progress_steps = max(int(min_progress_steps), 1)
        warmup_steps = max(int(warmup_steps), 0)
        last_step_attr = "_ang_z_curriculum_last_step"
        last_step = int(getattr(env, last_step_attr, warmup_steps))
        should_advance = (
            env.common_step_counter >= warmup_steps
            and (env.common_step_counter - last_step) >= min_progress_steps
        )
        if should_advance and lin_x_level < float(min_lin_x_level):
            should_advance = False
        if should_advance and terrain_level_threshold is not None:
            terrain_levels = getattr(getattr(env.scene, "terrain", None), "terrain_levels", None)
            if terrain_levels is not None:
                terrain_level_mean = float(torch.mean(terrain_levels.float()).item())
                should_advance = terrain_level_mean >= float(terrain_level_threshold)
            else:
                should_advance = False
        if should_advance:
            should_advance = _tracking_gate(
                env,
                component="ang_z",
                error_threshold=error_threshold,
                min_command_speed=min_command_speed,
                min_active_ratio=min_active_ratio,
            )

    if should_advance and (ang_z_level < max_ang_z_level):
        if ang_z_level_step > 0.0:
            ang_z_level = min(ang_z_level + ang_z_level_step, max_ang_z_level)
        else:
            ang_z_level = max_ang_z_level
        command.cfg.ang_z_level = ang_z_level
        if min_progress_steps is not None:
            setattr(env, "_ang_z_curriculum_last_step", int(env.common_step_counter))

    # Always apply the current curriculum level to the command ranges.
    denom = max(float(max_ang_z_level), 1.0e-6)
    ranges = command.cfg.ranges
    if (
        hasattr(ranges, "start_curriculum_ang_z")
        and ranges.start_curriculum_ang_z is not None
        and hasattr(ranges, "max_curriculum_ang_z")
        and ranges.max_curriculum_ang_z is not None
    ):
        start_min, start_max = ranges.start_curriculum_ang_z
        max_min, max_max = ranges.max_curriculum_ang_z
        step0 = (max_min - start_min) / denom
        step1 = (max_max - start_max) / denom
        ranges.ang_vel_z = (start_min + step0 * ang_z_level, start_max + step1 * ang_z_level)

    return torch.tensor(ang_z_level, device=env.device)


def terrain_type_progression(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    warmup_steps: int = 0,
    min_progress_steps: int = 9600,
    stage_names: Sequence[Sequence[str]] | None = None,
    terrain_level_threshold: float | None = None,
    terrain_level_thresholds: Sequence[float | None] | None = None,
    min_lin_x_level: float = 0.35,
    min_lin_x_levels: Sequence[float | None] | None = None,
    min_lin_y_level: float = 0.15,
    min_lin_y_levels: Sequence[float | None] | None = None,
    min_ang_z_level: float = 0.20,
    min_ang_z_levels: Sequence[float | None] | None = None,
    lin_error_threshold: float | None = None,
    lin_error_thresholds: Sequence[float | None] | None = None,
    ang_error_threshold: float | None = None,
    ang_error_thresholds: Sequence[float | None] | None = None,
    lin_min_command_speed: float = 0.2,
    ang_min_command_speed: float = 0.16,
    min_active_ratio: float = 0.05,
    max_stall_steps: int | None = None,
    reset_level_on_switch: int = 0,
) -> torch.Tensor:
    """Unlock terrain families progressively while keeping resets on easy rows first.

    Design notes:
    - Stage 0 is typically flat-only bootstrap. Requiring a high terrain row mean here can
      deadlock the terrain family curriculum because flat rows are not a meaningful mastery signal.
    - Later stages can still use terrain row thresholds, but should not stall forever if the agent
      tracks commands well and keeps surviving. ``max_stall_steps`` provides a bounded escape hatch.
    """
    terrain = getattr(env.scene, "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain is None or terrain_origins is None or terrain_types is None or terrain_levels is None:
        return torch.tensor(0.0, device=env.device)

    env_id_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if env_id_tensor.numel() == 0:
        return torch.tensor(float(_terrain_type_stage(env)), device=env.device)

    num_stages = max(len(stage_names) if stage_names is not None else 0, 1)
    max_stage = max(num_stages - 1, 0)
    stage_attr = "_terrain_type_curriculum_stage"
    last_step_attr = "_terrain_type_curriculum_last_step"
    stage = max(0, min(int(getattr(env, stage_attr, 0)), max_stage))
    prev_stage = stage
    last_step = int(getattr(env, last_step_attr, max(int(warmup_steps), 0)))
    stage_threshold = _stage_threshold(
        terrain_level_thresholds if terrain_level_thresholds is not None else terrain_level_threshold,
        stage=stage,
    )
    lin_stage_threshold = _stage_threshold(
        lin_error_thresholds if lin_error_thresholds is not None else lin_error_threshold,
        stage=stage,
    )
    ang_stage_threshold = _stage_threshold(
        ang_error_thresholds if ang_error_thresholds is not None else ang_error_threshold,
        stage=stage,
    )
    lin_x_stage_level = _stage_threshold(
        min_lin_x_levels if min_lin_x_levels is not None else min_lin_x_level,
        stage=stage,
    )
    lin_y_stage_level = _stage_threshold(
        min_lin_y_levels if min_lin_y_levels is not None else min_lin_y_level,
        stage=stage,
    )
    ang_z_stage_level = _stage_threshold(
        min_ang_z_levels if min_ang_z_levels is not None else min_ang_z_level,
        stage=stage,
    )
    stall_ready = (
        max_stall_steps is not None
        and (env.common_step_counter - last_step) >= max(int(max_stall_steps), 1)
    )

    should_advance = (
        env.common_step_counter >= max(int(warmup_steps), 0)
        and (env.common_step_counter - last_step) >= max(int(min_progress_steps), 1)
        and _command_level(env, "lin_x_level") >= float(lin_x_stage_level or 0.0)
        and _command_level(env, "lin_y_level") >= float(lin_y_stage_level or 0.0)
        and _command_level(env, "ang_z_level") >= float(ang_z_stage_level or 0.0)
    )
    if should_advance and stage_threshold is not None and not stall_ready:
        should_advance = _terrain_level_mean(env) >= float(stage_threshold)
    if should_advance:
        should_advance = _tracking_gate(
            env,
            component="lin_xy",
            error_threshold=lin_stage_threshold,
            min_command_speed=lin_min_command_speed,
            min_active_ratio=min_active_ratio,
        ) and _tracking_gate(
            env,
            component="ang_z",
            error_threshold=ang_stage_threshold,
            min_command_speed=ang_min_command_speed,
            min_active_ratio=min_active_ratio,
        )

    if should_advance and stage < max_stage:
        stage += 1
        setattr(env, stage_attr, stage)
        setattr(env, last_step_attr, int(env.common_step_counter))

    allowed_cols = _allowed_terrain_type_columns(env, stage_names, stage)
    allowed_cols_tensor = torch.as_tensor(allowed_cols, device=env.device, dtype=torch.long)
    sampled_cols = allowed_cols_tensor[
        torch.randint(
            low=0,
            high=max(int(allowed_cols_tensor.numel()), 1),
            size=(env_id_tensor.numel(),),
            device=env.device,
        )
    ]

    previous_cols = terrain.terrain_types[env_id_tensor].clone()
    switched_cols = sampled_cols != previous_cols
    terrain.terrain_types[env_id_tensor] = sampled_cols

    column_name_by_index = _terrain_type_name_by_column(env)
    same_family_switch = torch.zeros_like(switched_cols, dtype=torch.bool)
    if torch.any(switched_cols):
        previous_names = [
            column_name_by_index.get(int(col), "")
            for col in previous_cols.detach().cpu().tolist()
        ]
        sampled_names = [
            column_name_by_index.get(int(col), "")
            for col in sampled_cols.detach().cpu().tolist()
        ]
        same_family_switch = torch.as_tensor(
            [prev_name == next_name for prev_name, next_name in zip(previous_names, sampled_names)],
            device=env.device,
            dtype=torch.bool,
        )
        same_family_switch &= switched_cols

    # Preserve mastered rows when we only jump between columns that represent the same
    # terrain family (for example the three random_rough columns in the bootstrap stage).
    reset_mask = switched_cols & ~same_family_switch
    if torch.any(reset_mask):
        reset_level = max(int(reset_level_on_switch), 0)
        reset_level = min(reset_level, int(getattr(terrain, "max_terrain_level", 1)) - 1)
        terrain.terrain_levels[env_id_tensor[reset_mask]] = reset_level

    terrain.env_origins[env_id_tensor] = terrain.terrain_origins[
        terrain.terrain_levels[env_id_tensor], terrain.terrain_types[env_id_tensor]
    ]
    return torch.tensor(float(stage), device=env.device)


def domain_randomization_scale(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    warmup_steps: int = 0,
    min_progress_steps: int = 9600,
    max_level: float = 1.0,
    level_step: float = 0.05,
    min_lin_x_level: float = 0.35,
    min_lin_y_level: float = 0.15,
    min_ang_z_level: float = 0.25,
    terrain_level_threshold: float | None = None,
    min_terrain_type_stage: int = 0,
) -> torch.Tensor:
    """Ramp domain randomization from deterministic dynamics to configured ranges."""
    level_attr = "_domain_randomization_curriculum_level"
    last_step_attr = "_domain_randomization_curriculum_last_step"
    level = max(0.0, min(float(getattr(env, level_attr, 0.0)), float(max_level)))
    max_level = max(float(max_level), 1.0e-6)
    level_step = max(float(level_step), 0.0)

    last_step = int(getattr(env, last_step_attr, max(int(warmup_steps), 0)))
    should_advance = (
        env.common_step_counter >= max(int(warmup_steps), 0)
        and (env.common_step_counter - last_step) >= max(int(min_progress_steps), 1)
        and _command_level(env, "lin_x_level") >= float(min_lin_x_level)
        and _command_level(env, "lin_y_level") >= float(min_lin_y_level)
        and _command_level(env, "ang_z_level") >= float(min_ang_z_level)
        and _terrain_type_stage(env) >= int(min_terrain_type_stage)
    )
    if should_advance and terrain_level_threshold is not None:
        should_advance = _terrain_level_mean(env) >= float(terrain_level_threshold)

    if should_advance and level < max_level:
        level = min(level + level_step, max_level) if level_step > 0.0 else max_level
        setattr(env, level_attr, level)
        setattr(env, last_step_attr, int(env.common_step_counter))

    _apply_domain_randomization_targets(env, progress=_clamp_unit(level / max_level))
    return torch.tensor(level, device=env.device)
