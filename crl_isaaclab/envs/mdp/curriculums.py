# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for CRL tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

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
) -> torch.Tensor:
    """Update terrain levels based on distance walked under velocity commands.

    - If the robot walks farther than terrain.size * move_up_ratio -> move up.
    - If it walks much less than commanded distance * move_down_ratio -> move down.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] * move_up_ratio
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * move_down_ratio
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


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
        if should_advance and terrain_level_threshold is not None:
            terrain_levels = getattr(getattr(env.scene, "terrain", None), "terrain_levels", None)
            if terrain_levels is not None:
                terrain_level_mean = float(torch.mean(terrain_levels.float()).item())
                should_advance = terrain_level_mean >= float(terrain_level_threshold)
        if should_advance and error_threshold is not None:
            asset_name = getattr(command.cfg, "asset_name", "robot")
            asset: Articulation = env.scene[asset_name]
            cmd = env.command_manager.get_command("base_velocity")
            cmd_speed = torch.norm(cmd[:, :2], dim=1)
            active = cmd_speed > max(float(min_command_speed), 0.0)
            active_ratio = float(active.float().mean().item())
            should_advance = active_ratio >= max(float(min_active_ratio), 0.0)
            if should_advance and torch.any(active):
                lin_err = torch.norm(cmd[active, :2] - asset.data.root_lin_vel_b[active, :2], dim=1)
                should_advance = float(lin_err.mean().item()) <= float(error_threshold)

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
        if should_advance and float(getattr(command.cfg, "lin_x_level", 0.0)) < float(min_lin_x_level):
            should_advance = False
        if should_advance and terrain_level_threshold is not None:
            terrain_levels = getattr(getattr(env.scene, "terrain", None), "terrain_levels", None)
            if terrain_levels is not None:
                terrain_level_mean = float(torch.mean(terrain_levels.float()).item())
                should_advance = terrain_level_mean >= float(terrain_level_threshold)
        if should_advance and error_threshold is not None:
            asset_name = getattr(command.cfg, "asset_name", "robot")
            asset: Articulation = env.scene[asset_name]
            cmd = env.command_manager.get_command("base_velocity")
            cmd_speed = torch.abs(cmd[:, 2])
            # Prefer evaluating turning curriculum on true yaw-command environments.
            yaw_env_mask = getattr(command, "is_yaw_env", None)
            if torch.is_tensor(yaw_env_mask):
                yaw_env_mask = yaw_env_mask.to(device=cmd.device, dtype=torch.bool)
                if yaw_env_mask.ndim > 1:
                    yaw_env_mask = yaw_env_mask.squeeze(-1)
                if torch.any(yaw_env_mask):
                    candidate_mask = yaw_env_mask
                else:
                    candidate_mask = torch.ones_like(cmd_speed, dtype=torch.bool)
            else:
                candidate_mask = torch.ones_like(cmd_speed, dtype=torch.bool)

            active = candidate_mask & (cmd_speed > max(float(min_command_speed), 0.0))
            denom = torch.clamp(candidate_mask.float().sum(), min=1.0)
            active_ratio = float((active.float().sum() / denom).item())
            should_advance = active_ratio >= max(float(min_active_ratio), 0.0)
            if should_advance and torch.any(active):
                yaw_err = torch.abs(cmd[active, 2] - asset.data.root_ang_vel_b[active, 2])
                should_advance = float(yaw_err.mean().item()) <= float(error_threshold)

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
