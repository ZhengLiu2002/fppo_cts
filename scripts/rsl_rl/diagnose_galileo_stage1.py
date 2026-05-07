# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-1 diagnostics for Galileo CTS checkpoints.

This script is intentionally evaluation-only: it loads an existing policy,
collects per-terrain tracking/episode statistics, and runs a zero-command
``plane_stand`` probe with posture/action diagnostics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import os
from pathlib import Path
import statistics
import time
from typing import Any
import json

try:
    from scripts.rsl_rl.experiment_manager import (
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
    )
    from scripts.rsl_rl.runtime import (
        bootstrap_repo_paths,
        build_evaluation_output_path,
        build_log_root_path,
        build_run_manifest,
        resolve_checkpoint_path,
        resolve_task_variant,
        write_json_artifact,
    )
except ImportError:
    from experiment_manager import (  # type: ignore
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
    )
    from runtime import (  # type: ignore
        bootstrap_repo_paths,
        build_evaluation_output_path,
        build_log_root_path,
        build_run_manifest,
        resolve_checkpoint_path,
        resolve_task_variant,
        write_json_artifact,
    )


REPO_ROOT = bootstrap_repo_paths(__file__)

from isaaclab.app import AppLauncher

from scripts.rsl_rl import cli_args


parser = argparse.ArgumentParser(description="Run Galileo CTS stage-1 diagnostics.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=512,
    help="Completed episodes per diagnostic scenario.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional hard cap on environment steps per scenario.",
)
parser.add_argument(
    "--modes",
    type=str,
    default="mixed,plane_stand",
    help="Comma-separated diagnostics to run: mixed, plane_stand.",
)
parser.add_argument(
    "--stand_terrain",
    type=str,
    default="plane_stand",
    help="Terrain family used for the zero-command stand probe.",
)
parser.add_argument(
    "--summary_tag",
    type=str,
    default="stage1_diagnostics",
    help="Tag appended to the evaluation output directory.",
)
parser.add_argument(
    "--summary_dir",
    type=str,
    default=None,
    help="Optional explicit directory for diagnostic artifacts.",
)
parser.add_argument(
    "--real-time", action="store_true", default=False, help="Run in real-time, if possible."
)
parser.add_argument(
    "--progress_interval",
    type=int,
    default=250,
    help="Print scenario progress every this many environment steps.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.list_exp:
    available_presets = available_experiment_presets()
    if available_presets:
        print("Available experiment presets:")
        for entry in available_presets:
            description = f" - {entry['description']}" if entry["description"] else ""
            print(f"  {entry['name']}{description}")
    else:
        print("[INFO] No experiment presets found under `experiments/`.")
    raise SystemExit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from crl_isaaclab.envs.mdp.curriculums import initialize_domain_randomization_curriculum
from crl_isaaclab.terrains.runtime import resolve_env_terrain_names
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from crl_tasks.tasks.galileo.config.agents.rsl_rl_cfg import CRLRslRlOnPolicyRunnerCfg
from crl_tasks.tasks.galileo.config.mdp_cfg import LEG_JOINT_NAMES

from scripts.rsl_rl.vecenv_wrapper import CRLRslRlVecEnvWrapper

import crl_tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class RunningStats:
    """Small streaming scalar stats helper."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = None
        self.max = None

    def add(self, value: Any) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            tensor = value.detach().to(dtype=torch.float32).flatten()
            if tensor.numel() == 0:
                return
            tensor = tensor[torch.isfinite(tensor)]
            if tensor.numel() == 0:
                return
            count = int(tensor.numel())
            total = float(tensor.sum().item())
            total_sq = float(torch.square(tensor).sum().item())
            min_value = float(tensor.min().item())
            max_value = float(tensor.max().item())
        else:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.size == 0:
                return
            array = array[np.isfinite(array)]
            if array.size == 0:
                return
            count = int(array.size)
            total = float(array.sum())
            total_sq = float(np.square(array).sum())
            min_value = float(array.min())
            max_value = float(array.max())

        self.count += count
        self.sum += total
        self.sum_sq += total_sq
        self.min = min_value if self.min is None else min(self.min, min_value)
        self.max = max_value if self.max is None else max(self.max, max_value)

    def to_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": float(mean),
            "std": float(var**0.5),
            "min": self.min,
            "max": self.max,
        }


def _scalar_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values) if len(values) > 1 else 0.0),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _stats_map_to_dict(stats: dict[str, RunningStats]) -> dict[str, dict[str, Any]]:
    return {name: stat.to_dict() for name, stat in sorted(stats.items())}


def _extract_step_costs(
    extras: dict, device: torch.device | str, num_envs: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    payload = extras.get("cost") if isinstance(extras, dict) else None
    if payload is None:
        return torch.zeros(num_envs, device=device, dtype=torch.float32), {}

    if isinstance(payload, dict):
        term_costs: dict[str, torch.Tensor] = {}
        for key, value in sorted(payload.items()):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=device, dtype=torch.float32)
            value = value.to(device=device, dtype=torch.float32)
            if value.ndim == 0:
                value = value.expand(num_envs)
            elif value.ndim > 1:
                value = value.view(value.shape[0], -1).sum(dim=-1)
            term_costs[str(key)] = torch.clamp(value.reshape(num_envs), min=0.0)
        total_cost = torch.zeros(num_envs, device=device, dtype=torch.float32)
        for value in term_costs.values():
            total_cost = total_cost + value
        return total_cost, term_costs

    if not torch.is_tensor(payload):
        payload = torch.as_tensor(payload, device=device, dtype=torch.float32)
    payload = payload.to(device=device, dtype=torch.float32)
    if payload.ndim == 0:
        payload = payload.expand(num_envs)
    elif payload.ndim > 1:
        payload = payload.view(payload.shape[0], -1).sum(dim=-1)
    return torch.clamp(payload.reshape(num_envs), min=0.0), {}


def _resolve_eval_task_name(task_name: str) -> str:
    resolved_task = resolve_task_variant(
        task_name,
        variant="eval",
        registered_tasks=set(gym.registry.keys()),
    )
    if resolved_task != task_name:
        print(f"[INFO] Resolved evaluation task: {task_name} -> {resolved_task}")
    return resolved_task


def _terrain_names(base_env) -> list[str]:
    terrain = getattr(base_env.scene, "terrain", None)
    try:
        names = resolve_env_terrain_names(terrain)
    except Exception:
        names = None
    if names is None:
        return ["unknown"] * base_env.num_envs
    return [str(name) for name in np.asarray(names).reshape(-1).tolist()]


def _terrain_levels(base_env) -> torch.Tensor:
    terrain = getattr(base_env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None)
    if torch.is_tensor(levels):
        return levels.to(device=base_env.device, dtype=torch.float32).reshape(base_env.num_envs)
    return torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.float32)


def _command_tensor(base_env) -> torch.Tensor:
    try:
        command_term = base_env.command_manager.get_term("base_velocity")
    except Exception:
        command_term = None
    command = getattr(command_term, "command", None)
    if torch.is_tensor(command) and command.shape[0] == base_env.num_envs:
        return command.to(device=base_env.device, dtype=torch.float32)
    return torch.zeros(base_env.num_envs, 3, device=base_env.device, dtype=torch.float32)


def _zero_base_velocity_commands(base_env) -> None:
    try:
        command_term = base_env.command_manager.get_term("base_velocity")
    except Exception:
        return
    command = getattr(command_term, "vel_command_b", None)
    if torch.is_tensor(command):
        command.zero_()
    for attr_name, fill_value in (
        ("is_heading_env", False),
        ("is_yaw_env", False),
        ("is_standing_env", True),
    ):
        mask = getattr(command_term, attr_name, None)
        if torch.is_tensor(mask):
            mask.fill_(fill_value)


def _index_by_names(available: list[str], requested: list[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(available)}
    return [name_to_idx[name] for name in requested if name in name_to_idx]


def _body_ids_by_suffix(body_names: list[str], suffix: str) -> list[int]:
    return [idx for idx, name in enumerate(body_names) if str(name).endswith(suffix)]


def _pairwise_mean_distance(xy: torch.Tensor) -> torch.Tensor:
    if xy.shape[1] < 2:
        return torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype)
    diff = xy[:, :, None, :] - xy[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)
    pair_count = xy.shape[1] * (xy.shape[1] - 1)
    return dist.sum(dim=(1, 2)) / float(pair_count)


def _collect_step_metrics(base_env, actions: torch.Tensor | None) -> dict[str, torch.Tensor]:
    robot = base_env.scene["robot"]
    command = _command_tensor(base_env)
    root_lin_vel = robot.data.root_lin_vel_b
    root_ang_vel = robot.data.root_ang_vel_b
    metrics: dict[str, torch.Tensor] = {
        "cmd_abs_vx": torch.abs(command[:, 0]),
        "cmd_abs_vy": torch.abs(command[:, 1]),
        "cmd_abs_wz": torch.abs(command[:, 2]),
        "cmd_xy_norm": torch.linalg.norm(command[:, :2], dim=1),
        "tracking_lin_error": torch.linalg.norm(command[:, :2] - root_lin_vel[:, :2], dim=1),
        "tracking_yaw_error": torch.abs(command[:, 2] - root_ang_vel[:, 2]),
        "base_lin_xy_speed": torch.linalg.norm(root_lin_vel[:, :2], dim=1),
        "base_yaw_abs_speed": torch.abs(root_ang_vel[:, 2]),
        "base_height": robot.data.root_pos_w[:, 2],
    }

    if actions is not None:
        actions = actions.to(device=base_env.device, dtype=torch.float32)
        metrics["action_l2"] = torch.linalg.norm(actions, dim=1)
        metrics["action_mean_abs"] = torch.mean(torch.abs(actions), dim=1)
        metrics["action_abs_max"] = torch.max(torch.abs(actions), dim=1).values

    joint_names = list(getattr(robot, "joint_names", []))
    leg_ids = _index_by_names(joint_names, LEG_JOINT_NAMES)
    hip_ids = [idx for idx, name in enumerate(joint_names) if str(name).endswith("_hip_joint")]
    if leg_ids:
        ids = torch.as_tensor(leg_ids, device=base_env.device, dtype=torch.long)
        diff = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
        metrics["joint_default_l2"] = torch.linalg.norm(diff, dim=1)
        metrics["joint_default_mean_abs"] = torch.mean(torch.abs(diff), dim=1)
        metrics["joint_default_abs_max"] = torch.max(torch.abs(diff), dim=1).values
    if hip_ids:
        ids = torch.as_tensor(hip_ids, device=base_env.device, dtype=torch.long)
        diff = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
        metrics["hip_default_l2"] = torch.linalg.norm(diff, dim=1)
        metrics["hip_default_mean_abs"] = torch.mean(torch.abs(diff), dim=1)
        metrics["hip_default_abs_max"] = torch.max(torch.abs(diff), dim=1).values

    body_names = list(getattr(robot, "body_names", []))
    foot_ids = _body_ids_by_suffix(body_names, "_foot")
    if foot_ids:
        ids = torch.as_tensor(foot_ids, device=base_env.device, dtype=torch.long)
        foot_xy = robot.data.body_pos_w[:, ids, :2]
        metrics["foot_spread_x"] = foot_xy[:, :, 0].max(dim=1).values - foot_xy[:, :, 0].min(dim=1).values
        metrics["foot_spread_y"] = foot_xy[:, :, 1].max(dim=1).values - foot_xy[:, :, 1].min(dim=1).values
        metrics["foot_pairwise_dist_mean"] = _pairwise_mean_distance(foot_xy)

    contact_sensor = base_env.scene.sensors.get("contact_forces", None)
    if contact_sensor is not None:
        try:
            sensor_foot_ids, _ = contact_sensor.find_bodies([".*_foot"], preserve_order=True)
        except Exception:
            sensor_foot_ids = []
        if sensor_foot_ids:
            forces = contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :]
            force_mag = forces.norm(dim=-1).max(dim=1)[0]
            contact_mask = force_mag > 5.0
            metrics["foot_contact_count"] = contact_mask.float().sum(dim=1)
            metrics["foot_force_sum"] = force_mag.sum(dim=1)
            force_mean = force_mag.mean(dim=1)
            force_std = force_mag.std(dim=1, unbiased=False)
            metrics["foot_force_cv"] = force_std / torch.clamp(force_mean, min=1.0e-6)

    return metrics


def _append_step_stats(
    *,
    metrics: dict[str, torch.Tensor],
    terrain_names: list[str],
    terrain_levels: torch.Tensor,
    global_stats: dict[str, RunningStats],
    per_terrain_stats: dict[str, dict[str, RunningStats]],
) -> None:
    global_stats["terrain_level"].add(terrain_levels)
    for name, value in metrics.items():
        global_stats[name].add(value)

    unique_names = sorted(set(terrain_names))
    for terrain_name in unique_names:
        mask = torch.as_tensor(
            [name == terrain_name for name in terrain_names],
            device=terrain_levels.device,
            dtype=torch.bool,
        )
        if not mask.any():
            continue
        per_terrain_stats[terrain_name]["terrain_level"].add(terrain_levels[mask])
        for metric_name, value in metrics.items():
            per_terrain_stats[terrain_name][metric_name].add(value[mask])


def _summarize_episode_records(records: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "reward": _scalar_stats(records.get("reward", [])),
        "episode_length": _scalar_stats(records.get("length", [])),
        "episode_cost": _scalar_stats(records.get("cost", [])),
        "terrain_level_at_end": _scalar_stats(records.get("terrain_level", [])),
        "terminated_rate": _rate(records.get("terminated", [])),
        "timeout_rate": _rate(records.get("timeout", [])),
    }


def _rate(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _configure_scenario_env(env_cfg, scenario_name: str, stand_terrain: str) -> None:
    if scenario_name != "plane_stand":
        return
    env_cfg.terrain_profile = "rough"
    env_cfg.terrain_debug_single_family = stand_terrain
    if hasattr(env_cfg, "apply_experiment_overrides"):
        env_cfg.apply_experiment_overrides()
    if hasattr(env_cfg, "apply_eval_runtime_overrides"):
        env_cfg.apply_eval_runtime_overrides()
    command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    ranges = getattr(command_cfg, "ranges", None)
    if ranges is not None:
        ranges.lin_vel_x = (0.0, 0.0)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        ranges.start_curriculum_lin_x = (0.0, 0.0)
        ranges.start_curriculum_lin_y = (0.0, 0.0)
        ranges.start_curriculum_ang_z = (0.0, 0.0)
        ranges.max_curriculum_lin_x = (0.0, 0.0)
        ranges.max_curriculum_lin_y = (0.0, 0.0)
        ranges.max_curriculum_ang_z = (0.0, 0.0)
        ranges.heading_command_prob = 0.0
        ranges.yaw_command_prob = 0.0
        ranges.standing_command_prob = 1.0
    if command_cfg is not None:
        command_cfg.rel_standing_envs = 1.0


def _build_cfgs(resolved_task_name: str, scenario_name: str):
    env_cfg = parse_env_cfg(
        resolved_task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: CRLRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        resolved_task_name, args_cli
    )
    experiment_preset = load_experiment_preset(selection=args_cli.exp, file_path=args_cli.exp_file)
    if experiment_preset is not None:
        apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=experiment_preset)
        if hasattr(env_cfg, "apply_experiment_overrides"):
            env_cfg.apply_experiment_overrides()
        if hasattr(env_cfg, "apply_eval_runtime_overrides"):
            env_cfg.apply_eval_runtime_overrides()
        agent_cfg = cli_args.reapply_rsl_rl_cli_overrides(agent_cfg, args_cli)
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        print(
            f"[INFO] Applied experiment preset: {experiment_preset.name} ({experiment_preset.path})"
        )
    elif args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    _configure_scenario_env(env_cfg, scenario_name, args_cli.stand_terrain)
    return env_cfg, agent_cfg, experiment_preset


def _run_scenario(
    *,
    scenario_name: str,
    resolved_task_name: str,
    resume_path: str,
    agent_cfg: CRLRslRlOnPolicyRunnerCfg,
    env_cfg,
) -> dict[str, Any]:
    print(f"[INFO] Running diagnostic scenario: {scenario_name}", flush=True)
    env = gym.make(resolved_task_name, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if initialize_domain_randomization_curriculum(env.unwrapped):
        env.reset()
        print("[INFO] Initialized domain-randomization curriculum and reset the environment.", flush=True)

    env = CRLRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    ppo_runner = OnPolicyRunnerWithExtractor(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=base_env.device)

    obs, _extras = env.get_observations()
    target_episodes = max(int(args_cli.max_episodes), 1)
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else None

    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    cur_episode_cost = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    overall_episode_records: dict[str, list[float]] = defaultdict(list)
    per_terrain_episode_records: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    global_step_stats: dict[str, RunningStats] = defaultdict(RunningStats)
    per_terrain_step_stats: dict[str, dict[str, RunningStats]] = defaultdict(
        lambda: defaultdict(RunningStats)
    )
    per_term_step_stats: dict[str, RunningStats] = defaultdict(RunningStats)

    completed = 0
    step_count = 0
    start_wall = time.time()
    while completed < target_episodes:
        if max_steps is not None and step_count >= int(max_steps):
            break

        if scenario_name == "plane_stand":
            _zero_base_velocity_commands(base_env)
            obs, _ = env.get_observations()

        pre_terrain_names = _terrain_names(base_env)
        pre_terrain_levels = _terrain_levels(base_env)

        step_start = time.time()
        with torch.inference_mode():
            actions = policy(obs, hist_encoding=True)

        step_metrics = _collect_step_metrics(base_env, actions)
        _append_step_stats(
            metrics=step_metrics,
            terrain_names=pre_terrain_names,
            terrain_levels=pre_terrain_levels,
            global_stats=global_step_stats,
            per_terrain_stats=per_terrain_step_stats,
        )

        obs, rews, dones, extras = env.step(actions)
        if scenario_name == "plane_stand":
            _zero_base_velocity_commands(base_env)

        step_total_cost, step_term_costs = _extract_step_costs(extras, env.device, env.num_envs)
        for name, value in step_term_costs.items():
            per_term_step_stats[name].add(value)

        rews = rews.reshape(env.num_envs)
        dones = dones.reshape(env.num_envs)
        cur_reward_sum += rews
        cur_episode_length += 1
        cur_episode_cost += step_total_cost
        step_count += 1

        terminated = getattr(base_env, "reset_terminated", torch.zeros_like(dones)).reshape(
            env.num_envs
        )
        timeouts = getattr(base_env, "reset_time_outs", torch.zeros_like(dones)).reshape(
            env.num_envs
        )
        new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if new_ids.numel() > 0:
            for env_id in new_ids.detach().cpu().tolist():
                terrain_name = pre_terrain_names[env_id] if env_id < len(pre_terrain_names) else "unknown"
                terrain_level = float(pre_terrain_levels[env_id].detach().cpu().item())
                values = {
                    "reward": float(cur_reward_sum[env_id].detach().cpu().item()),
                    "length": float(cur_episode_length[env_id].detach().cpu().item()),
                    "cost": float(cur_episode_cost[env_id].detach().cpu().item()),
                    "terrain_level": terrain_level,
                    "terminated": float(bool(terminated[env_id].detach().cpu().item())),
                    "timeout": float(bool(timeouts[env_id].detach().cpu().item())),
                }
                for key, value in values.items():
                    overall_episode_records[key].append(value)
                    per_terrain_episode_records[terrain_name][key].append(value)
            completed = len(overall_episode_records["reward"])
            cur_reward_sum[new_ids] = 0.0
            cur_episode_length[new_ids] = 0.0
            cur_episode_cost[new_ids] = 0.0

        sleep_time = base_env.step_dt - (time.time() - step_start)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        progress_interval = max(int(args_cli.progress_interval), 0)
        if progress_interval and step_count % progress_interval == 0:
            print(
                f"[INFO] {scenario_name}: steps={step_count}, "
                f"episodes={completed}/{target_episodes}",
                flush=True,
            )

    env.close()

    return {
        "scenario": scenario_name,
        "episodes_completed": completed,
        "steps_collected": step_count,
        "wall_time_s": time.time() - start_wall,
        "overall": {
            "episodes": _summarize_episode_records(overall_episode_records),
            "step": _stats_map_to_dict(global_step_stats),
            "cost_terms_step": _stats_map_to_dict(per_term_step_stats),
        },
        "per_terrain": {
            terrain_name: {
                "episodes": _summarize_episode_records(records),
                "step": _stats_map_to_dict(per_terrain_step_stats.get(terrain_name, {})),
            }
            for terrain_name, records in sorted(per_terrain_episode_records.items())
        },
    }


def main() -> None:
    if args_cli.task is None:
        raise ValueError("Please provide --task for Galileo diagnostics.")

    resolved_task_name = _resolve_eval_task_name(args_cli.task)
    base_env_cfg, base_agent_cfg, experiment_preset = _build_cfgs(resolved_task_name, "mixed")
    log_root_path = build_log_root_path(base_agent_cfg.experiment_name)
    resume_path = resolve_checkpoint_path(
        task_name=resolved_task_name,
        log_root_path=log_root_path,
        load_run=base_agent_cfg.load_run,
        load_checkpoint=base_agent_cfg.load_checkpoint,
        checkpoint=args_cli.checkpoint,
        use_pretrained_checkpoint=args_cli.use_pretrained_checkpoint,
        algo_name=getattr(args_cli, "algo", None),
    )
    if not resume_path:
        raise RuntimeError("Unable to resolve a checkpoint for diagnostics.")
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Diagnostic checkpoint: {resume_path}", flush=True)

    requested_modes = [mode.strip() for mode in args_cli.modes.split(",") if mode.strip()]
    valid_modes = {"mixed", "plane_stand"}
    unknown_modes = sorted(set(requested_modes) - valid_modes)
    if unknown_modes:
        raise ValueError(f"Unknown diagnostic modes: {unknown_modes}. Valid: {sorted(valid_modes)}")

    if args_cli.summary_dir is not None:
        summary_path = Path(args_cli.summary_dir).expanduser().resolve() / "stage1_diagnostics.json"
    else:
        summary_path = build_evaluation_output_path(
            log_dir,
            resolved_task_name,
            resume_path,
            summary_tag=args_cli.summary_tag,
            filename="stage1_diagnostics.json",
        )

    manifest = build_run_manifest(
        stage="stage1_diagnostics",
        task_name=resolved_task_name,
        log_dir=log_dir,
        agent_cfg=base_agent_cfg,
        env_cfg=base_env_cfg,
        args=args_cli,
        preset=experiment_preset,
        checkpoint_path=resume_path,
        repo_root=REPO_ROOT,
        extra={"summary_path": str(summary_path.resolve())},
    )

    def _write_summary(scenarios: dict[str, Any], *, complete: bool) -> None:
        summary = {
            "requested_task_name": args_cli.task,
            "resolved_task_name": resolved_task_name,
            "checkpoint_path": str(Path(resume_path).resolve()),
            "requested_modes": requested_modes,
            "completed_modes": sorted(scenarios),
            "complete": complete,
            "scenarios": scenarios,
            "manifest": manifest,
        }
        write_json_artifact(summary_path, summary)
        write_json_artifact(summary_path.parent / "run_manifest.json", manifest)

    scenarios: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            existing_scenarios = existing_summary.get("scenarios", {})
            if isinstance(existing_scenarios, dict):
                scenarios.update(existing_scenarios)
                print(
                    f"[INFO] Loaded existing diagnostics with scenarios: {sorted(scenarios)}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[WARN] Could not merge existing diagnostics: {exc}", flush=True)
    for mode in requested_modes:
        env_cfg, agent_cfg, _preset = _build_cfgs(resolved_task_name, mode)
        scenarios[mode] = _run_scenario(
            scenario_name=mode,
            resolved_task_name=resolved_task_name,
            resume_path=resume_path,
            agent_cfg=agent_cfg,
            env_cfg=env_cfg,
        )
        _write_summary(scenarios, complete=False)
        print(f"[INFO] Wrote partial diagnostics to: {summary_path}", flush=True)

    _write_summary(scenarios, complete=True)

    print("[INFO] Stage-1 diagnostics complete.")
    print(f"[INFO] Wrote diagnostics to: {summary_path}")
    print_dict(
        {
            mode: {
                "episodes": data["episodes_completed"],
                "reward_mean": data["overall"]["episodes"]["reward"]["mean"],
                "length_mean": data["overall"]["episodes"]["episode_length"]["mean"],
                "terminated_rate": data["overall"]["episodes"]["terminated_rate"],
            }
            for mode, data in scenarios.items()
        },
        nesting=2,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
