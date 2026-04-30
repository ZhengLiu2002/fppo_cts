# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump effective Galileo parameters after env construction and reset."""

from __future__ import annotations

import argparse

try:
    from scripts.rsl_rl.experiment_manager import (
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
    )
    from scripts.rsl_rl.runtime import bootstrap_repo_paths
except ImportError:
    from experiment_manager import (  # type: ignore
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
    )
    from runtime import bootstrap_repo_paths  # type: ignore

REPO_ROOT = bootstrap_repo_paths(__file__)

from isaaclab.app import AppLauncher
from scripts.rsl_rl import cli_args


parser = argparse.ArgumentParser(
    description="Dump effective Galileo sim/config parameters after reset."
)
parser.add_argument("--task", type=str, default="Isaac-Galileo-CTS-v0", help="Gym task id to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to dump.")
parser.add_argument("--all_envs", action="store_true", default=False, help="Dump all environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed passed to cfg and env.reset.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--material_detail",
    type=str,
    default="summary",
    choices=["summary", "per_shape", "none"],
    help="How much material data to export.",
)
parser.add_argument("--output", type=str, default=None, help="Output file path.")
parser.add_argument("--format", type=str, default="json", choices=["json", "yaml"], help="Output format.")

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

import json
import os
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import crl_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from scripts.rsl_rl.effective_params import build_effective_config_summary


def _to_serializable(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "__name__"):
        return value.__name__
    return value


def _select_env(value: Any, env_id: int, num_envs: int):
    if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == num_envs:
        return value[env_id]
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == num_envs:
        return value[env_id]
    return value


def _safe_call(obj: Any, method_name: str):
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def _material_summary(materials: Any):
    if isinstance(materials, torch.Tensor):
        arr = materials.detach().cpu().numpy()
    else:
        arr = np.asarray(materials)
    if arr.size == 0:
        return {}
    arr = np.reshape(arr, (-1, arr.shape[-1]))
    summary = {}
    for index, name in enumerate(("static_friction", "dynamic_friction", "restitution")):
        if index >= arr.shape[-1]:
            continue
        col = arr[:, index]
        summary[name] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
        }
    return summary


def _material_cfg_to_dict(cfg: Any):
    if cfg is None:
        return None
    keys = (
        "static_friction",
        "dynamic_friction",
        "restitution",
        "friction_combine_mode",
        "restitution_combine_mode",
    )
    return {key: _to_serializable(getattr(cfg, key, None)) for key in keys}


def _actuator_indices(actuator: Any, num_joints: int):
    indices = getattr(actuator, "joint_indices", None)
    if indices is None or isinstance(indices, slice):
        return list(range(num_joints))
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().tolist()
    return list(indices)


def _actuator_param(actuator: Any, name: str, env_id: int, num_envs: int):
    value = getattr(actuator, name, None)
    if value is None:
        return None
    return _to_serializable(_select_env(value, env_id, num_envs))


def _actuator_delay_steps(actuator: Any, env_id: int):
    for buffer_name in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
        buffer = getattr(actuator, buffer_name, None)
        time_lags = getattr(buffer, "time_lags", None)
        if time_lags is not None:
            try:
                return int(time_lags[env_id].item())
            except Exception:
                return _to_serializable(time_lags)
    return None


def _resolve_terrain_names(unwrapped: Any):
    try:
        from crl_isaaclab.terrains.runtime import resolve_env_terrain_names

        names = resolve_env_terrain_names(unwrapped.scene.terrain)
        return list(names) if names is not None else None
    except Exception:
        return None


def _command_dump(unwrapped: Any, env_id: int):
    command_manager = getattr(unwrapped, "command_manager", None)
    if command_manager is None:
        return {}
    try:
        command_term = command_manager.get_term("base_velocity")
    except Exception:
        command_term = getattr(command_manager, "_terms", {}).get("base_velocity")
    if command_term is None:
        return {}
    return {
        "command": _to_serializable(_select_env(getattr(command_term, "command", None), env_id, unwrapped.scene.num_envs)),
        "heading_target": _to_serializable(
            _select_env(getattr(command_term, "heading_target", None), env_id, unwrapped.scene.num_envs)
        ),
        "is_heading_env": _to_serializable(
            _select_env(getattr(command_term, "is_heading_env", None), env_id, unwrapped.scene.num_envs)
        ),
        "is_yaw_env": _to_serializable(
            _select_env(getattr(command_term, "is_yaw_env", None), env_id, unwrapped.scene.num_envs)
        ),
        "is_standing_env": _to_serializable(
            _select_env(getattr(command_term, "is_standing_env", None), env_id, unwrapped.scene.num_envs)
        ),
    }


def _build_env_dump(env: gym.Env, env_id: int, material_detail: str):
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    num_envs = scene.num_envs
    if hasattr(scene, "articulations") and "robot" in scene.articulations:
        robot = scene.articulations["robot"]
    else:
        robot = scene["robot"]

    joint_names = list(getattr(robot, "joint_names", []))
    body_names = list(getattr(robot, "body_names", []))
    physx_view = getattr(robot, "root_physx_view", None)

    masses = _select_env(_safe_call(physx_view, "get_masses"), env_id, num_envs)
    coms = _select_env(_safe_call(physx_view, "get_coms"), env_id, num_envs)
    inertias = _select_env(_safe_call(physx_view, "get_inertias"), env_id, num_envs)
    materials = _select_env(_safe_call(physx_view, "get_material_properties"), env_id, num_envs)

    default_mass = _select_env(getattr(robot.data, "default_mass", None), env_id, num_envs)
    default_inertia = _select_env(getattr(robot.data, "default_inertia", None), env_id, num_envs)
    default_joint_pos = _select_env(getattr(robot.data, "default_joint_pos", None), env_id, num_envs)

    body_info = []
    for body_index, body_name in enumerate(body_names):
        body_mass = masses[body_index] if masses is not None else None
        body_default_mass = default_mass[body_index] if default_mass is not None else None
        body_info.append(
            {
                "name": body_name,
                "mass": float(body_mass) if body_mass is not None else None,
                "default_mass": float(body_default_mass) if body_default_mass is not None else None,
                "delta_mass": (
                    float(body_mass - body_default_mass)
                    if body_mass is not None and body_default_mass is not None
                    else None
                ),
                "com": _to_serializable(coms[body_index]) if coms is not None else None,
                "inertia": _to_serializable(inertias[body_index]) if inertias is not None else None,
                "default_inertia": (
                    _to_serializable(default_inertia[body_index]) if default_inertia is not None else None
                ),
            }
        )

    actuator_entries = {}
    joint_params = {name: {} for name in joint_names}
    for actuator_name, actuator in getattr(robot, "actuators", {}).items():
        indices = _actuator_indices(actuator, len(joint_names))
        params = {
            "stiffness": _actuator_param(actuator, "stiffness", env_id, num_envs),
            "damping": _actuator_param(actuator, "damping", env_id, num_envs),
            "effort_limit": _actuator_param(actuator, "effort_limit", env_id, num_envs),
            "velocity_limit": _actuator_param(actuator, "velocity_limit", env_id, num_envs),
            "friction": _actuator_param(actuator, "friction", env_id, num_envs),
            "armature": _actuator_param(actuator, "armature", env_id, num_envs),
        }
        actuator_entries[actuator_name] = {
            "type": type(actuator).__name__,
            "joint_names": _to_serializable(getattr(actuator, "joint_names", None)),
            "joint_indices": indices,
            "min_delay": getattr(getattr(actuator, "cfg", None), "min_delay", None),
            "max_delay": getattr(getattr(actuator, "cfg", None), "max_delay", None),
            "delay_steps": _actuator_delay_steps(actuator, env_id),
            **params,
        }

        for param_name, values in params.items():
            if values is None:
                continue
            if isinstance(values, list) and len(values) == len(indices):
                for local_index, joint_index in enumerate(indices):
                    joint_params[joint_names[joint_index]][param_name] = values[local_index]
            else:
                for joint_index in indices:
                    joint_params[joint_names[joint_index]][param_name] = values

    material_data = {}
    if material_detail != "none" and materials is not None:
        material_data["summary"] = _material_summary(materials)
        if material_detail == "per_shape":
            material_data["per_shape"] = _to_serializable(materials)

    terrain_names = _resolve_terrain_names(unwrapped)
    terrain_levels = getattr(scene.terrain, "terrain_levels", None) if hasattr(scene, "terrain") else None
    terrain_types = getattr(scene.terrain, "terrain_types", None) if hasattr(scene, "terrain") else None

    return {
        "env_id": env_id,
        "terrain": {
            "name": terrain_names[env_id] if terrain_names and env_id < len(terrain_names) else None,
            "level": _to_serializable(_select_env(terrain_levels, env_id, num_envs)),
            "type_index": _to_serializable(_select_env(terrain_types, env_id, num_envs)),
        },
        "command": _command_dump(unwrapped, env_id),
        "robot": {
            "joint_names": joint_names,
            "body_names": body_names,
            "default_joint_pos": _to_serializable(default_joint_pos),
            "bodies": body_info,
            "materials": material_data,
        },
        "actuators": actuator_entries,
        "joint_params": joint_params,
    }


def _write_dump(data: dict[str, Any], output_path: str, output_format: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    serializable = _to_serializable(data)
    if output_format == "yaml":
        try:
            import yaml

            with open(output_path, "w", encoding="utf-8") as file_obj:
                yaml.safe_dump(serializable, file_obj, sort_keys=False)
            return
        except Exception as exc:
            print(f"[WARN] Failed to write YAML ({exc}); falling back to JSON.")
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(serializable, file_obj, indent=2, ensure_ascii=False)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    experiment_preset = load_experiment_preset(selection=args_cli.exp, file_path=args_cli.exp_file)
    if experiment_preset is not None:
        apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=experiment_preset)
        if hasattr(env_cfg, "apply_experiment_overrides"):
            env_cfg.apply_experiment_overrides()
        agent_cfg = cli_args.reapply_rsl_rl_cli_overrides(agent_cfg, args_cli)
        print(f"[INFO] Applied experiment preset: {experiment_preset.name} ({experiment_preset.path})")
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    unwrapped = env.unwrapped
    num_envs = unwrapped.scene.num_envs
    if not args_cli.all_envs and (args_cli.env_id < 0 or args_cli.env_id >= num_envs):
        raise ValueError(f"env_id {args_cli.env_id} out of range [0, {num_envs - 1}]")

    sim_cfg = unwrapped.sim.cfg
    env_ids = list(range(num_envs)) if args_cli.all_envs else [args_cli.env_id]
    data = {
        "task": args_cli.task,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args_cli.seed,
        "device": getattr(unwrapped, "device", args_cli.device),
        "config_summary": build_effective_config_summary(unwrapped.cfg, agent_cfg),
        "sim": {
            "physics_dt": float(getattr(unwrapped, "physics_dt", 0.0)),
            "decimation": int(unwrapped.cfg.decimation),
            "step_dt": float(getattr(unwrapped, "step_dt", 0.0)),
            "gravity": _to_serializable(getattr(sim_cfg, "gravity", None)),
            "use_fabric": _to_serializable(getattr(sim_cfg, "use_fabric", None)),
        },
        "terrain_physics_material": _material_cfg_to_dict(getattr(unwrapped.cfg.sim, "physics_material", None)),
        "envs": [
            _build_env_dump(env, env_id, material_detail=args_cli.material_detail)
            for env_id in env_ids
        ],
    }

    output_path = args_cli.output
    if output_path is None:
        safe_task = args_cli.task.replace(":", "_").replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", "effective_params")
        output_path = os.path.join(output_dir, f"{safe_task}_{timestamp}.{args_cli.format}")

    _write_dump(data, output_path, args_cli.format)
    print(f"[INFO] Effective parameter dump saved to: {os.path.abspath(output_path)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
