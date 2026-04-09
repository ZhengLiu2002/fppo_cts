# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a checkpoint with an RSL-RL agent."""

import argparse
import statistics

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

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during evaluation."
)
parser.add_argument(
    "--video_length", type=int, default=500, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time", action="store_true", default=False, help="Run in real-time, if possible."
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=512,
    help="Stop evaluation after this many completed episodes.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional hard cap on environment steps. Defaults to no explicit cap.",
)
parser.add_argument(
    "--summary_tag",
    type=str,
    default=None,
    help="Optional tag appended to the evaluation output directory.",
)
parser.add_argument(
    "--summary_dir",
    type=str,
    default=None,
    help="Optional explicit directory for evaluation artifacts.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
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

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
from pathlib import Path
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from crl_tasks.tasks.galileo.config.agents.rsl_rl_cfg import CRLRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.vecenv_wrapper import CRLRslRlVecEnvWrapper

import crl_tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _scalar_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": float(mean),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _extract_step_costs(extras: dict, device: torch.device | str, num_envs: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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


def main():
    """Evaluate a trained RSL-RL agent."""
    resolved_task_name = _resolve_eval_task_name(args_cli.task)

    env_cfg = parse_env_cfg(
        resolved_task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: CRLRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(resolved_task_name, args_cli)
    experiment_preset = load_experiment_preset(selection=args_cli.exp, file_path=args_cli.exp_file)
    if experiment_preset is not None:
        apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=experiment_preset)
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

    # specify directory for logging experiments
    log_root_path = build_log_root_path(agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = resolve_checkpoint_path(
        task_name=resolved_task_name,
        log_root_path=log_root_path,
        load_run=agent_cfg.load_run,
        load_checkpoint=agent_cfg.load_checkpoint,
        checkpoint=args_cli.checkpoint,
        use_pretrained_checkpoint=args_cli.use_pretrained_checkpoint,
        algo_name=getattr(args_cli, "algo", None),
    )
    if not resume_path:
        print(
            "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
        )
        return

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(
        resolved_task_name,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = CRLRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerWithExtractor(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    # reset environment
    obs, _extras = env.get_observations()
    timestep = 0
    reward_buffer: list[float] = []
    length_buffer: list[float] = []
    episode_cost_buffer: list[float] = []
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    cur_episode_cost = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    cur_term_episode_costs: dict[str, torch.Tensor] = {}
    per_term_episode_costs: dict[str, list[float]] = {}
    per_term_step_sum: dict[str, float] = {}
    per_term_step_positive: dict[str, float] = {}
    total_step_cost_sum = 0.0
    total_step_positive = 0.0
    total_step_samples = 0
    step_count = 0

    max_steps = args_cli.max_steps if args_cli.max_steps is not None else None
    target_episodes = max(int(args_cli.max_episodes), 1)

    while len(reward_buffer) < target_episodes:
        if max_steps is not None and step_count >= int(max_steps):
            break
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs, hist_encoding=True)
        obs, rews, dones, extras = env.step(actions)
        step_total_cost, step_term_costs = _extract_step_costs(extras, env.device, env.num_envs)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        rews = rews.reshape(env.num_envs)
        dones = dones.reshape(env.num_envs)
        cur_reward_sum += rews
        cur_episode_length += 1
        cur_episode_cost += step_total_cost
        total_step_cost_sum += float(step_total_cost.sum().item())
        total_step_positive += float((step_total_cost > 0).float().sum().item())
        total_step_samples += int(step_total_cost.numel())
        step_count += 1

        for name, value in step_term_costs.items():
            if name not in cur_term_episode_costs:
                cur_term_episode_costs[name] = torch.zeros(
                    env.num_envs, dtype=torch.float32, device=env.device
                )
            cur_term_episode_costs[name] += value
            per_term_step_sum[name] = per_term_step_sum.get(name, 0.0) + float(value.sum().item())
            per_term_step_positive[name] = per_term_step_positive.get(name, 0.0) + float(
                (value > 0).float().sum().item()
            )

        new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if new_ids.numel() > 0:
            reward_buffer.extend(float(v) for v in cur_reward_sum[new_ids].detach().cpu().tolist())
            length_buffer.extend(
                float(v) for v in cur_episode_length[new_ids].detach().cpu().tolist()
            )
            episode_cost_buffer.extend(
                float(v) for v in cur_episode_cost[new_ids].detach().cpu().tolist()
            )
            for name, buffer in cur_term_episode_costs.items():
                term_episode_values = per_term_episode_costs.setdefault(name, [])
                term_episode_values.extend(
                    float(v) for v in buffer[new_ids].detach().cpu().tolist()
                )
                buffer[new_ids] = 0.0
            cur_reward_sum[new_ids] = 0.0
            cur_episode_length[new_ids] = 0.0
            cur_episode_cost[new_ids] = 0.0

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if len(reward_buffer) >= target_episodes:
            break

    env.close()
    summary_path = (
        Path(args_cli.summary_dir).expanduser().resolve() / "summary.json"
        if args_cli.summary_dir is not None
        else build_evaluation_output_path(
            log_dir,
            resolved_task_name,
            resume_path,
            summary_tag=args_cli.summary_tag,
        )
    )
    evaluation_summary = {
        "requested_task_name": args_cli.task,
        "resolved_task_name": resolved_task_name,
        "checkpoint_path": str(Path(resume_path).resolve()),
        "episodes_completed": len(reward_buffer),
        "steps_collected": step_count,
        "reward": _scalar_stats(reward_buffer),
        "episode_length": _scalar_stats(length_buffer),
        "cost": {
            "mean_step_cost": (total_step_cost_sum / total_step_samples) if total_step_samples else None,
            "step_positive_rate": (
                total_step_positive / total_step_samples if total_step_samples else None
            ),
            "episode_cost": _scalar_stats(episode_cost_buffer),
            "per_term": {
                name: {
                    "mean_step_cost": (
                        per_term_step_sum[name] / total_step_samples if total_step_samples else None
                    ),
                    "step_positive_rate": (
                        per_term_step_positive[name] / total_step_samples
                        if total_step_samples
                        else None
                    ),
                    "episode_cost": _scalar_stats(values),
                }
                for name, values in sorted(per_term_episode_costs.items())
            },
        },
        "manifest": build_run_manifest(
            stage="eval",
            task_name=resolved_task_name,
            log_dir=log_dir,
            agent_cfg=agent_cfg,
            env_cfg=env_cfg,
            args=args_cli,
            preset=experiment_preset,
            checkpoint_path=resume_path,
            repo_root=REPO_ROOT,
            extra={
                "summary_path": str(summary_path.resolve()),
                "requested_task_name": args_cli.task,
            },
        ),
    }
    write_json_artifact(summary_path, evaluation_summary)
    write_json_artifact(summary_path.parent / "run_manifest.json", evaluation_summary["manifest"])

    reward_stats = evaluation_summary["reward"]
    length_stats = evaluation_summary["episode_length"]
    cost_stats = evaluation_summary["cost"]
    print(
        "Mean reward: {mean:.2f}+-{std:.2f}".format(
            mean=reward_stats["mean"] or 0.0,
            std=reward_stats["std"] or 0.0,
        )
    )
    print(
        "Mean episode length: {mean:.2f}+-{std:.2f}".format(
            mean=length_stats["mean"] or 0.0,
            std=length_stats["std"] or 0.0,
        )
    )
    print(
        "Mean step cost: {mean:.4f}, positive-rate: {rate:.2%}".format(
            mean=cost_stats["mean_step_cost"] or 0.0,
            rate=cost_stats["step_positive_rate"] or 0.0,
        )
    )
    print(f"[INFO] Wrote evaluation summary to: {summary_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
