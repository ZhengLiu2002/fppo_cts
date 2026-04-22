# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with an RSL-RL agent."""

import argparse
import os
import shutil
import sys
import warnings

try:
    from scripts.rsl_rl.experiment_manager import (
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
    )
    from scripts.rsl_rl.runtime import (
        bootstrap_repo_paths,
        build_log_root_path,
        configure_safe_play_args,
        resolve_checkpoint_path,
    )
except ImportError:
    from experiment_manager import (  # type: ignore
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
    )
    from runtime import (  # type: ignore
        bootstrap_repo_paths,
        build_log_root_path,
        configure_safe_play_args,
        resolve_checkpoint_path,
    )

bootstrap_repo_paths(__file__)

from isaaclab.app import AppLauncher

warnings.filterwarnings(
    "ignore",
    message=r".*Overriding environment Isaac-Parkour-Galileo-v0 already in registry.*",
    category=UserWarning,
)

from scripts.rsl_rl import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RSL-RL policy.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during playback."
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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
    "--force_gui",
    action="store_true",
    default=False,
    help="Force GUI mode and skip auto headless fallback.",
)
parser.add_argument(
    "--export_only",
    action="store_true",
    default=False,
    help="Export deployment artifacts and exit without running the playback loop.",
)
parser.add_argument(
    "--export_dir",
    type=str,
    default=None,
    help="Directory to store exported deployment artifacts. Defaults to <checkpoint_dir>/exported_policy.",
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

configure_safe_play_args(args_cli)

print(
    "[INFO] Play launch selection: "
    f"force_gui={args_cli.force_gui}, "
    f"headless={args_cli.headless}, "
    f"DISPLAY={os.environ.get('DISPLAY', '')}, "
    f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')}, "
    f"LIVESTREAM={os.environ.get('LIVESTREAM', '')}"
)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import gymnasium as gym
import numpy as np
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from crl_tasks.tasks.galileo.config.agents.rsl_rl_cfg import CRLRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.exporter import (
    export_inference_cfg,
    export_policy_as_onnx_dual_input,
    export_policy_as_onnx_grouped_inputs,
)
from scripts.rsl_rl.vecenv_wrapper import CRLRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _print_terrain_summary(base_env):
    """Log terrain generator shape and current terrain level/type distribution."""
    try:
        terrain = base_env.scene.terrain
        levels = terrain.terrain_levels.cpu().numpy()
        types = terrain.terrain_types.cpu().numpy()
        uniq_lvl, lvl_counts = np.unique(levels, return_counts=True)
        uniq_typ, typ_counts = np.unique(types, return_counts=True)
        gen_cfg = terrain.cfg.terrain_generator
        print(
            "[INFO] Terrain generator layout:"
            f" rows={getattr(gen_cfg, 'num_rows', '?')},"
            f" cols={getattr(gen_cfg, 'num_cols', '?')},"
            f" curriculum={getattr(gen_cfg, 'curriculum', '?')},"
            f" difficulty_range={getattr(gen_cfg, 'difficulty_range', '?')}"
        )
        print(
            "[INFO] Terrain level histogram (level:count):",
            {int(k): int(v) for k, v in zip(uniq_lvl, lvl_counts)},
        )
        print(
            "[INFO] Terrain type histogram (col_idx:count):",
            {int(k): int(v) for k, v in zip(uniq_typ, typ_counts)},
        )
    except Exception as exc:  # pragma: no cover - debug aid
        print(f"[WARN] Unable to print terrain summary: {exc}")


def main():
    """Play a trained RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: CRLRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    experiment_preset = load_experiment_preset(selection=args_cli.exp, file_path=args_cli.exp_file)
    if experiment_preset is not None:
        apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=experiment_preset)
        agent_cfg = cli_args.reapply_rsl_rl_cli_overrides(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        print(
            f"[INFO] Applied experiment preset: {experiment_preset.name} ({experiment_preset.path})"
        )

    # specify directory for logging experiments
    log_root_path = build_log_root_path(agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = resolve_checkpoint_path(
        task_name=args_cli.task,
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Play 模式尽量避免超时重置，只在到达最后目标或摔倒时重置（训练不受影响）
    try:
        base_env = env.unwrapped
        if hasattr(base_env, "max_episode_length"):
            prop = getattr(type(base_env), "max_episode_length", None)
            can_set = not isinstance(prop, property) or prop.fset is not None
            if can_set:
                new_len = int(
                    base_env.max_episode_length * 100
                )  # effectively disable time-out for play
                base_env.max_episode_length = new_len
                if hasattr(base_env, "cfg") and hasattr(base_env.cfg, "episode_length_s"):
                    base_env.cfg.episode_length_s = base_env.cfg.episode_length_s * 100.0
                print(
                    f"[INFO] Play mode: time-out relaxed to {new_len} steps (goal/fall still terminate)."
                )
            else:
                # Read-only property (e.g., CRLManagerBasedRLEnv); keep original to avoid warnings.
                print("[INFO] Play mode: max_episode_length is read-only; skip relaxing time-out.")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to relax play time-out: {exc}")

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
        print("[INFO] Recording videos during playback.")
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
    policy_nn = ppo_runner.alg.policy
    should_export = args_cli.export_only or args_cli.export_dir is not None
    if should_export:
        export_model_dir = args_cli.export_dir or os.path.join(
            os.path.dirname(resume_path), "exported_policy"
        )
        run_policy_cfg_path = os.path.join(log_dir, "policy.yaml")
        if not os.path.isfile(run_policy_cfg_path):
            raise FileNotFoundError(
                "Run-local policy.yaml is missing. Re-train this run with the current pipeline "
                "so deployment config is generated from the live training config."
            )
        export_cfg = export_inference_cfg(
            env,
            env_cfg,
            export_model_dir,
            agent_cfg=agent_cfg,
            actor_critic=policy_nn,
        )
        shutil.copy2(run_policy_cfg_path, os.path.join(export_model_dir, "policy.yaml"))
        export_input_order = list(export_cfg.get("export_input_order") or export_cfg["input_names"])
        export_input_dims = {
            name: int(export_cfg["export_input_dims"][name]) for name in export_input_order
        }
        if len(export_input_order) == 1:
            export_policy_as_onnx_dual_input(
                policy_nn,
                normalizer=ppo_runner.obs_normalizer,
                path=export_model_dir,
                filename="policy.onnx",
                actor_obs_dim=export_input_dims[export_input_order[0]],
            )
        else:
            export_policy_as_onnx_grouped_inputs(
                policy_nn,
                normalizer=ppo_runner.obs_normalizer,
                path=export_model_dir,
                filename="policy.onnx",
                input_groups=[(name, export_input_dims[name]) for name in export_input_order],
            )
        print(f"[INFO] Exported deployment artifacts to: {export_model_dir}")

    if args_cli.export_only:
        env.close()
        # Omniverse teardown can hang after export-only headless runs even though
        # the artifacts are already written. Flush logs and terminate cleanly
        # from the CLI user's perspective instead of waiting on the viewer stack.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    dt = env.unwrapped.step_dt
    # reset environment
    obs, extras = env.get_observations()
    _print_terrain_summary(env.unwrapped)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs, hist_encoding=True)
        obs, _, _, extras = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
