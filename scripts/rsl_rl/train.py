# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RSL-RL agents in Isaac Lab."""

import argparse
import os
import sys

try:
    from scripts.rsl_rl.experiment_manager import (
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
        write_experiment_metadata,
    )
    from scripts.rsl_rl.algorithms.registry import get_algorithm_spec
    from scripts.rsl_rl.runtime import (
        bootstrap_repo_paths,
        build_log_root_path,
        configure_torch_backends,
        create_run_directory_name,
        dump_pickle,
        ensure_min_rsl_rl_version,
        resolve_checkpoint_path,
    )
except ImportError:
    from experiment_manager import (  # type: ignore
        apply_experiment_preset,
        available_experiment_presets,
        load_experiment_preset,
        write_experiment_metadata,
    )
    from algorithms.registry import get_algorithm_spec  # type: ignore
    from runtime import (  # type: ignore
        bootstrap_repo_paths,
        build_log_root_path,
        configure_torch_backends,
        create_run_directory_name,
        dump_pickle,
        ensure_min_rsl_rl_version,
        resolve_checkpoint_path,
    )

bootstrap_repo_paths(__file__)

from isaaclab.app import AppLauncher

from scripts.rsl_rl import cli_args


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length", type=int, default=200, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

ensure_min_rsl_rl_version(distributed=args_cli.distributed)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from crl_tasks.tasks.galileo.config.agents.rsl_rl_cfg import CRLRslRlOnPolicyRunnerCfg
from scripts.rsl_rl.vecenv_wrapper import CRLRslRlVecEnvWrapper

# import isaaclab_tasks  # noqa: F401
import crl_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from crl_isaaclab.envs import CRLManagerBasedRLEnv

# PLACEHOLDER: Extension template (do not remove this comment)

configure_torch_backends(torch)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: CRLManagerBasedRLEnv | ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: CRLRslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent."""

    # override configurations with non-hydra CLI arguments
    experiment_preset = load_experiment_preset(selection=args_cli.exp, file_path=args_cli.exp_file)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if experiment_preset is not None:
        apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=experiment_preset)
        agent_cfg = cli_args.reapply_rsl_rl_cli_overrides(agent_cfg, args_cli)
        print(
            f"[INFO] Applied experiment preset: {experiment_preset.name} ({experiment_preset.path})"
        )
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    algorithm_training_type = get_algorithm_spec(agent_cfg.algorithm.class_name).training_type

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = build_log_root_path(agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    run_dir_name, needs_exact_name_log = create_run_directory_name(
        agent_cfg.run_name,
        experiment_slug=experiment_preset.slug if experiment_preset is not None else None,
    )
    if needs_exact_name_log:
        # The Ray Tune workflow extracts experiment name using the logging line below.
        print(f"Exact experiment name requested from command line: {run_dir_name}")
    log_dir = os.path.join(log_root_path, run_dir_name)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or algorithm_training_type == "dagger":
        resume_path = resolve_checkpoint_path(
            task_name=args_cli.task,
            log_root_path=log_root_path,
            load_run=agent_cfg.load_run,
            load_checkpoint=agent_cfg.load_checkpoint,
            checkpoint=args_cli.checkpoint,
            algo_name=getattr(args_cli, "algo", None),
        )

    # # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = CRLRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # # create runner from rsl-rl
    runner = OnPolicyRunnerWithExtractor(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    # # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or algorithm_training_type == "dagger":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    if experiment_preset is not None:
        write_experiment_metadata(log_dir, experiment_preset, args=args_cli)

    # # # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
