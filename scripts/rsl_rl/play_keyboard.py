# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation viewer for Galileo checkpoints."""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
import warnings
import weakref

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

parser = argparse.ArgumentParser(description="Keyboard teleoperation for a Galileo checkpoint.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Ignored. Keyboard teleop always runs a single robot.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Galileo-CRL-Student-Play-v0",
    help="Galileo task to load. Override with the teacher play task when needed.",
)
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
    "--lin-speed",
    type=float,
    default=1.0,
    help="Forward/backward command magnitude used for W/S or Up/Down.",
)
parser.add_argument(
    "--lat-speed",
    type=float,
    default=0.5,
    help="Lateral command magnitude used for A/D.",
)
parser.add_argument(
    "--yaw-speed",
    type=float,
    default=0.5,
    help="Yaw-rate command magnitude used for Q/E or Left/Right.",
)
parser.add_argument(
    "--lin-accel",
    type=float,
    default=2.5,
    help="Linear x acceleration used by progressive keyboard control.",
)
parser.add_argument(
    "--lat-accel",
    type=float,
    default=2.0,
    help="Linear y acceleration used by progressive keyboard control.",
)
parser.add_argument(
    "--yaw-accel",
    type=float,
    default=2.5,
    help="Yaw acceleration used by progressive keyboard control.",
)
parser.add_argument(
    "--step-height",
    type=float,
    default=0.16,
    help="Height of the fixed stair tile used in teleop mode.",
)
parser.add_argument(
    "--stair-steps",
    type=int,
    default=8,
    help="Approximate stair riser count on each side of the teleop tile. Defaults to >5.",
)
parser.add_argument(
    "--debug-keys",
    action="store_true",
    default=False,
    help="Print raw keyboard events to help debug focus/key-name issues.",
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

args_cli.num_envs = 1
configure_safe_play_args(args_cli)
if args_cli.headless:
    raise SystemExit(
        "Keyboard teleop requires a GUI session. Re-run with a local display or pass --force_gui."
    )

print(
    "[INFO] Keyboard teleop launch selection: "
    f"force_gui={args_cli.force_gui}, "
    f"headless={args_cli.headless}, "
    f"DISPLAY={os.environ.get('DISPLAY', '')}, "
    f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')}"
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import isaaclab.terrains as terrain_gen
import torch
import omni.appwindow as omni_appwindow

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import crl_tasks.tasks.galileo  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import CRLRslRlVecEnvWrapper
from crl_tasks.tasks.galileo.config.agents.rsl_rl_cfg import CRLRslRlOnPolicyRunnerCfg


def _build_single_step_terrain_cfg(step_height: float, stair_steps: int) -> TerrainGeneratorCfg:
    """Create a fixed single stair tile for teleoperation."""
    fixed_height = float(max(step_height, 0.02))
    step_width = 0.35
    platform_width = 2.0
    border_width = 0.5
    stair_steps = max(int(stair_steps), 6)
    half_size = border_width + platform_width * 0.5 + stair_steps * step_width + step_width
    tile_size = max(10.0, 2.0 * half_size)
    return TerrainGeneratorCfg(
        size=(tile_size, tile_size),
        border_width=4.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(1.0, 1.0),
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(fixed_height, fixed_height),
                step_width=step_width,
                platform_width=platform_width,
                border_width=border_width,
                holes=False,
            ),
        },
    )


def _set_if_present(obj, name: str, value) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _configure_keyboard_play_env(env_cfg) -> None:
    """Shrink the scene to a single teleop robot on a simple stair tile."""
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.0
    env_cfg.episode_length_s = max(float(getattr(env_cfg, "episode_length_s", 60.0)), 300.0)

    terrain_cfg = _build_single_step_terrain_cfg(args_cli.step_height, args_cli.stair_steps)
    env_cfg.scene.terrain.terrain_generator = terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = 0
    print(
        "[INFO] Teleop terrain configured: stairs only, "
        f"approx_steps_per_side={max(int(args_cli.stair_steps), 6)}, "
        f"step_height={float(args_cli.step_height):.3f} m, "
        "single robot scene."
    )

    base_velocity = env_cfg.commands.base_velocity
    base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    base_velocity.rel_standing_envs = 0.0
    base_velocity.ranges.standing_command_prob = 0.0

    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        for name in (
            "random_camera_position",
            "randomize_base_com",
            "randomize_base_mass",
            "push_robot_vel",
            "push_robot_torque",
        ):
            _set_if_present(env_cfg.events, name, None)

    if hasattr(env_cfg, "viewer") and env_cfg.viewer is not None:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        env_cfg.viewer.eye = (-2.8, 2.4, 1.8)


class ManualCommandInjector:
    """Inject keyboard commands into the command term and visible observations."""

    def __init__(self, env):
        self.env = env
        self.command_term = env.unwrapped.command_manager.get_term("base_velocity")
        self.obs_manager = env.unwrapped.observation_manager
        self._group_layout = self._build_group_layout()

    def _build_group_layout(self) -> dict[str, list[dict[str, object]]]:
        layout: dict[str, list[dict[str, object]]] = {}
        active_terms = getattr(self.obs_manager, "active_terms", {})
        term_dims = getattr(self.obs_manager, "group_obs_term_dim", {})
        term_cfgs = getattr(self.obs_manager, "_group_obs_term_cfgs", {})

        for group_name, group_term_names in active_terms.items():
            group_layout: list[dict[str, object]] = []
            offset = 0
            for term_name, dims, term_cfg in zip(
                group_term_names,
                term_dims.get(group_name, ()),
                term_cfgs.get(group_name, ()),
            ):
                flat_dim = int(math.prod(dims))
                group_layout.append(
                    {
                        "name": term_name,
                        "slice": slice(offset, offset + flat_dim),
                        "flat_dim": flat_dim,
                        "cfg": term_cfg,
                    }
                )
                offset += flat_dim
            layout[group_name] = group_layout
        return layout

    @staticmethod
    def _scale_command(command: torch.Tensor, scale) -> torch.Tensor:
        if scale is None:
            return command
        scale_tensor = torch.as_tensor(scale, device=command.device, dtype=command.dtype)
        if scale_tensor.ndim == 0:
            return command * scale_tensor
        return command * scale_tensor.view(1, -1)

    def _patch_group_obs(self, group_name: str, obs_tensor: torch.Tensor, command: torch.Tensor) -> None:
        if not torch.is_tensor(obs_tensor):
            return

        for entry in self._group_layout.get(group_name, ()):
            term_cfg = entry["cfg"]
            params = getattr(term_cfg, "params", {}) or {}
            if params.get("command_name") != "base_velocity":
                continue

            term_slice = entry["slice"]
            flat_dim = int(entry["flat_dim"])
            term_func = getattr(term_cfg, "func", None)

            if hasattr(term_func, "_obs_history_buffer"):
                scale_cfg = dict(params.get("scales") or {})
                scaled_command = self._scale_command(command, scale_cfg.get("commands", 1.0))
                history_buffer = getattr(term_func, "_obs_history_buffer", None)
                if history_buffer is not None and history_buffer.shape[-1] >= scaled_command.shape[1]:
                    reset_mask = self.env.unwrapped.episode_length_buf <= 1
                    if torch.any(reset_mask):
                        history_buffer[reset_mask, :, -scaled_command.shape[1] :] = scaled_command[
                            reset_mask
                        ].unsqueeze(1)
                    history_buffer[:, -1, -scaled_command.shape[1] :] = scaled_command
                obs_tensor[:, term_slice.stop - scaled_command.shape[1] : term_slice.stop] = scaled_command
                continue

            if flat_dim == command.shape[1]:
                scaled_command = self._scale_command(command, getattr(term_cfg, "scale", None))
                obs_tensor[:, term_slice] = scaled_command

    def apply(self, command: torch.Tensor, obs: torch.Tensor | None = None, extras: dict | None = None) -> None:
        self.command_term.vel_command_b[:, :] = command
        self.command_term.is_standing_env[:] = False

        observations = extras.get("observations") if isinstance(extras, dict) else None
        if isinstance(observations, dict):
            for group_name, group_obs in observations.items():
                self._patch_group_obs(group_name, group_obs, command)
            if torch.is_tensor(obs) and "policy" in observations and obs.data_ptr() != observations["policy"].data_ptr():
                obs.copy_(observations["policy"])
        elif torch.is_tensor(obs):
            self._patch_group_obs("policy", obs, command)

        obs_buffer = getattr(self.obs_manager, "_obs_buffer", None)
        if isinstance(obs_buffer, dict):
            for group_name, group_obs in obs_buffer.items():
                if torch.is_tensor(group_obs):
                    self._patch_group_obs(group_name, group_obs, command)


class KeyboardCommandController:
    """Track pressed keys and convert them into a base-velocity command."""

    _KEY_ALIASES = {
        "UP_ARROW": "UP",
        "DOWN_ARROW": "DOWN",
        "LEFT_ARROW": "LEFT",
        "RIGHT_ARROW": "RIGHT",
        "KEY_W": "W",
        "KEY_A": "A",
        "KEY_S": "S",
        "KEY_D": "D",
        "KEY_Q": "Q",
        "KEY_E": "E",
        "KEY_R": "R",
        "KEY_ESCAPE": "ESCAPE",
        "KEY_SPACE": "SPACE",
    }
    _TERMINAL_ESCAPE_SEQUENCES = {
        "\x1b[A": "UP",
        "\x1b[B": "DOWN",
        "\x1b[C": "RIGHT",
        "\x1b[D": "LEFT",
    }

    def __init__(
        self,
        device: torch.device,
        lin_range: tuple[float, float],
        lat_range: tuple[float, float],
        yaw_range: tuple[float, float],
    ):
        self.device = device
        self._input = carb.input.acquire_input_interface()
        app_window = omni_appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("No Omniverse app window is available for keyboard teleoperation.")
        self._keyboard = app_window.get_keyboard()
        if self._keyboard is None:
            raise RuntimeError("The app window does not expose a keyboard handle.")
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._reset_requested = False
        self._quit_requested = False
        self._command = torch.zeros(1, 3, device=self.device)
        self._target_command = torch.zeros(1, 3, device=self.device)
        self._last_status_time = 0.0
        self._last_status_command: torch.Tensor | None = None
        self._last_input_label = "-"
        self._terminal_fd: int | None = None
        self._terminal_old_attrs = None
        self._terminal_buffer = ""

        self.forward_speed = min(abs(float(args_cli.lin_speed)), max(float(lin_range[1]), 0.0))
        self.backward_speed = min(abs(float(args_cli.lin_speed)), max(-float(lin_range[0]), 0.0))
        self.lateral_speed = min(
            abs(float(args_cli.lat_speed)),
            max(abs(float(lat_range[0])), abs(float(lat_range[1]))),
        )
        self.yaw_speed = min(
            abs(float(args_cli.yaw_speed)),
            max(abs(float(yaw_range[0])), abs(float(yaw_range[1]))),
        )
        self._accel = torch.tensor(
            [[
                max(float(args_cli.lin_accel), 1.0e-4),
                max(float(args_cli.lat_accel), 1.0e-4),
                max(float(args_cli.yaw_accel), 1.0e-4),
            ]],
            device=self.device,
        )
        self._command_step = torch.tensor(
            [[
                max(self.forward_speed / 10.0, 0.05),
                max(self.lateral_speed / 10.0, 0.05),
                max(self.yaw_speed / 10.0, 0.05),
            ]],
            device=self.device,
        )

        self._try_enable_terminal_fallback()

        print("[INFO] Keyboard teleop controls:")
        print("  W/S or Up/Down    : increase/decrease forward command")
        print("  A/D               : increase/decrease lateral command")
        print("  Q/E or Left/Right : increase/decrease yaw command")
        print("  Space             : clear target/current command")
        print("  R                 : reset robot")
        print("  Esc               : quit (GUI input)")
        if self._terminal_fd is not None:
            print("  X                 : quit (terminal fallback)")

    def __del__(self):
        if (
            hasattr(self, "_input")
            and self._input is not None
            and hasattr(self, "_keyboard")
            and self._keyboard is not None
            and hasattr(self, "_keyboard_sub")
            and self._keyboard_sub is not None
        ):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        self._restore_terminal()

    @staticmethod
    def _motion_keys() -> tuple[str, ...]:
        return ("W", "S", "UP", "DOWN", "A", "D", "Q", "E", "LEFT", "RIGHT")

    @classmethod
    def _normalize_key_name(cls, key_name: str) -> str:
        normalized = str(key_name).upper()
        normalized = cls._KEY_ALIASES.get(normalized, normalized)
        if normalized.startswith("KEY_"):
            normalized = normalized[4:]
        return normalized

    def _try_enable_terminal_fallback(self) -> None:
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; terminal keyboard fallback is unavailable.")
            return
        try:
            self._terminal_fd = sys.stdin.fileno()
            self._terminal_old_attrs = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            print("[INFO] Terminal keyboard fallback enabled. You can press keys in the launch terminal.")
        except Exception as exc:
            self._terminal_fd = None
            self._terminal_old_attrs = None
            print(f"[WARN] Failed to enable terminal keyboard fallback: {exc}")

    def _restore_terminal(self) -> None:
        if self._terminal_fd is not None and self._terminal_old_attrs is not None:
            try:
                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_old_attrs)
            except Exception:
                pass
            self._terminal_old_attrs = None

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    def _set_last_input(self, label: str) -> None:
        self._last_input_label = label

    @staticmethod
    def _move_towards(current: torch.Tensor, target: torch.Tensor, max_delta: torch.Tensor) -> torch.Tensor:
        return current + torch.clamp(target - current, min=-max_delta, max=max_delta)

    def _apply_incremental_keypress(self, key_name: str, *, source: str) -> None:
        self._set_last_input(f"{source}:{key_name}")
        target = self._target_command.clone()

        if key_name in ("W", "UP"):
            target[:, 0] = self._clamp(
                float(target[:, 0].item() + self._command_step[:, 0].item()),
                -self.backward_speed,
                self.forward_speed,
            )
        elif key_name in ("S", "DOWN"):
            target[:, 0] = self._clamp(
                float(target[:, 0].item() - self._command_step[:, 0].item()),
                -self.backward_speed,
                self.forward_speed,
            )
        elif key_name == "A":
            target[:, 1] = self._clamp(
                float(target[:, 1].item() + self._command_step[:, 1].item()),
                -self.lateral_speed,
                self.lateral_speed,
            )
        elif key_name == "D":
            target[:, 1] = self._clamp(
                float(target[:, 1].item() - self._command_step[:, 1].item()),
                -self.lateral_speed,
                self.lateral_speed,
            )
        elif key_name in ("Q", "LEFT"):
            target[:, 2] = self._clamp(
                float(target[:, 2].item() + self._command_step[:, 2].item()),
                -self.yaw_speed,
                self.yaw_speed,
            )
        elif key_name in ("E", "RIGHT"):
            target[:, 2] = self._clamp(
                float(target[:, 2].item() - self._command_step[:, 2].item()),
                -self.yaw_speed,
                self.yaw_speed,
            )
        elif key_name == "SPACE":
            self._command.zero_()
            target.zero_()
        elif key_name == "R":
            self._reset_requested = True
        elif key_name in ("ESCAPE", "X"):
            self._quit_requested = True

        self._target_command.copy_(target)

    def _poll_terminal_input(self) -> None:
        if self._terminal_fd is None:
            return
        while True:
            readable, _, _ = select.select([self._terminal_fd], [], [], 0.0)
            if not readable:
                break
            chunk = os.read(self._terminal_fd, 32)
            if not chunk:
                break
            self._terminal_buffer += chunk.decode(errors="ignore")

        while self._terminal_buffer:
            if self._terminal_buffer.startswith("\x1b[") and len(self._terminal_buffer) >= 3:
                seq = self._terminal_buffer[:3]
                key_name = self._TERMINAL_ESCAPE_SEQUENCES.get(seq)
                if key_name is not None:
                    self._terminal_buffer = self._terminal_buffer[3:]
                    self._apply_incremental_keypress(key_name, source="tty")
                    if args_cli.debug_keys:
                        print(f"\n[KEY] source=tty raw={seq!r} normalized={key_name}")
                    continue
            char = self._terminal_buffer[0]
            self._terminal_buffer = self._terminal_buffer[1:]
            if char in ("\n", "\r", "\t"):
                continue
            normalized = self._normalize_key_name("SPACE" if char == " " else char)
            if normalized == "\x1b":
                continue
            if normalized in self._motion_keys() or normalized in ("SPACE", "R", "X"):
                self._apply_incremental_keypress(normalized, source="tty")
                if args_cli.debug_keys:
                    print(f"\n[KEY] source=tty raw={char!r} normalized={normalized}")

    def _print_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        cmd = self._command.detach().cpu().view(-1)
        target = self._target_command.detach().cpu().view(-1)
        if (
            not force
            and self._last_status_command is not None
            and now - self._last_status_time < 0.2
            and torch.max(torch.abs(cmd - self._last_status_command)).item() < 0.02
        ):
            return

        sys.stdout.write(
            "\r[CMD] "
            f"last_input={self._last_input_label:<20} "
            f"target(vx={target[0]:+.2f}, vy={target[1]:+.2f}, wz={target[2]:+.2f}) "
            f"current(vx={cmd[0]:+.2f}, vy={cmd[1]:+.2f}, wz={cmd[2]:+.2f})      "
        )
        sys.stdout.flush()
        self._last_status_time = now
        self._last_status_command = cmd.clone()

    def _on_keyboard_event(self, event, *args) -> None:
        raw_key_name = event.input.name
        key_name = self._normalize_key_name(raw_key_name)
        if args_cli.debug_keys:
            print(
                "\n[KEY] "
                f"type={event.type} raw={raw_key_name} normalized={key_name}"
            )
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            if key_name in self._motion_keys() or key_name in ("SPACE", "R", "ESCAPE"):
                self._apply_incremental_keypress(key_name, source="gui")
                self._print_status(force=True)

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def consume_reset_request(self) -> bool:
        if not self._reset_requested:
            return False
        self._reset_requested = False
        return True

    def reset_state(self) -> None:
        self._command.zero_()
        self._target_command.zero_()
        self._last_input_label = "reset"

    def step(self, dt: float, *, force_print: bool = False) -> torch.Tensor:
        self._poll_terminal_input()
        accel = torch.where(self._target_command == 0.0, self._accel * 2.0, self._accel)
        max_delta = accel * max(float(dt), 0.0)
        self._command = self._move_towards(self._command, self._target_command, max_delta)
        self._print_status(force=force_print)
        return self._command.clone()


def main() -> None:
    if "Galileo" not in args_cli.task:
        raise SystemExit("This keyboard teleop helper currently supports Galileo tasks only.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: CRLRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    experiment_preset = load_experiment_preset(selection=args_cli.exp, file_path=args_cli.exp_file)
    if experiment_preset is not None:
        apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=experiment_preset)
        agent_cfg = cli_args.reapply_rsl_rl_cli_overrides(agent_cfg, args_cli)
        print(
            f"[INFO] Applied experiment preset: {experiment_preset.name} ({experiment_preset.path})"
        )

    _configure_keyboard_play_env(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

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
        raise RuntimeError(
            "A pretrained checkpoint is unavailable for this task and no explicit checkpoint was provided."
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = CRLRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunnerWithExtractor(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    command_cfg = env_cfg.commands.base_velocity.ranges
    keyboard = KeyboardCommandController(
        device=env.unwrapped.device,
        lin_range=tuple(command_cfg.lin_vel_x),
        lat_range=tuple(command_cfg.lin_vel_y),
        yaw_range=tuple(command_cfg.ang_vel_z),
    )
    injector = ManualCommandInjector(env)

    dt = env.unwrapped.step_dt
    obs, extras = env.reset()
    keyboard.reset_state()
    injector.apply(keyboard.step(0.0, force_print=True), obs=obs, extras=extras)

    while simulation_app.is_running() and not keyboard.quit_requested:
        if keyboard.consume_reset_request():
            obs, extras = env.reset()
            keyboard.reset_state()
            injector.apply(keyboard.step(0.0, force_print=True), obs=obs, extras=extras)
            continue

        start_time = time.time()
        injector.apply(keyboard.step(dt), obs=obs, extras=extras)
        with torch.inference_mode():
            actions = policy(obs, hist_encoding=True)
        obs, _, dones, extras = env.step(actions)
        if torch.any(dones):
            print("\n[INFO] Episode finished. Resetting command ramp for the next rollout.")
            keyboard.reset_state()
            injector.apply(keyboard.step(0.0, force_print=True), obs=obs, extras=extras)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
