from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from crl_isaaclab.terrains.runtime import resolve_env_terrain_names

if TYPE_CHECKING:
    from crl_isaaclab.envs import CRLManagerBasedEnv
    from .crl_command_cfg import CRLCommandCfg


class UniformCRLCommand(CommandTerm):
    cfg: CRLCommandCfg

    def __init__(self, cfg: CRLCommandCfg, env: CRLManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # Import GalileoDefaults to access terrain-specific command ranges.
        try:
            from crl_tasks.tasks.galileo.config.defaults import GalileoDefaults

            self.terrain_ranges = GalileoDefaults.command.ranges
        except ImportError:
            omni.log.warn("Cannot import GalileoDefaults, falling back to default ranges")
            self.terrain_ranges = None

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time: {self.cfg.resampling_time_range}\n"
        msg += f"\tSmall command to zero: {self.cfg.small_commands_to_zero}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

    def _resolve_env_id_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Return environment ids as a 1-D tensor on this command's device."""
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long).flatten()

    def _get_env_terrain_names(self, env_id_tensor: torch.Tensor) -> list[str | None]:
        """Resolve terrain names for each environment id in the batch."""
        if env_id_tensor.numel() == 0 or not hasattr(self._env.scene, "terrain"):
            return [None] * int(env_id_tensor.numel())

        env_id_list = env_id_tensor.detach().cpu().tolist()
        if not hasattr(self._env.scene, "terrain"):
            return [None] * len(env_id_list)

        terrain = self._env.scene.terrain
        resolved_names: list[str | None] | None = None

        # 尝试从 crl_event 获取逐环境地形名称
        if hasattr(self._env, "crl_manager") and self._env.crl_manager is not None:
            try:
                crl_event = self._env.crl_manager._terms.get("base_crl")
                if crl_event is not None and hasattr(crl_event, "env_per_terrain_name"):
                    terrain_names = crl_event.env_per_terrain_name
                    resolved_names = []
                    for env_id in env_id_list:
                        if 0 <= env_id < len(terrain_names):
                            resolved_names.append(str(terrain_names[env_id]))
                        else:
                            resolved_names.append(None)
            except Exception:
                resolved_names = None

        if resolved_names is not None:
            return resolved_names

        # 如果无法从 crl_event 获取，尝试直接从 terrain 获取
        try:
            env_names = resolve_env_terrain_names(terrain)
            if env_names is not None:
                resolved_names = []
                for env_id in env_id_list:
                    if 0 <= env_id < len(env_names):
                        resolved_names.append(str(env_names[env_id]))
                    else:
                        resolved_names.append(None)
                return resolved_names
        except Exception:
            pass

        return [None] * len(env_id_list)

    def _get_terrain_specific_ranges(self, terrain_name: str | None):
        """根据地形名称获取对应的指令范围配置。

        如果找不到对应的配置，会记录警告并使用 None，强制使用 cfg.ranges 中的默认值。
        这样可以确保每种地形都必须有明确的配置，避免静默使用错误的配置。
        """
        if terrain_name is None or self.terrain_ranges is None:
            return None

        # 如果地形名称在配置中存在，返回对应的配置
        if terrain_name in self.terrain_ranges:
            return self.terrain_ranges[terrain_name]

        # 如果找不到配置，记录警告（但不在每次调用时都打印，避免日志过多）
        if not hasattr(self, "_terrain_warning_printed"):
            self._terrain_warning_printed = set()

        if terrain_name not in self._terrain_warning_printed:
            omni.log.warn(
                f"[UniformCRLCommand] 未找到地形 '{terrain_name}' 的指令配置，"
                f"将使用 cfg.ranges 中的默认配置。请确保在 GalileoDefaults.command.ranges 中为所有地形添加配置。"
            )
            self._terrain_warning_printed.add(terrain_name)

        # 返回 None，强制使用 cfg.ranges 中的默认值
        return None

    def _resample_command_group(self, env_id_tensor: torch.Tensor, terrain_name: str | None):
        # 获取地形名称并选择对应的指令配置
        terrain_ranges = self._get_terrain_specific_ranges(terrain_name)
        # Terrain-specific curriculum endpoints define the x/yaw command ranges.
        use_lin_x_cfg_range = (
            self.cfg.ranges.start_curriculum_lin_x is not None
            and self.cfg.ranges.max_curriculum_lin_x is not None
        )
        use_ang_z_cfg_range = (
            self.cfg.ranges.start_curriculum_ang_z is not None
            and self.cfg.ranges.max_curriculum_ang_z is not None
        )

        # 如果找到地形特定的配置，使用它；否则使用默认配置
        if terrain_ranges is not None:
            # 使用地形特定的配置
            lin_vel_x_base = (
                self.cfg.ranges.lin_vel_x if use_lin_x_cfg_range else terrain_ranges["lin_vel_x"]
            )
            lin_vel_y_base = terrain_ranges["lin_vel_y"]
            ang_vel_z_base = (
                self.cfg.ranges.ang_vel_z if use_ang_z_cfg_range else terrain_ranges["ang_vel_z"]
            )
            standing_command_prob = terrain_ranges.get(
                "standing_command_prob", self.cfg.ranges.standing_command_prob
            )
            start_curriculum_lin_x = terrain_ranges.get(
                "start_curriculum_lin_x", self.cfg.ranges.start_curriculum_lin_x
            )
            start_curriculum_ang_z = terrain_ranges.get(
                "start_curriculum_ang_z", self.cfg.ranges.start_curriculum_ang_z
            )
            max_curriculum_lin_x = terrain_ranges.get(
                "max_curriculum_lin_x", self.cfg.ranges.max_curriculum_lin_x
            )
            max_curriculum_ang_z = terrain_ranges.get(
                "max_curriculum_ang_z", self.cfg.ranges.max_curriculum_ang_z
            )
        else:
            # 使用默认配置
            lin_vel_x_base = self.cfg.ranges.lin_vel_x
            lin_vel_y_base = self.cfg.ranges.lin_vel_y
            ang_vel_z_base = self.cfg.ranges.ang_vel_z
            standing_command_prob = self.cfg.ranges.standing_command_prob
            start_curriculum_lin_x = self.cfg.ranges.start_curriculum_lin_x
            start_curriculum_ang_z = self.cfg.ranges.start_curriculum_ang_z
            max_curriculum_lin_x = self.cfg.ranges.max_curriculum_lin_x
            max_curriculum_ang_z = self.cfg.ranges.max_curriculum_ang_z

        # sample velocity commands
        r = torch.empty(len(env_id_tensor), device=self.device)
        lin_x_range = lin_vel_x_base
        ang_z_range = ang_vel_z_base
        lin_x_level = float(getattr(self.cfg, "lin_x_level", 0.0))
        max_lin_x_level = max(float(getattr(self.cfg, "max_lin_x_level", 1.0)), 1.0e-6)
        ang_z_level = float(getattr(self.cfg, "ang_z_level", 0.0))
        max_ang_z_level = max(float(getattr(self.cfg, "max_ang_z_level", 1.0)), 1.0e-6)

        def _interp_range(
            start_range: tuple[float, float], end_range: tuple[float, float], level: float, max_level: float
        ) -> tuple[float, float]:
            progress = min(max(level / max_level, 0.0), 1.0)
            return (
                float(start_range[0] + (end_range[0] - start_range[0]) * progress),
                float(start_range[1] + (end_range[1] - start_range[1]) * progress),
            )

        if use_lin_x_cfg_range and start_curriculum_lin_x is not None and max_curriculum_lin_x is not None:
            lin_x_range = _interp_range(
                start_curriculum_lin_x,
                max_curriculum_lin_x,
                lin_x_level,
                max_lin_x_level,
            )

        if use_ang_z_cfg_range and start_curriculum_ang_z is not None and max_curriculum_ang_z is not None:
            ang_z_range = _interp_range(
                start_curriculum_ang_z,
                max_curriculum_ang_z,
                ang_z_level,
                max_ang_z_level,
            )

        progress = None
        # curriculum-aware range scaling (optional)
        if (
            self.cfg.terrain_level_range_scaling
            and self.cfg.ranges.start_curriculum_lin_x is not None
            and self.cfg.ranges.max_curriculum_lin_x is not None
            and hasattr(self._env.scene, "terrain")
            and getattr(self._env.scene.terrain, "terrain_levels", None) is not None
        ):
            levels = self._env.scene.terrain.terrain_levels[env_id_tensor].float()
            terrain_gen = getattr(self._env.scene.terrain, "terrain_generator_class", None)
            num_rows = float(getattr(terrain_gen, "num_rows", 1))
            denom = max(num_rows - 1.0, 1.0)
            progress = torch.clamp(levels / denom, 0.0, 1.0)
            start_min, start_max = (
                start_curriculum_lin_x if start_curriculum_lin_x is not None else lin_vel_x_base
            )
            max_min, max_max = (
                max_curriculum_lin_x if max_curriculum_lin_x is not None else lin_vel_x_base
            )
            lin_x_range = (
                float(start_min + (max_min - start_min) * progress.mean().item()),
                float(start_max + (max_max - start_max) * progress.mean().item()),
            )
        if (
            progress is not None
            and start_curriculum_ang_z is not None
            and max_curriculum_ang_z is not None
        ):
            start_min, start_max = start_curriculum_ang_z
            max_min, max_max = max_curriculum_ang_z
            ang_z_range = (
                float(start_min + (max_min - start_min) * progress.mean().item()),
                float(start_max + (max_max - start_max) * progress.mean().item()),
            )
        # -- linear velocity - x direction
        self.vel_command_b[env_id_tensor, 0] = r.uniform_(*lin_x_range)
        # -- linear velocity - y direction
        self.vel_command_b[env_id_tensor, 1] = r.uniform_(*lin_vel_y_base)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_id_tensor, 2] = r.uniform_(*ang_z_range)
        # update standing envs
        # 使用地形特定的 standing_command_prob，如果存在；否则使用配置中的值
        standing_prob = (
            standing_command_prob if terrain_ranges is not None else self.cfg.rel_standing_envs
        )
        self.is_standing_env[env_id_tensor] = r.uniform_(0.0, 1.0) <= standing_prob

        # update standing envs
        if self.cfg.small_commands_to_zero:
            xy_norm = torch.norm(self.vel_command_b[env_id_tensor, :2], dim=1, keepdim=True)
            self.vel_command_b[env_id_tensor, :2] *= (xy_norm > self.cfg.clips.lin_vel_clip)

        # enforce minimum absolute command magnitudes (optional)
        min_abs_x = self.cfg.min_abs_lin_vel_x
        if min_abs_x is not None and min_abs_x > 0.0:
            x_vals = self.vel_command_b[env_id_tensor, 0]
            x_abs = torch.abs(x_vals)
            mask = x_abs < min_abs_x
            if mask.any():
                signs = torch.sign(x_vals[mask])
                zeros = signs == 0
                if zeros.any():
                    signs[zeros] = torch.sign(torch.rand_like(signs[zeros]) - 0.5)
                self.vel_command_b[env_id_tensor[mask], 0] = signs * min_abs_x

        min_abs_y = self.cfg.min_abs_lin_vel_y
        if min_abs_y is not None and min_abs_y > 0.0:
            y_vals = self.vel_command_b[env_id_tensor, 1]
            y_abs = torch.abs(y_vals)
            mask = y_abs < min_abs_y
            if mask.any():
                signs = torch.sign(y_vals[mask])
                zeros = signs == 0
                if zeros.any():
                    signs[zeros] = torch.sign(torch.rand_like(signs[zeros]) - 0.5)
                self.vel_command_b[env_id_tensor[mask], 1] = signs * min_abs_y

    def _resample_command(self, env_ids: Sequence[int]):
        env_id_tensor = self._resolve_env_id_tensor(env_ids)
        if env_id_tensor.numel() == 0:
            return

        terrain_names = self._get_env_terrain_names(env_id_tensor)
        terrain_groups: dict[str | None, list[int]] = {}
        for local_idx, terrain_name in enumerate(terrain_names):
            terrain_groups.setdefault(terrain_name, []).append(local_idx)

        for terrain_name, local_indices in terrain_groups.items():
            group_env_ids = env_id_tensor[torch.as_tensor(local_indices, device=self.device)]
            self._resample_command_group(group_env_ids, terrain_name)

    def _update_command(self):
        # Optionally zero very small yaw-rate commands, matching lin_vel_clip behavior.
        if self.cfg.small_commands_to_zero:
            self.vel_command_b[:, 2] *= (
                torch.abs(self.vel_command_b[:, 2]) > self.cfg.clips.ang_vel_clip
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if standing_env_ids.numel() > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0
