from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
from .uniform_crl_command import UniformCRLCommand


@configclass
class CRLCommandCfg(CommandTermCfg):
    class_type: type = UniformCRLCommand
    asset_name: str = MISSING
    small_commands_to_zero: bool = True
    rel_standing_envs: float = 0.0
    # If enabled, command ranges are additionally scaled by terrain levels in
    # UniformCRLCommand._resample_command. Keep disabled when using curriculum
    # terms as the single source of command difficulty.
    terrain_level_range_scaling: bool = False
    # curriculum levels for progressive command ranges
    lin_x_level: float = 0.0
    max_lin_x_level: float = 1.0
    lin_x_level_step: float = 0.1
    ang_z_level: float = 0.0
    max_ang_z_level: float = 1.0
    ang_z_level_step: float = 0.1
    # minimum absolute command magnitudes (optional)
    min_abs_lin_vel_x: float | None = None
    min_abs_lin_vel_y: float | None = None

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = (-0.35, 0.35)
        ang_vel_z: tuple[float, float] = (-0.2, 0.2)
        standing_command_prob: float = 0.0
        start_curriculum_lin_x: tuple[float, float] | None = None
        start_curriculum_ang_z: tuple[float, float] | None = None
        max_curriculum_lin_x: tuple[float, float] | None = None
        max_curriculum_ang_z: tuple[float, float] | None = None

    @configclass
    class Clips:
        lin_vel_clip: float = MISSING
        ang_vel_clip: float = MISSING

    ranges: Ranges = MISSING
    clips: Clips = MISSING
