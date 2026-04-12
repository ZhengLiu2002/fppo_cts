"""CTS scene configuration for Galileo blind omni-directional locomotion."""

from __future__ import annotations

import copy

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from .defaults import (
    GALILEO_ROUGH_TERRAIN_CFG,
    GalileoBaseSceneCfg,
    GalileoDefaults,
    build_galileo_robot_cfg,
)


@configclass
class GalileoCTSSceneCfg(GalileoBaseSceneCfg):
    """Scene shared by all CTS train/eval/play variants."""

    robot: ArticulationCfg = build_galileo_robot_cfg()
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
        track_air_time=True,
        debug_vis=False,
        force_threshold=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot = build_galileo_robot_cfg()
        self.terrain.terrain_generator = copy.deepcopy(GALILEO_ROUGH_TERRAIN_CFG)
        if getattr(GalileoDefaults.terrain, "flat_only_pretrain", False):
            flat_name = getattr(GalileoDefaults.terrain, "flat_subterrain_name", "crl_flat")
            flat_cfg = copy.deepcopy(self.terrain.terrain_generator.sub_terrains[flat_name])
            flat_cfg.proportion = 1.0
            self.terrain.terrain_generator.sub_terrains = {flat_name: flat_cfg}
        self.terrain.terrain_generator.size = GalileoDefaults.terrain.size
        self.terrain.terrain_generator.border_width = GalileoDefaults.terrain.border_width
        self.terrain.terrain_generator.num_rows = GalileoDefaults.terrain.num_rows
        self.terrain.terrain_generator.num_cols = GalileoDefaults.terrain.num_cols
        self.terrain.terrain_generator.horizontal_scale = GalileoDefaults.terrain.horizontal_scale
        self.terrain.terrain_generator.vertical_scale = GalileoDefaults.terrain.vertical_scale
        self.terrain.terrain_generator.slope_threshold = GalileoDefaults.terrain.slope_threshold
        self.terrain.terrain_generator.curriculum = GalileoDefaults.terrain.curriculum
        self.terrain.terrain_generator.difficulty_range = GalileoDefaults.terrain.difficulty_range


__all__ = ["GalileoCTSSceneCfg"]
