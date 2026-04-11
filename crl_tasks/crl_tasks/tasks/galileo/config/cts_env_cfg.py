"""CTS environment configuration for Galileo CRL tasks."""

from __future__ import annotations

from isaaclab.utils import configclass

from crl_isaaclab.envs import CRLManagerBasedRLEnvCfg

from .defaults import GalileoDefaults, VIEWER_CFG
from .mdp_cfg import (
    ActionsCfg,
    CTSCostsCfg,
    CommandsCfg,
    CTSCurriculumCfg,
    CTSObservationsCfg,
    CTSRewardsCfg,
    EventCfg,
    TerminationsCfg,
)
from .scene_cfg import GalileoCTSSceneCfg


@configclass
class GalileoCTSCRLEnvCfg(CRLManagerBasedRLEnvCfg):
    scene: GalileoCTSSceneCfg = GalileoCTSSceneCfg(
        num_envs=GalileoDefaults.env.num_envs,
        env_spacing=1.0,
    )
    observations: CTSObservationsCfg = CTSObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CTSRewardsCfg = CTSRewardsCfg()
    costs: CTSCostsCfg = CTSCostsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CTSCurriculumCfg = CTSCurriculumCfg()
    crl_events = None
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = GalileoDefaults.general.decimation
        self.episode_length_s = GalileoDefaults.general.episode_length_s
        self.sim.dt = GalileoDefaults.sim.dt
        self.sim.render_interval = GalileoDefaults.general.render_interval
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.sim.physx.gpu_found_lost_pairs_capacity = 10 * 1024 * 1024
        self.scene.terrain.terrain_generator.curriculum = GalileoDefaults.terrain.curriculum
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8
        self.events.random_camera_position = None
        self.events.randomize_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.randomize_base_com.params["asset_cfg"].body_names = "base_link"
        self.events.push_robot_torque.params["asset_cfg"].body_names = "base_link"


@configclass
class GalileoCTSCRLEnvCfg_EVAL(GalileoCTSCRLEnvCfg):
    viewer = VIEWER_CFG

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = GalileoDefaults.env.eval_num_envs
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_base_com = None
        self.events.randomize_base_mass = None
        self.events.push_robot_vel.interval_range_s = (6.0, 6.0)


@configclass
class GalileoCTSCRLEnvCfg_PLAY(GalileoCTSCRLEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = GalileoDefaults.env.play_num_envs
        self.episode_length_s = 60.0
        self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.scene.terrain.terrain_generator.curriculum = False
        self.events.push_robot_vel = None
