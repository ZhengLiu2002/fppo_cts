"""CTS environment configuration for Galileo CRL tasks."""

from __future__ import annotations

import copy

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
    _student_height_scan_obs_term,
)
from .scene_cfg import GalileoCTSSceneCfg
from .terrain_profiles import restrict_terrain_generator_to_named_subterrains


def _set_optional_block_field(block, key: str, value) -> None:
    """Update either a config object or a plain dict restored from preset JSON."""
    if block is None:
        return
    if isinstance(block, dict):
        block[key] = value
    else:
        setattr(block, key, value)


def _config_field(block, key: str):
    if block is None:
        return None
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _config_params(block) -> dict | None:
    params = _config_field(block, "params")
    return params if isinstance(params, dict) else None


def _config_param(block, key: str):
    params = _config_params(block)
    return None if params is None else params.get(key)


def _set_config_param(block, key: str, value) -> None:
    params = _config_params(block)
    if params is not None:
        params[key] = copy.deepcopy(value)


_PRIVILEGED_OBS_NORMALIZATION_BINDINGS = (
    (
        "randomize_base_mass",
        "mass_distribution_params",
        "mass_delta_range",
        (("policy", "teacher_base_mass"), ("critic", "base_mass")),
        "add",
    ),
    (
        "randomize_base_com",
        "com_range",
        "com_range",
        (("policy", "teacher_base_com"), ("critic", "base_com")),
        None,
    ),
    (
        "physics_material",
        "friction_range",
        "friction_range",
        (("policy", "teacher_ground_friction"), ("critic", "ground_friction")),
        None,
    ),
)


def _set_observation_term_param(observations, group_name: str, term_name: str, key: str, value) -> None:
    group = _config_field(observations, group_name)
    term = _config_field(group, term_name)
    _set_config_param(term, key, value)


@configclass
class GalileoCTSCRLEnvCfg(CRLManagerBasedRLEnvCfg):
    terrain_profile: str = "rough"
    terrain_debug_single_family: str | None = None
    enable_student_height_scan: bool = False
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
        self.config_summary = GalileoDefaults
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
        self.apply_experiment_overrides()

    def _configure_student_observations(self) -> None:
        """Enable optional student terrain scan only for dedicated probes."""
        policy_obs = getattr(self.observations, "policy", None)
        if policy_obs is None or not hasattr(policy_obs, "student_height_scan"):
            return
        if self.enable_student_height_scan:
            policy_obs.student_height_scan = _student_height_scan_obs_term()
        else:
            policy_obs.student_height_scan = None

    def _restrict_to_single_terrain_family(self, terrain_name: str) -> None:
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_name not in terrain_generator.sub_terrains:
            available = ", ".join(sorted(terrain_generator.sub_terrains.keys()))
            raise ValueError(
                f"Unknown terrain_debug_single_family={terrain_name!r}. Available: {available}"
            )
        restrict_terrain_generator_to_named_subterrains(terrain_generator, (terrain_name,))

    def _sync_privileged_observation_normalization(self) -> None:
        """Keep privileged observation normalization tied to active DR targets."""
        for (
            event_name,
            event_param,
            obs_param,
            obs_terms,
            required_operation,
        ) in _PRIVILEGED_OBS_NORMALIZATION_BINDINGS:
            event_term = _config_field(self.events, event_name)
            value = _config_param(event_term, event_param)
            if value is None:
                continue
            operation = _config_param(event_term, "operation")
            if (
                required_operation is not None
                and operation is not None
                and str(operation).lower() != required_operation
            ):
                continue
            for group_name, term_name in obs_terms:
                _set_observation_term_param(
                    self.observations,
                    group_name,
                    term_name,
                    obs_param,
                    value,
                )

    def apply_experiment_overrides(self) -> None:
        """Apply preset-driven profile switches after JSON overrides land."""
        if self.terrain_profile == "flat":
            terrain_generator = self.scene.terrain.terrain_generator
            restrict_terrain_generator_to_named_subterrains(
                terrain_generator,
                getattr(GalileoDefaults.terrain, "flat_subterrain_names", ("plane_run",)),
                proportions=getattr(GalileoDefaults.terrain, "flat_subterrain_proportions", None),
            )
            terrain_generator.curriculum = False
            terrain_generator.difficulty_range = (0.0, 0.0)
            self.scene.terrain.max_init_terrain_level = 0
        elif self.terrain_profile == "rough":
            if self.terrain_debug_single_family:
                self._restrict_to_single_terrain_family(self.terrain_debug_single_family)
        else:
            raise ValueError(f"Unknown Galileo terrain_profile: {self.terrain_profile!r}")
        self._configure_student_observations()
        self._sync_privileged_observation_normalization()


@configclass
class GalileoCTSCRLEnvCfg_EVAL(GalileoCTSCRLEnvCfg):
    viewer = VIEWER_CFG

    def apply_eval_runtime_overrides(self) -> None:
        """Re-apply eval-safe overrides after preset merges."""
        self._sync_privileged_observation_normalization()
        self.scene.num_envs = GalileoDefaults.env.eval_num_envs
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_base_com = None
        self.events.randomize_base_mass = None
        _set_optional_block_field(self.events.push_robot_vel, "interval_range_s", (6.0, 6.0))

    def __post_init__(self):
        super().__post_init__()
        self.apply_eval_runtime_overrides()


@configclass
class GalileoCTSCRLEnvCfg_PLAY(GalileoCTSCRLEnvCfg_EVAL):
    def apply_play_runtime_overrides(self) -> None:
        """Re-apply play-safe overrides after preset merges."""
        self.apply_eval_runtime_overrides()
        self.scene.num_envs = GalileoDefaults.env.play_num_envs
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
            self.scene.terrain.terrain_generator.curriculum = False
        self.events.push_robot_vel = None

    def __post_init__(self):
        super().__post_init__()
        self.apply_play_runtime_overrides()
