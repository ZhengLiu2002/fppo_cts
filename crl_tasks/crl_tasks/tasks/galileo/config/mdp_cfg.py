"""MDP term configuration for Galileo CRL tasks."""

from __future__ import annotations

from importlib import import_module

mdp = import_module("crl_isaaclab.envs.mdp")
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    RewardTermCfg as CostTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from .defaults import GalileoDefaults


LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)

FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
LEFT_FOOT_BODY_NAMES = ["FL_foot", "RL_foot"]
RIGHT_FOOT_BODY_NAMES = ["FR_foot", "RR_foot"]
BASE_BODY_NAMES = ["base_link"]

# Joint-position feasibility cost uses the articulation soft limits directly.


def _student_height_scan_obs_term() -> ObsTerm:
    """Optional student-facing terrain scan used by diagnostic probes only."""
    return ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
    )


@configclass
class CommandsCfg:
    """全向底盘速度指令。

    - 训练时按地形采样不同的 `(vx, vy, wz)` 指令范围。
    - x/yaw 方向使用 terrain-specific curriculum 扩展范围。
    """

    base_velocity = mdp.CRLCommandCfg(
        asset_name="robot",
        resampling_time_range=GalileoDefaults.command.resampling_time_range,
        lin_x_level=GalileoDefaults.command.lin_x_level,
        max_lin_x_level=GalileoDefaults.command.max_lin_x_level,
        lin_x_level_step=GalileoDefaults.command.lin_x_level_step,
        lin_y_level=GalileoDefaults.command.lin_y_level,
        max_lin_y_level=GalileoDefaults.command.max_lin_y_level,
        lin_y_level_step=GalileoDefaults.command.lin_y_level_step,
        ang_z_level=GalileoDefaults.command.ang_z_level,
        max_ang_z_level=GalileoDefaults.command.max_ang_z_level,
        ang_z_level_step=GalileoDefaults.command.ang_z_level_step,
        heading_control_stiffness=GalileoDefaults.command.heading_control_stiffness,
        velocity_x_forward_scale=GalileoDefaults.command.velocity_x_forward_scale,
        velocity_x_backward_scale=GalileoDefaults.command.velocity_x_backward_scale,
        velocity_y_scale=GalileoDefaults.command.velocity_y_scale,
        velocity_yaw_scale=GalileoDefaults.command.velocity_yaw_scale,
        max_velocity=GalileoDefaults.command.max_velocity,
        min_abs_lin_vel_x=GalileoDefaults.command.min_abs_lin_vel_x,
        min_abs_lin_vel_y=GalileoDefaults.command.min_abs_lin_vel_y,
        rel_standing_envs=GalileoDefaults.command.default.standing_command_prob,
        terrain_level_range_scaling=False,
        ranges=mdp.CRLCommandCfg.Ranges(
            lin_vel_x=GalileoDefaults.command.default.lin_vel_x,
            lin_vel_y=GalileoDefaults.command.default.lin_vel_y,
            ang_vel_z=GalileoDefaults.command.default.ang_vel_z,
            heading=GalileoDefaults.command.default.heading,
            heading_command_prob=GalileoDefaults.command.default.heading_command_prob,
            yaw_command_prob=GalileoDefaults.command.default.yaw_command_prob,
            standing_command_prob=GalileoDefaults.command.default.standing_command_prob,
            start_curriculum_lin_x=GalileoDefaults.command.default.start_curriculum_lin_x,
            start_curriculum_lin_y=GalileoDefaults.command.default.start_curriculum_lin_y,
            start_curriculum_ang_z=GalileoDefaults.command.default.start_curriculum_ang_z,
            max_curriculum_lin_x=GalileoDefaults.command.default.max_curriculum_lin_x,
            max_curriculum_lin_y=GalileoDefaults.command.default.max_curriculum_lin_y,
            max_curriculum_ang_z=GalileoDefaults.command.default.max_curriculum_ang_z,
        ),
        clips=mdp.CRLCommandCfg.Clips(
            lin_vel_clip=GalileoDefaults.command.clips["lin_vel_clip"],
            ang_vel_clip=GalileoDefaults.command.clips["ang_vel_clip"],
        ),
    )


@configclass
class _PrivilegedObsGroupCfg(ObsGroup):
    """Shared privileged observation terms for teacher and critic views."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
    )
    base_com = ObsTerm(
        func=mdp.base_com,
        params={
            "body_name": "base_link",
            "normalize": True,
            "com_range": GalileoDefaults.priv_obs_norm.base_com_range,
        },
    )
    base_mass = ObsTerm(
        func=mdp.base_mass,
        params={
            "body_name": "base_link",
            "normalize": True,
            "mass_delta_range": GalileoDefaults.priv_obs_norm.base_mass_delta_range,
        },
    )
    ground_friction = ObsTerm(
        func=mdp.ground_friction,
        params={
            "normalize": True,
            "friction_range": GalileoDefaults.priv_obs_norm.ground_friction_range,
        },
    )


@configclass
class CTSObservationsCfg:
    """CTS benchmark observations for blind omni-directional locomotion."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel_nad = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity_nad = ObsTerm(func=mdp.projected_gravity, scale=1.0)
        joint_pos_rel_nad = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        joint_vel_rel_nad = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions_gt = ObsTerm(func=mdp.last_action, scale=0.25)
        base_commands_gt = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        student_height_scan = _student_height_scan_obs_term()
        teacher_base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        teacher_base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        teacher_projected_gravity = ObsTerm(func=mdp.projected_gravity)
        teacher_joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        teacher_joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        teacher_actions = ObsTerm(func=mdp.last_action)
        teacher_velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        teacher_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        )
        teacher_base_com = ObsTerm(
            func=mdp.base_com,
            params={
                "body_name": "base_link",
                "normalize": True,
                "com_range": GalileoDefaults.priv_obs_norm.base_com_range,
            },
        )
        teacher_base_mass = ObsTerm(
            func=mdp.base_mass,
            params={
                "body_name": "base_link",
                "normalize": True,
                "mass_delta_range": GalileoDefaults.priv_obs_norm.base_mass_delta_range,
            },
        )
        teacher_ground_friction = ObsTerm(
            func=mdp.ground_friction,
            params={
                "normalize": True,
                "friction_range": GalileoDefaults.priv_obs_norm.ground_friction_range,
            },
        )
        proprio_history = ObsTerm(
            func=mdp.PolicyHistory,
            params={
                "history_length": GalileoDefaults.obs.cts.actor_num_hist,
                "include_base_lin_vel": False,
                "command_name": "base_velocity",
                "scales": {
                    "base_lin_vel": 1.0,
                    "base_ang_vel": 0.25,
                    "projected_gravity": 1.0,
                    "joint_pos": 1.0,
                    "joint_vel": 0.05,
                    "last_action": 0.25,
                    "commands": (2.0, 2.0, 0.25),
                },
            },
        )

    @configclass
    class CriticCfg(_PrivilegedObsGroupCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CTSCostsCfg:
    """Constraint terms used by all CTS benchmark algorithms."""

    prob_joint_pos = CostTerm(
        func=mdp.joint_pos_prob_constraint,
        weight=1.0,
        params={
            "margin": 0.0,
            "limit": 1.0,
            "asset_cfg": LEG_JOINT_CFG,
        },
    )
    prob_joint_vel = CostTerm(
        func=mdp.joint_vel_prob_constraint,
        weight=1.0,
        params={
            "limit": 15.0,
            "soft_ratio": 0.95,
            "cost_limit": 1.0,
            "asset_cfg": LEG_JOINT_CFG,
        },
    )
    prob_joint_torque = CostTerm(
        func=mdp.joint_torque_prob_constraint,
        weight=1.0,
        params={
            "limit": 80.0,
            "soft_ratio": 0.95,
            "cost_limit": 1.0,
            "asset_cfg": LEG_JOINT_CFG,
        },
    )


# Backwards-compatible generic alias used by helper modules.
CostsCfg = CTSCostsCfg


@configclass
class CTSRewardsCfg:
    """Shared locomotion reward for the CTS blind-locomotion benchmark."""

    only_positive_rewards: bool = True

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
            "min_command_speed": None,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.7,
        params={
            "command_name": "base_velocity",
            "std": 0.3,
            "min_command_speed": None,
        },
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torque_l2,
        weight=-3.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-9,
    )
    dof_error_l2 = RewTerm(
        func=mdp.dof_error_l2,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "low_speed_threshold": 0.1,
            "high_speed_threshold": 0.35,
            "low_speed_scale": 1.2,
            "high_speed_scale": 0.15,
        },
    )
    hip_pos_l2 = RewTerm(
        func=mdp.hip_pos_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "command_name": "base_velocity",
            "low_speed_threshold": 0.1,
            "high_speed_threshold": 0.35,
            "low_speed_scale": 1.5,
            "high_speed_scale": 0.8,
        },
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1.0e-3,
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.25,
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "default_scale": 1.5,
            "terrain_scales": {
                "plane_run": 1.5,
                "plane_yaw": 1.5,
                "plane_stand": 1.5,
                "random_rough": 1.5,
                "boxes": 1.2,
                "pyramid_stairs": 0.25,
                "pyramid_stairs_inv": 0.25,
                "hf_pyramid_slope": 0.25,
                "hf_pyramid_slope_inv": 0.25,
            },
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.25,
            "low_speed_threshold": 0.25,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "command_name": "base_velocity",
            "target_height": 0.06,
            "low_speed_threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.35,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
        },
    )
    gait_contact_symmetry = RewTerm(
        func=mdp.gait_contact_symmetry,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "left_foot_names": LEFT_FOOT_BODY_NAMES,
            "right_foot_names": RIGHT_FOOT_BODY_NAMES,
            "contact_threshold": 5.0,
            "command_name": None,
        },
    )
    trot_phase_reward = RewTerm(
        func=mdp.trot_phase_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": FOOT_BODY_NAMES,
            "contact_threshold": 5.0,
            "contact_smoothing": 1.5,
            "ema_decay": 0.9,
            "command_name": "base_velocity",
            "min_command_speed": 0.12,
            "low_speed_threshold": 0.35,
            "max_abs_yaw_cmd": 0.3,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": GalileoDefaults.termination.fallen_limit_angle,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class EventCfg:
    """域随机化配置（按照优化后的清单）。

    一、机身/动力学（reset 时）
    - 机身质量：randomize_base_mass
    - 机身质心：randomize_base_com
    - 摩擦系数：physics_material（静摩擦、动摩擦、恢复系数）

    二、重置位姿/速度（reset 时）
    - 基座位姿：reset_base_pose
    - 基座速度：reset_base_velocity（仅线速度随机）

    三、腿部关节（reset 时）
    - 腿部关节位置：reset_leg_joints

    四、执行器增益（reset 时）
    - Kp/Kd：randomize_actuator_gains
    - Kt：需要扩展功能（当前不支持）

    五、外力扰动（interval，训练中周期施加）
    - 速度扰动：push_robot_vel
    - 力矩扰动：push_robot_torque
    """

    # ========== 一、机身/动力学（reset 时） ==========
    # 机身质量随机化（对 base_link 质量做加性随机）
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": GalileoDefaults.event.randomize_base_mass.mass_distribution_params,
            "operation": GalileoDefaults.event.randomize_base_mass.operation,
        },
    )

    # 机身质心随机化（对 base_link 的 CoM 在 body 系下随机偏移）
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": GalileoDefaults.event.randomize_base_com.com_range,
        },
    )

    # 物理材质随机化（静摩擦、动摩擦、恢复系数）
    # 注意：restitution_range 在 __init__ 中使用，但不作为 __call__ 的参数
    # 我们需要在配置后手动处理，但由于事件管理器在创建实例前检查参数，
    # 我们需要使用一个包装函数或者修改 __call__ 签名来接受但不使用这个参数
    # 暂时移除 make_consistent，因为它会导致参数检查失败
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": GalileoDefaults.event.physics_material.friction_range,
            "restitution_range": GalileoDefaults.event.physics_material.restitution_range,
            "num_buckets": GalileoDefaults.event.physics_material.num_buckets,
        },
    )

    # ========== 二、重置位姿/速度（reset 时） ==========
    # 基座位姿随机化
    reset_base_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": GalileoDefaults.event.reset_base_pose.pose_range,
            "velocity_range": GalileoDefaults.event.reset_base_pose.velocity_range,
        },
        mode="reset",
    )

    # ========== 三、腿部关节（reset 时） ==========
    # 腿部关节位置随机化（仅对 12 个腿关节）
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": LEG_JOINT_CFG,
            "position_range": GalileoDefaults.event.reset_leg_joints.position_range,
            "velocity_range": GalileoDefaults.event.reset_leg_joints.velocity_range,
        },
    )

    # ========== 四、执行器增益（reset 时） ==========
    # Kp/Kd 增益随机化（对腿 + 臂执行器）
    randomize_actuator_kp_kd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),  # 所有关节
            "stiffness_distribution_params": GalileoDefaults.event.randomize_actuator_kp_kd_gains.stiffness_distribution_params,
            "damping_distribution_params": GalileoDefaults.event.randomize_actuator_kp_kd_gains.damping_distribution_params,
            "operation": GalileoDefaults.event.randomize_actuator_kp_kd_gains.operation,
            "distribution": GalileoDefaults.event.randomize_actuator_kp_kd_gains.distribution,
        },
    )

    # ========== 五、外力扰动（interval，训练中周期施加） ==========
    # 速度扰动（对基座施加瞬时速度扰动）
    push_robot_vel = EventTerm(
        func=mdp.push_by_setting_velocity,
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # push_by_setting_velocity 作用于整个 asset
            "velocity_range": GalileoDefaults.event.push_robot_vel.velocity_range,
        },
        interval_range_s=GalileoDefaults.event.push_robot_vel.interval_range_s,
        is_global_time=GalileoDefaults.event.push_robot_vel.is_global_time,
        mode="interval",
    )

    # 力矩扰动（对 base_link 施加外力矩）
    push_robot_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": GalileoDefaults.event.push_robot_torque.force_range,
            "torque_range": GalileoDefaults.event.push_robot_torque.torque_range,
        },
        interval_range_s=GalileoDefaults.event.push_robot_torque.interval_range_s,
        is_global_time=GalileoDefaults.event.push_robot_torque.is_global_time,
        mode="interval",
    )

    # 保留相机随机化（如果需要）
    random_camera_position = EventTerm(
        func=mdp.random_camera_position,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "rot_noise_range": {"pitch": (-5, 5)},
            "convention": "ros",
        },
    )


@configclass
class CurriculumCfg:
    # Ported from extreme_load: terrain rows progress by distance walked, and
    # the x/yaw command envelopes widen on a fixed episode-step schedule.
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={
            "move_up_ratio": GalileoDefaults.curriculum.terrain_move_up_ratio,
            "move_down_ratio": GalileoDefaults.curriculum.terrain_move_down_ratio,
        },
    )
    lin_vel_x_command_threshold = CurrTerm(
        func=mdp.lin_vel_x_command_threshold,
        params={
            "episodes_per_level": GalileoDefaults.curriculum.episodes_per_level,
            "warmup_steps": GalileoDefaults.curriculum.command_warmup_steps,
            "min_progress_steps": GalileoDefaults.curriculum.command_min_progress_steps,
            "terrain_level_threshold": GalileoDefaults.curriculum.command_terrain_gate_level,
            "error_threshold": GalileoDefaults.curriculum.lin_tracking_error_threshold,
            "min_command_speed": GalileoDefaults.curriculum.lin_eval_min_command_speed,
            "min_active_ratio": GalileoDefaults.curriculum.command_min_active_ratio,
        },
    )
    ang_vel_z_command_threshold = CurrTerm(
        func=mdp.ang_vel_z_command_threshold,
        params={
            "episodes_per_level": GalileoDefaults.curriculum.episodes_per_level,
            "warmup_steps": GalileoDefaults.curriculum.command_warmup_steps,
            "min_progress_steps": GalileoDefaults.curriculum.command_min_progress_steps,
            "terrain_level_threshold": GalileoDefaults.curriculum.command_terrain_gate_level,
            "error_threshold": GalileoDefaults.curriculum.ang_tracking_error_threshold,
            "min_command_speed": GalileoDefaults.curriculum.ang_eval_min_command_speed,
            "min_active_ratio": GalileoDefaults.curriculum.command_min_active_ratio,
            "min_lin_x_level": GalileoDefaults.curriculum.ang_min_lin_x_level,
        },
    )
    lin_vel_y_command_threshold = None
    terrain_type_progression = None
    domain_randomization_scale = None


@configclass
class CTSCurriculumCfg(CurriculumCfg):
    """CTS benchmark uses a single unified command-and-terrain curriculum."""


@configclass
class ActionsCfg:
    """动作配置（延迟的关节位置控制）。

    - action_delay_steps: 两帧管线延迟模拟通讯/执行滞后。
    - history_length: 叠加历史动作，帮助策略理解真实控制通道。
    """

    joint_pos = mdp.DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        preserve_order=True,
        scale=GalileoDefaults.action.scale,
        use_default_offset=GalileoDefaults.action.use_default_offset,
        action_delay_steps=GalileoDefaults.action.action_delay_steps,
        delay_update_global_steps=GalileoDefaults.action.delay_update_global_steps,
        history_length=GalileoDefaults.action.history_length,
        use_delay=GalileoDefaults.action.use_delay,
        clip=GalileoDefaults.action.clip,
    )
