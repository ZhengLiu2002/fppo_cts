"""Shared defaults for Galileo CRL tasks."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from crl_isaaclab.actuators.crl_actuator_cfg import CRLDCMotorCfg
from crl_isaaclab.terrains import CRLTerrainImporter
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

try:
    from galileo_parkour.assets.galileo import GALILEO_CFG as _GALILEO_CFG
except ImportError:
    _GALILEO_CFG = None

# File location: crl_tasks/crl_tasks/tasks/galileo/config/defaults.py
# parents[0]=config, [1]=galileo, [2]=tasks, [3]=crl_tasks, [4]=crl_tasks, [5]=repo root
DEFAULT_GALILEO_USD_PATH = (
    Path(__file__).resolve().parents[5]
    / "source/extensions/galileo_parkour/galileo_parkour/assets/usd/robot/galileo_v2d3.usd"
)
GALILEO_USD_PATH = os.environ.get("GALILEO_USD_PATH", str(DEFAULT_GALILEO_USD_PATH))


def _build_galileo_robot_cfg() -> ArticulationCfg:
    """Build the Galileo articulation configuration."""
    if _GALILEO_CFG is not None:
        return _GALILEO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GALILEO_USD_PATH,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            joint_pos={
                "FL_hip_joint": -0.05,
                "FL_thigh_joint": 0.75,
                "FL_calf_joint": -1.5,
                "FR_hip_joint": 0.05,
                "FR_thigh_joint": 0.75,
                "FR_calf_joint": -1.5,
                "RL_hip_joint": -0.05,
                "RL_thigh_joint": 0.75,
                "RL_calf_joint": -1.5,
                "RR_hip_joint": 0.05,
                "RR_thigh_joint": 0.75,
                "RR_calf_joint": -1.5,
            },
        ),
        actuators={},
    )


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    """Convert Euler XYZ to quaternion (ROS convention)."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.0, 1.0, 1.0, -1.0])
    return tuple(convert.numpy().tolist())


@configclass
class GalileoBaseSceneCfg(InteractiveSceneCfg):
    """Default Galileo scene configuration shared across tasks."""

    robot: ArticulationCfg = _build_galileo_robot_cfg()

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type=CRLTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        self.robot.actuators["base_legs"] = CRLDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                ".*_hip_joint": 80.0,
                ".*_thigh_joint": 80.0,
                ".*_calf_joint": 80.0,
            },
            saturation_effort={
                ".*_hip_joint": 80.0,
                ".*_thigh_joint": 80.0,
                ".*_calf_joint": 80.0,
            },
            velocity_limit={
                ".*_hip_joint": 16.0,
                ".*_thigh_joint": 16.0,
                ".*_calf_joint": 16.0,
            },
            stiffness=70.0,
            damping=3.5,
            friction=0.05,
        )


VIEWER_CFG = ViewerCfg(
    eye=(-0.0, 2.6, 1.6),
    asset_name="robot",
    origin_type="asset_root",
)


# ========== 地形配置 ==========
# Rough terrain mix used for Galileo FPPO tasks.
GALILEO_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "crl_flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
    },
)


class GalileoDefaults:
    class general:
        decimation = 8
        episode_length_s = 20.0
        render_interval = 4

    class curriculum:
        # Command curriculum uses a gated progression policy:
        # warmup -> minimum interval between upgrades -> tracking quality gate.
        # episodes_per_level is retained for backward compatibility with legacy
        # time-based logic in curriculums.py.
        episodes_per_level = 15
        terrain_move_up_ratio = 0.5
        terrain_move_down_ratio = 0.1
        command_warmup_steps = 4800
        command_min_progress_steps = 7200
        command_terrain_gate_level = 4.0
        command_min_active_ratio = 0.12
        lin_tracking_error_threshold = 0.22
        ang_tracking_error_threshold = 0.30
        lin_eval_min_command_speed = 0.20
        ang_eval_min_command_speed = 0.16
        ang_min_lin_x_level = 0.32

    class sim:
        dt = 0.0025

    class env:
        num_envs = 4096
        play_num_envs = 16
        eval_num_envs = 128

    class obs:
        """CTS-only actor/critic observation layout."""

        # 通用环境/动作维度
        num_envs = 4096
        num_actions = 12

        class cts:
            actor_num_prop = 45
            actor_num_scan = 0
            actor_num_priv_explicit = 0
            actor_num_priv_latent = 185
            actor_history_latent_dim = 32
            actor_num_hist = 20
            num_actor_obs = (
                actor_num_prop
                + actor_num_scan
                + actor_num_priv_explicit
                + actor_num_priv_latent
                + actor_num_prop * actor_num_hist
            )

            critic_num_prop = 48
            critic_num_scan = 132
            critic_num_priv_explicit = 5
            critic_num_priv_latent = 0
            critic_num_hist = 0
            num_critic_obs = (
                critic_num_prop
                + critic_num_scan
                + critic_num_priv_explicit
                + critic_num_priv_latent
                + critic_num_prop * critic_num_hist
            )

        # 观测历史相关（如需）
        obs_history_length = 20
        action_history_length = 3
        clip_actions = 100.0
        clip_obs = 100.0

    class terrain:
        size = (8.0, 8.0)
        border_width = 60.0
        num_rows = 10
        num_cols = 20
        horizontal_scale = 0.1
        vertical_scale = 0.005
        slope_threshold = 0.75
        curriculum = True
        random_difficulty = False
        difficulty_range = (0.0, 1.0)
        # Flat-only pretraining mode keeps the CTS task on plane terrain until
        # FPPO stabilizes, while preserving the rough-terrain generator for later.
        flat_only_pretrain = True
        flat_subterrain_name = "crl_flat"

    class command:
        lin_x_level: float = 0.3
        max_lin_x_level: float = 1.0
        ang_z_level: float = 0.3
        max_ang_z_level: float = 1.0
        lin_x_level_step: float = 0.01
        ang_z_level_step: float = 0.01
        min_abs_lin_vel_x: float = 0.0
        min_abs_lin_vel_y: float = 0.0

        # 默认配置（用于 CommandsCfg 的初始化，不在地形特定配置中）
        class default:
            standing_command_prob: float = 0.10
            lin_vel_x = (-0.5, 1.0)
            lin_vel_y = (-0.5, 0.5)
            ang_vel_z = (-0.5, 0.5)
            start_curriculum_lin_x = (-0.15, 0.3)
            start_curriculum_ang_z = (-0.08, 0.08)
            max_curriculum_lin_x = (-0.5, 1.0)
            max_curriculum_ang_z = (-0.5, 0.5)

        ranges = {
            "pyramid_stairs": dict(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                standing_command_prob=0.10,
                start_curriculum_lin_x=(-0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(-0.5, 1.0),
                max_curriculum_ang_z=(-0.5, 0.5),
            ),
            "pyramid_stairs_inv": dict(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                standing_command_prob=0.10,
                start_curriculum_lin_x=(-0.12, 0.25),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(-0.5, 1.0),
                max_curriculum_ang_z=(-0.5, 0.5),
            ),
            "boxes": dict(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                standing_command_prob=0.10,
                start_curriculum_lin_x=(-0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(-0.5, 1.0),
                max_curriculum_ang_z=(-0.5, 0.5),
            ),
            "random_rough": dict(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                standing_command_prob=0.10,
                start_curriculum_lin_x=(-0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(-0.5, 1.0),
                max_curriculum_ang_z=(-0.5, 0.5),
            ),
            "hf_pyramid_slope": dict(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                standing_command_prob=0.10,
                start_curriculum_lin_x=(-0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(-0.5, 1.0),
                max_curriculum_ang_z=(-0.5, 0.5),
            ),
            "hf_pyramid_slope_inv": dict(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                standing_command_prob=0.10,
                start_curriculum_lin_x=(-0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(-0.5, 1.0),
                max_curriculum_ang_z=(-0.5, 0.5),
            ),
            "crl_flat": dict(
                lin_vel_x=(-0.5, 1.5),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
                standing_command_prob=0.10,
                start_curriculum_lin_x=(-0.2, 0.5),
                start_curriculum_ang_z=(-0.12, 0.12),
                max_curriculum_lin_x=(-0.5, 1.5),
                max_curriculum_ang_z=(-0.5, 0.5),
            ),
        }

        resampling_time_range = (6.0, 6.0)
        clips = dict(lin_vel_clip=0.1, ang_vel_clip=0.02)

    class priv_obs_norm:
        """Normalization ranges for privileged observations."""

        base_mass_delta_range = (-3.0, 5.0)
        base_com_range = {
            "x": (-0.05, 0.05),
            "y": (-0.03, 0.03),
            "z": (-0.03, 0.05),
        }
        ground_friction_range = (0.25, 1.2)

    class event:
        """域随机化参数配置"""

        # ========== 一、机身/动力学（reset 时） ==========
        class randomize_base_mass:
            mass_distribution_params = (-3.0, 5.0)  # kg（加性）
            operation = "add"

        class randomize_base_com:
            com_range = {
                "x": (-0.05, 0.05),  # m
                "y": (-0.03, 0.03),  # m
                "z": (-0.03, 0.05),  # m
            }

        class physics_material:
            friction_range = (0.25, 1.2)  # 静摩擦和动摩擦使用相同范围
            restitution_range = (0.0, 1.0)  # 恢复系数
            num_buckets = 64

        # ========== 二、重置位姿/速度（reset 时） ==========
        class reset_base_pose:
            pose_range = {
                "x": (-0.5, 0.5),  # m
                "y": (-0.5, 0.5),  # m
                "yaw": (-math.pi, math.pi),  # rad
            }
            velocity_range = {
                "x": (-0.5, 0.5),  # m/s
                "y": (-0.5, 0.5),  # m/s
                "z": (-0.5, 0.5),  # m/s
                "roll": (0.0, 0.0),  # rad/s（角速度不随机）
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }

        # ========== 三、腿部关节（reset 时） ==========
        class reset_leg_joints:
            position_range = (-0.8, 0.8)  # 标幺/偏移
            velocity_range = (0.0, 0.0)

        # ========== 四、执行器增益（reset 时） ==========
        class randomize_actuator_kp_kd_gains:
            stiffness_distribution_params = (0.8, 1.2)  # Kp 乘性
            damping_distribution_params = (0.8, 1.2)  # Kd 乘性
            operation = "scale"
            distribution = "uniform"

        # ========== 五、外力扰动（interval，训练中周期施加） ==========
        class push_robot_vel:
            velocity_range = {
                "x": (-0.3, 0.3),  # m/s
                "y": (-0.3, 0.3),  # m/s
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            interval_range_s = (8.0, 8.0)
            is_global_time = True

        class push_robot_torque:
            force_range = (0.0, 0.0)
            torque_range = (-1.0, 1.0)  # N·m
            interval_range_s = (8.0, 8.0)
            is_global_time = True

    class algorithm:
        """CTS benchmark algorithm profiles.

        The CTS task is the only training scaffold in the repo. Algorithm
        selection changes the constrained-RL update rule while keeping the same
        blind omni-directional locomotion task, observation layout, reward, and
        constraint definitions.
        """

        # 与 CLI `--algo` 对齐：{"cts","fppo","np3o","ppo","ppo_lagrange","cpo","pcpo","focops"}
        name: str = "fppo"

        # CLI/代码中的 class_name 映射（最终写入 agent_cfg.algorithm.class_name）
        class_name_map = {
            "cts": "CTS",
            "fppo": "FPPO",
            "np3o": "NP3O",
            "ppo": "PPO",
            "ppo_lagrange": "PPOLagrange",
            "cpo": "CPO",
            "pcpo": "PCPO",
            "focops": "FOCOPS",
        }

        # 共享默认值（两边都通用）
        base = dict(
            # PPO common
            value_loss_coef=0.5,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            desired_kl=0.004,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            max_grad_norm=1.0,
            # CMDP common
            cost_limit=1.3,
            # NP3O-style shared shaping
            cost_viol_loss_coef=0.1,
            k_value=0.2,
            k_growth=1.0003,
            k_max=1.0,
        )

        shared_constraint_limits = dict(
            base_contact_force=0.02,
            contact_velocity=0.025,
            foot_clearance=0.08,
            foot_height_limit=0.04,
            orthogonal_velocity=0.20,
            prob_body_contact=0.03,
            prob_com_angle=0.02,
            prob_com_height=0.18,
            prob_gait_pattern=0.05,
            prob_joint_pos=0.15,
            prob_joint_torque=0.04,
            prob_joint_vel=0.08,
            symmetric=0.10,
        )
        shared_constraint_limits_start = {
            name: float(limit * 2.0) for name, limit in shared_constraint_limits.items()
        }
        constraint_adapter_base = dict()
        constraint_adapter_per_algo = {
            "fppo": dict(
                enabled=False,
                ema_beta=0.9,
                min_scale=1e-3,
                max_scale=10.0,
                clip=5.0,
                huber_delta=0.2,
                agg_tau=0.5,
                scale_by_gamma=True,
                cost_scale=None,
            ),
        }
        # 各算法差异化字段（只写需要的即可）
        per_algo = {
            "cts": dict(
                value_loss_coef=0.5,
                cost_value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                desired_kl=0.01,
                num_learning_epochs=5,
                num_mini_batches=4,
                learning_rate=3.0e-4,
                reconstruction_learning_rate=1.0e-3,
                num_reconstruction_epochs=2,
                schedule="adaptive",
                gamma=0.99,
                lam=0.95,
                max_grad_norm=1.0,
                entropy_coef=0.01,
                student_group_ratio=0.25,
                detach_student_encoder_during_rl=True,
                cost_limit=1.3,
                cost_viol_loss_coef=0.1,
                k_value=0.2,
                k_growth=1.0003,
                k_max=1.0,
                constraint_limits={
                    "prob_joint_pos": 0.15,
                    "prob_joint_vel": 0.08,
                    "prob_joint_torque": 0.04,
                },
            ),
            # FPPO（更激进参数：更大步长、更松约束、更少回溯）
            "fppo": dict(
                cost_value_loss_coef=1.0,
                num_learning_epochs=4,
                learning_rate=5.0e-4,
                desired_kl=0.006,
                delta_kl=0.010,
                predictor_desired_kl=0.015,
                predictor_kl_hard_limit=0.03,
                step_size=4.0e-4,
                cost_gamma=None,
                cost_lam=None,
                epsilon_safe=0.002,
                backtrack_coeff=0.5,
                max_backtracks=10,
                projection_eps=1e-6,
                softproj_max_iters=40,
                softproj_tol=1e-6,
                constraint_limits=dict(shared_constraint_limits),
                constraint_limits_start=dict(shared_constraint_limits_start),
                constraint_limits_final=dict(shared_constraint_limits),
                adaptive_constraint_curriculum=True,
                constraint_curriculum_names=[
                    "prob_joint_pos",
                    "prob_joint_vel",
                    "prob_joint_torque",
                ],
                constraint_curriculum_ema_decay=0.95,
                constraint_curriculum_check_interval=40,
                constraint_curriculum_alpha=0.7,
                constraint_curriculum_shrink=0.985,
                step_size_adaptive=False,
            ),
            # NP3O
            "np3o": dict(
                constraint_limits=dict(shared_constraint_limits),
                cost_value_loss_coef=1.0,
                learning_rate=2.5e-4,
                schedule="adaptive",
                desired_kl=0.007,
                cost_limit=1.6,
                cost_gamma=None,
                cost_lam=None,
                normalize_cost_advantage=True,
                cost_viol_loss_coef=0.12,
                k_value=0.03,
                k_growth=1.00007,
                k_max=0.4,
                k_decay=0.9999,
                k_min=0.01,
                k_violation_threshold=0.02,
            ),
            # PPO（示例：如果你切到 PPO，只需要写 PPO 特有/你要覆写的字段）
            "ppo": dict(
                constraint_limits=dict(shared_constraint_limits),
            ),
            # PPO-Lagrange：拉格朗日乘子惩罚约束
            "ppo_lagrange": dict(
                constraint_limits=dict(shared_constraint_limits),
                cost_value_loss_coef=1.0,
                cost_gamma=None,
                cost_lam=None,
                normalize_cost_advantage=False,
                lagrangian_multiplier_init=0.0,
                lagrange_lr=1.0e-2,
                lagrange_optimizer="Adam",
                lagrange_max=100.0,
            ),
            "cpo": dict(
                constraint_limits=dict(shared_constraint_limits),
                cost_value_loss_coef=1.0,
                cost_gamma=None,
                cost_lam=None,
                normalize_cost_advantage=False,
                desired_kl=0.01,
                backtrack_coeff=0.8,
                max_backtracks=20,
                cg_iters=10,
                cg_damping=1.0e-2,
                fvp_sample_freq=1,
            ),
            "pcpo": dict(
                constraint_limits=dict(shared_constraint_limits),
                cost_value_loss_coef=1.0,
                cost_gamma=None,
                cost_lam=None,
                normalize_cost_advantage=False,
                desired_kl=0.01,
                backtrack_coeff=0.8,
                max_backtracks=20,
                cg_iters=10,
                cg_damping=1.0e-2,
                fvp_sample_freq=1,
            ),
            "focops": dict(
                constraint_limits=dict(shared_constraint_limits),
                cost_value_loss_coef=1.0,
                cost_gamma=None,
                cost_lam=None,
                normalize_cost_advantage=False,
                lagrangian_multiplier_init=0.0,
                lagrange_lr=1.0e-2,
                lagrange_optimizer="Adam",
                lagrange_max=100.0,
                focops_eta=0.02,
                focops_lambda=1.0,
            ),
        }


# -----------------------------------------------------------------------------
# Convenience aliases for common naming patterns.
# -----------------------------------------------------------------------------

ConfigSummary = GalileoDefaults
CRLDefaultSceneCfg = GalileoBaseSceneCfg
VIEWER = VIEWER_CFG
GALILEO_ROUGH_TERRAINS_CFG = GALILEO_ROUGH_TERRAIN_CFG

build_galileo_robot_cfg = _build_galileo_robot_cfg
_galileo_robot_cfg = _build_galileo_robot_cfg

quat_from_euler_xyz_tuple = quat_from_euler_xyz

__all__ = [
    "DEFAULT_GALILEO_USD_PATH",
    "GALILEO_USD_PATH",
    "build_galileo_robot_cfg",
    "_galileo_robot_cfg",
    "quat_from_euler_xyz",
    "quat_from_euler_xyz_tuple",
    "GalileoBaseSceneCfg",
    "CRLDefaultSceneCfg",
    "VIEWER_CFG",
    "VIEWER",
    "GALILEO_ROUGH_TERRAIN_CFG",
    "GALILEO_ROUGH_TERRAINS_CFG",
    "GalileoDefaults",
    "ConfigSummary",
]
