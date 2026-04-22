from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pytest

pytest.importorskip("torch")

from scripts.rsl_rl import exporter


JOINT_NAMES = [
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

DEFAULT_JOINT_POS = [-0.05, 0.05, -0.05, 0.05, 0.75, 0.75, 0.75, 0.75, -1.5, -1.5, -1.5, -1.5]


def _term(scale=None, params=None):
    return SimpleNamespace(func="obs_term", scale=scale, params=dict(params or {}))


def _policy_terms(command_scale):
    return SimpleNamespace(
        base_ang_vel_nad=_term(scale=0.25),
        projected_gravity_nad=_term(scale=1.0),
        joint_pos_rel_nad=_term(scale=1.0),
        joint_vel_rel_nad=_term(scale=0.05),
        actions_gt=_term(scale=0.25),
        base_commands_gt=_term(
            scale=command_scale,
            params={"command_name": "base_velocity"},
        ),
        teacher_velocity_commands=_term(
            scale=None,
            params={"command_name": "base_velocity"},
        ),
        proprio_history=_term(
            scale=None,
            params={
                "history_length": 10,
                "include_base_lin_vel": False,
                "command_name": "base_velocity",
                "scales": {
                    "base_ang_vel": 0.25,
                    "projected_gravity": 1.0,
                    "joint_pos": 1.0,
                    "joint_vel": 0.05,
                    "last_action": 0.25,
                    "commands": command_scale,
                },
            },
        ),
    )


def _config_summary():
    return SimpleNamespace(
        observation=SimpleNamespace(export_name_map={"teacher_velocity_commands": "base_commands_gt"}),
        env=SimpleNamespace(clip_actions=None, clip_obs=None),
        action=SimpleNamespace(scale=None),
        command=None,
    )


def _actor_critic_stub():
    actor = SimpleNamespace(
        num_prop=45,
        num_hist=10,
        num_scan=0,
        num_priv_explicit=0,
        num_priv_latent=235,
        student_in_features=45 + 45 * 10,
        in_features=45 + 235 + 45 * 10,
    )
    return SimpleNamespace(actor=actor, is_recurrent=False)


def _scene_robot(stiffness, damping, torque, default_joint_pos=None):
    return SimpleNamespace(
        init_state=SimpleNamespace(
            joint_pos=dict(zip(JOINT_NAMES, default_joint_pos or DEFAULT_JOINT_POS, strict=True))
        ),
        actuators={
            "base_legs": SimpleNamespace(
                stiffness=stiffness,
                damping=damping,
                effort_limit={
                    ".*_hip_joint": torque,
                    ".*_thigh_joint": torque,
                    ".*_calf_joint": torque,
                },
            )
        },
    )


def _env_cfg(
    *,
    action_scale,
    command_scale,
    stiffness,
    damping,
    torque,
    ranges,
    max_lin_x_level=1.0,
    max_ang_z_level=1.0,
    default_joint_pos=None,
):
    return SimpleNamespace(
        decimation=8,
        sim=SimpleNamespace(dt=0.0025),
        actions=SimpleNamespace(joint_pos=SimpleNamespace(joint_names=list(JOINT_NAMES), scale=action_scale)),
        observations=SimpleNamespace(policy=_policy_terms(command_scale)),
        scene=SimpleNamespace(
            robot=_scene_robot(
                stiffness=stiffness,
                damping=damping,
                torque=torque,
                default_joint_pos=default_joint_pos,
            )
        ),
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                ranges=dict(ranges),
                max_lin_x_level=max_lin_x_level,
                max_ang_z_level=max_ang_z_level,
                velocity_x_forward_scale=1.0,
                velocity_x_backward_scale=0.8,
                velocity_y_scale=0.4,
                velocity_yaw_scale=1.0,
                max_velocity=(1.0, 0.4, 1.5),
            )
        ),
        config_summary=_config_summary(),
    )


class ExporterPolicyCfgTest(unittest.TestCase):
    def test_build_export_input_layout_keeps_actor_scale_when_teacher_alias_collides(self) -> None:
        raw_obs_names = [
            "base_ang_vel_nad",
            "projected_gravity_nad",
            "joint_pos_rel_nad",
            "joint_vel_rel_nad",
            "actions_gt",
            "base_commands_gt",
            "teacher_velocity_commands",
            "proprio_history",
        ]
        term_cfgs = [
            _term(scale=0.25),
            _term(scale=1.0),
            _term(scale=1.0),
            _term(scale=0.05),
            _term(scale=0.25),
            _term(scale=(2.0, 2.0, 0.25), params={"command_name": "base_velocity"}),
            _term(scale=None, params={"command_name": "base_velocity"}),
            _term(
                scale=None,
                params={
                    "history_length": 10,
                    "include_base_lin_vel": False,
                    "command_name": "base_velocity",
                    "scales": {
                        "base_ang_vel": 0.25,
                        "projected_gravity": 1.0,
                        "joint_pos": 1.0,
                        "joint_vel": 0.05,
                        "last_action": 0.25,
                        "commands": (2.0, 2.0, 0.25),
                    },
                },
            ),
        ]
        obs_mgr = SimpleNamespace(
            group_obs_dim={},
            _group_obs_term_names={"policy": raw_obs_names},
            _group_obs_term_cfgs={"policy": term_cfgs},
        )
        fake_env_cfg = SimpleNamespace(config_summary=_config_summary())
        fake_env = SimpleNamespace(
            unwrapped=SimpleNamespace(
                observation_manager=obs_mgr,
                cfg=fake_env_cfg,
            )
        )

        layout = exporter._build_export_input_layout(fake_env, fake_env_cfg, actor_critic=_actor_critic_stub())

        self.assertEqual(
            layout["input_obs_scales_map"]["actor_obs"]["base_commands_gt"],
            [2.0, 2.0, 0.25],
        )
        self.assertEqual(
            layout["input_obs_scales_map"]["proprio_history"]["base_commands_gt"],
            [2.0, 2.0, 0.25],
        )

    def test_export_inference_cfg_uses_live_runtime_metadata(self) -> None:
        live_env_cfg = _env_cfg(
            action_scale=0.9,
            command_scale=1.0,
            stiffness=11.0,
            damping=1.1,
            torque=22.0,
            ranges={
                "lin_vel_x": (-0.1, 0.2),
                "lin_vel_y": (-0.1, 0.1),
                "ang_vel_z": (-0.2, 0.2),
            },
            default_joint_pos=[0.1] * len(JOINT_NAMES),
        )
        live_env = SimpleNamespace(
            unwrapped=SimpleNamespace(
                cfg=live_env_cfg,
                physics_dt=None,
                step_dt=0.5,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            export_cfg = exporter.export_inference_cfg(
                live_env,
                live_env_cfg,
                path=str(run_dir / "exported_policy"),
                agent_cfg=SimpleNamespace(clip_actions=5.0),
                actor_critic=_actor_critic_stub(),
            )

        self.assertEqual(export_cfg["dt"], 0.02)
        self.assertEqual(export_cfg["action_scale"], 0.9)
        self.assertEqual(export_cfg["clip_actions"], 5.0)
        self.assertEqual(export_cfg["joint_kp"], [11.0] * len(JOINT_NAMES))
        self.assertEqual(export_cfg["joint_kd"], [1.1] * len(JOINT_NAMES))
        self.assertEqual(export_cfg["max_torques"], [22.0] * len(JOINT_NAMES))
        self.assertEqual(export_cfg["default_joint_pos"], [0.1] * len(JOINT_NAMES))
        self.assertEqual(
            export_cfg["input_obs_scales_map"]["actor_obs"]["base_commands_gt"],
            1.0,
        )
        self.assertEqual(export_cfg["obs_history_length"], {"actor_obs": 1, "proprio_history": 10})
        self.assertEqual(export_cfg["max_velocity"], [1.0, 0.4, 1.5])
        self.assertEqual(export_cfg["velocity_x_forward_scale"], 1.0)
        self.assertEqual(export_cfg["velocity_x_backward_scale"], 0.8)
        self.assertEqual(export_cfg["velocity_y_scale"], 0.4)
        self.assertEqual(export_cfg["velocity_yaw_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
