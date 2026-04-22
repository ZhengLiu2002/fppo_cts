from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from scripts.rsl_rl.algorithms.cts import CTS
from scripts.rsl_rl.modules.actor_critic_with_encoder import ActorCriticRMA


def _make_cts_policy() -> ActorCriticRMA:
    return ActorCriticRMA(
        num_critic_obs=13,
        num_actions=3,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        cost_critic_hidden_dims=[32, 16],
        activation="elu",
        init_noise_std=1.0,
        normalize_latent=True,
        tanh_encoder_output=False,
        scan_encoder_dims=[],
        priv_encoder_dims=[16],
        num_prop=5,
        num_scan=0,
        num_priv_explicit=0,
        num_priv_latent=8,
        history_latent_dim=4,
        num_hist=20,
        critic_use_latent=True,
        encode_scan_for_critic=False,
        critic_num_prop=13,
        critic_num_scan=0,
        critic_num_priv_explicit=0,
        critic_num_priv_latent=0,
        critic_num_hist=0,
        actor={
            "class_name": "Actor",
            "num_prop": 5,
            "num_scan": 0,
            "num_priv_explicit": 0,
            "num_priv_latent": 8,
            "history_latent_dim": 4,
            "history_reconstruction_dim": 0,
            "velocity_estimation_dim": 3,
            "velocity_estimator_channel_size": 32,
            "num_hist": 20,
            "normalize_latent": True,
            "state_history_encoder": {
                "class_name": "TCNHistoryEncoder",
                "num_prop": 5,
                "num_hist": 20,
                "history_latent_dim": 4,
                "channel_size": 32,
            },
        },
    )


def test_cts_student_inference_accepts_student_only_observations() -> None:
    policy = _make_cts_policy()
    batch_size = 4
    full_obs = torch.randn(batch_size, 5 + 8 + 20 * 5)
    student_obs = torch.cat([full_obs[:, :5], full_obs[:, -20 * 5 :]], dim=1)

    actions_from_full = policy.act_student_inference(full_obs)
    actions_from_student = policy.act_student_inference(student_obs)

    assert actions_from_full.shape == (batch_size, 3)
    assert actions_from_student.shape == (batch_size, 3)
    assert torch.allclose(actions_from_full, actions_from_student, atol=1e-6)


def test_cts_latents_are_unit_normalized() -> None:
    policy = _make_cts_policy()
    full_obs = torch.randn(6, 5 + 8 + 20 * 5)

    teacher_latent = policy.teacher_latent(full_obs)
    student_latent = policy.student_latent(full_obs)

    teacher_norm = torch.linalg.norm(teacher_latent, dim=1)
    student_norm = torch.linalg.norm(student_latent, dim=1)

    assert teacher_latent.shape == (6, 4)
    assert student_latent.shape == (6, 4)
    assert torch.allclose(teacher_norm, torch.ones_like(teacher_norm), atol=1e-5, rtol=1e-4)
    assert torch.allclose(student_norm, torch.ones_like(student_norm), atol=1e-5, rtol=1e-4)


def test_cts_algorithm_runs_joint_rl_and_reconstruction_update() -> None:
    policy = _make_cts_policy()
    algorithm = CTS(
        policy=policy,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=3.0e-4,
        reconstruction_learning_rate=1.0e-3,
        num_reconstruction_epochs=1,
        velocity_estimation_loss_coef=0.1,
        roa_teacher_reg_coef_end=0.05,
        roa_teacher_reg_warmup_updates=0,
        roa_teacher_reg_ramp_updates=1,
        student_group_ratio=0.5,
        constraint_limits=[1.0],
    )
    algorithm.init_storage("cts", 2, 2, (5 + 8 + 20 * 5,), (13,), (3,))

    obs = torch.randn(2, 5 + 8 + 20 * 5)
    critic_obs = torch.randn(2, 13)
    actor_is_student = torch.tensor([False, True], dtype=torch.bool)

    for step in range(2):
        _ = algorithm.act(obs + step * 0.1, critic_obs + step * 0.1, actor_is_student)
        algorithm.process_env_step(
            obs,
            rewards=torch.full((2,), 1.0 + 0.1 * step),
            dones=torch.zeros(2, dtype=torch.uint8),
            infos={},
        )

    algorithm.compute_returns(obs, critic_obs, actor_is_student)
    loss_dict = algorithm.update()

    assert set(loss_dict).issuperset(
        {
            "surrogate",
            "value_function",
            "velocity_estimation",
            "teacher_latent_reg",
            "teacher_latent_reg_weighted",
        }
    )
    for value in loss_dict.values():
        assert math.isfinite(float(value))
    assert math.isfinite(float(algorithm.train_metrics["teacher_latent_reg_coef"]))
    assert algorithm.train_metrics["teacher_latent_reg_coef"] > 0.0
