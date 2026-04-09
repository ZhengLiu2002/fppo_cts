from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.rsl_rl.modules.actor_critic_with_encoder import ActorCriticRMA


def _make_student_policy() -> ActorCriticRMA:
    return ActorCriticRMA(
        num_critic_obs=185,
        num_actions=12,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        cost_critic_hidden_dims=[64, 32],
        activation="elu",
        init_noise_std=1.0,
        tanh_encoder_output=False,
        scan_encoder_dims=[],
        priv_encoder_dims=[],
        num_prop=45,
        num_scan=0,
        num_priv_explicit=0,
        num_priv_latent=0,
        history_latent_dim=32,
        num_hist=20,
        encode_scan_for_critic=False,
        critic_num_prop=48,
        critic_num_scan=132,
        critic_num_priv_explicit=5,
        critic_num_priv_latent=0,
        critic_num_hist=0,
        actor={
            "class_name": "Actor",
            "num_prop": 45,
            "num_scan": 0,
            "num_priv_explicit": 0,
            "num_priv_latent": 0,
            "history_latent_dim": 32,
            "history_reconstruction_dim": 140,
            "num_hist": 20,
            "state_history_encoder": {
                "class_name": "TCNHistoryEncoder",
                "num_prop": 45,
                "num_hist": 20,
                "history_latent_dim": 32,
                "channel_size": 32,
            },
        },
    )


def test_student_history_encoder_outputs_32d_latent() -> None:
    policy = _make_student_policy()
    obs = torch.randn(4, 45 + 20 * 45)

    latent = policy.actor.infer_hist_latent(obs)
    actions = policy.act_inference(obs, hist_encoding=True)

    assert latent.shape == (4, 32)
    assert actions.shape == (4, 12)


def test_student_history_reconstruction_targets_hidden_privileged_terms() -> None:
    policy = _make_student_policy()
    obs = torch.randn(3, 45 + 20 * 45)
    critic_obs = torch.randn(3, 48 + 132 + 5)

    prediction, target = policy.history_reconstruction(obs, critic_obs)
    expected = torch.cat([critic_obs[:, :3], critic_obs[:, 48:180], critic_obs[:, 180:185]], dim=1)

    assert prediction is not None
    assert target is not None
    assert prediction.shape == (3, 140)
    assert target.shape == (3, 140)
    assert torch.allclose(target, expected)
