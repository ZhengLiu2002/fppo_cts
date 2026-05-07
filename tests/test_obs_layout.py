from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from scripts.rsl_rl.obs_layout import (
    add_policy_layout_metadata,
    extract_observation_layout,
    validate_policy_layout,
)


def _term(scale=1.0, params=None):
    return SimpleNamespace(func=lambda env: None, scale=scale, params=params or {})


def _env_stub():
    obs_mgr = SimpleNamespace(
        group_obs_dim={"policy": (15,), "critic": (8,)},
        group_obs_concatenate={"policy": True, "critic": True},
        active_terms={"policy": ["prop", "history"], "critic": ["priv"]},
        group_obs_term_dim={"policy": [(5,), (10,)], "critic": [(8,)]},
        _group_obs_term_names={"policy": ["prop", "history"], "critic": ["priv"]},
        _group_obs_term_cfgs={
            "policy": [_term(scale=0.5), _term(params={"scale": 1.0})],
            "critic": [_term(scale=1.0)],
        },
    )
    cfg = SimpleNamespace(config_summary=SimpleNamespace(__name__="Summary"))
    return SimpleNamespace(unwrapped=SimpleNamespace(observation_manager=obs_mgr, cfg=cfg))


def test_extract_observation_layout_includes_stable_hash_and_term_dims() -> None:
    layout = extract_observation_layout(_env_stub())

    assert layout["layout_hash"]
    assert layout["groups"]["policy"]["dim"] == 15
    assert layout["groups"]["policy"]["terms"][0]["name"] == "prop"
    assert layout["groups"]["policy"]["terms"][1]["dim"] == 10


def test_add_policy_layout_metadata_and_validate_accept_student_export_dim() -> None:
    env = _env_stub()
    actor_critic = SimpleNamespace(
        actor=SimpleNamespace(
            in_features=20,
            student_in_features=15,
            num_prop=5,
            num_hist=2,
            num_scan=0,
            num_priv_explicit=0,
            num_priv_latent=5,
        )
    )
    policy_cfg = {
        "input_names": ["actor_obs", "proprio_history"],
        "input_obs_names_map": {"actor_obs": ["prop"], "proprio_history": ["prop"]},
        "input_obs_scales_map": {"actor_obs": {"prop": 1.0}, "proprio_history": {"prop": 1.0}},
        "input_obs_size_map": {"actor_obs": 5, "proprio_history": 5},
        "obs_history_length": {"actor_obs": 1, "proprio_history": 2},
        "export_input_order": ["actor_obs", "proprio_history"],
        "joint_names": ["j0"],
    }

    enriched = add_policy_layout_metadata(policy_cfg, env=env, actor_critic=actor_critic)

    assert enriched["layout_hash"]
    assert enriched["observation_layout_hash"] == enriched["observation_layout"]["layout_hash"]
    assert enriched["export_input_dims"] == {"actor_obs": 5, "proprio_history": 10}
    assert validate_policy_layout(enriched, env=env, actor_critic=actor_critic, strict=False) == []


def test_validate_policy_layout_reports_mismatch() -> None:
    policy_cfg = {
        "input_names": ["actor_obs"],
        "input_obs_size_map": {"actor_obs": 5},
        "obs_history_length": {"actor_obs": 1},
        "export_input_dims": {"actor_obs": 5},
    }
    actor_critic = SimpleNamespace(actor=SimpleNamespace(in_features=7))

    issues = validate_policy_layout(policy_cfg, actor_critic=actor_critic, strict=False)

    assert issues
    assert "does not match actor dims" in issues[0]
