from __future__ import annotations

import argparse
import sys
import types

import torch
import torch.nn as nn

from scripts.rsl_rl import cli_args
from scripts.rsl_rl.algorithms.dagger import DAgger
from scripts.rsl_rl.algorithms.registry import get_algorithm_class, get_algorithm_spec


class _SimpleActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_hist = 1
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, obs: torch.Tensor, hist_encoding: bool, scandots_latent=None) -> torch.Tensor:
        return self.linear(obs)


class _SimplePolicy(nn.Module):
    is_recurrent = False

    def __init__(self) -> None:
        super().__init__()
        self.actor = _SimpleActor()

    def act(self, observations: torch.Tensor, hist_encoding: bool = False, **kwargs) -> torch.Tensor:
        return self.act_inference(observations, hist_encoding=hist_encoding, **kwargs)

    def act_inference(
        self, observations: torch.Tensor, hist_encoding: bool = False, **kwargs
    ) -> torch.Tensor:
        return self.actor(observations, hist_encoding)

    def reset(self, dones=None, hidden_states=None) -> None:
        return None

    def history_reconstruction(
        self, observations: torch.Tensor, privileged_observations: torch.Tensor
    ) -> tuple[None, None]:
        return None, None


class _NaNReconstructionPolicy(_SimplePolicy):
    def history_reconstruction(
        self, observations: torch.Tensor, privileged_observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = torch.full_like(observations, torch.nan)
        target = torch.zeros_like(privileged_observations)
        return prediction, target


class _TeacherPolicy(nn.Module):
    def act_inference(
        self, observations: torch.Tensor, hist_encoding: bool = False, **kwargs
    ) -> torch.Tensor:
        return observations.clone()


def _make_algorithm(
    buffer_size: int = 4,
    *,
    batch_size: int = 2,
    num_epochs: int = 2,
    batches_per_update: int = 0,
    min_buffer_size: int = 1,
    policy: nn.Module | None = None,
    reconstruction_loss_coef: float = 0.0,
) -> DAgger:
    alg = DAgger(
        policy=policy or _SimplePolicy(),
        teacher_policy=_TeacherPolicy(),
        num_learning_epochs=num_epochs,
        learning_rate=0.2,
        reconstruction_loss_coef=reconstruction_loss_coef,
        dagger_buffer_size=buffer_size,
        dagger_batch_size=batch_size,
        dagger_min_buffer_size=min_buffer_size,
        dagger_batches_per_update=batches_per_update,
        deterministic_rollout=True,
        device="cpu",
    )
    alg.init_storage(
        "dagger",
        num_envs=1,
        num_transitions_per_env=2,
        student_obs_shape=(1,),
        teacher_obs_shape=(1,),
        actions_shape=(1,),
    )
    return alg


def _add_step(alg: DAgger, value: float) -> None:
    obs = torch.tensor([[value]], dtype=torch.float32)
    alg.act(obs, obs)
    alg.process_env_step(
        obs,
        rewards=torch.zeros(1, dtype=torch.float32),
        dones=torch.zeros(1, dtype=torch.uint8),
        infos={},
    )


def test_dagger_replay_buffer_aggregates_across_rollouts() -> None:
    alg = _make_algorithm(buffer_size=4)

    _add_step(alg, 0.0)
    _add_step(alg, 1.0)
    first_loss = alg.update()
    assert first_loss["dagger_buffer_size"] == 2.0

    _add_step(alg, 2.0)
    _add_step(alg, 3.0)
    second_loss = alg.update()

    assert second_loss["dagger_buffer_size"] == 4.0
    assert alg.replay_buffer is not None
    stored = torch.sort(alg.replay_buffer.observations[: len(alg.replay_buffer), 0]).values
    assert torch.allclose(stored, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_dagger_replay_buffer_respects_capacity() -> None:
    alg = _make_algorithm(buffer_size=3)

    for value in (0.0, 1.0):
        _add_step(alg, value)
    alg.update()

    for value in (2.0, 3.0):
        _add_step(alg, value)
    alg.update()

    assert alg.replay_buffer is not None
    assert len(alg.replay_buffer) == 3
    stored = torch.sort(alg.replay_buffer.observations[: len(alg.replay_buffer), 0]).values
    assert torch.allclose(stored, torch.tensor([1.0, 2.0, 3.0]))


def test_dagger_registry_entry_resolves_to_dagger_training_type() -> None:
    assert get_algorithm_class("dagger") is DAgger
    assert get_algorithm_spec("dagger").training_type == "dagger"


def test_dagger_update_respects_fixed_batch_budget() -> None:
    alg = _make_algorithm(buffer_size=8, batch_size=1, num_epochs=3, batches_per_update=5)

    for value in (0.0, 1.0):
        _add_step(alg, value)
    first_loss = alg.update()
    assert first_loss["dagger_num_batches"] == 5.0
    assert first_loss["dagger_batch_budget"] == 5.0

    for value in (2.0, 3.0):
        _add_step(alg, value)
    second_loss = alg.update()
    assert second_loss["dagger_num_batches"] == 5.0
    assert second_loss["dagger_batch_budget"] == 5.0


def test_dagger_sanitizes_nan_reconstruction_loss() -> None:
    alg = _make_algorithm(
        buffer_size=4,
        batch_size=2,
        num_epochs=1,
        policy=_NaNReconstructionPolicy(),
        reconstruction_loss_coef=0.25,
    )

    for value in (0.0, 1.0):
        _add_step(alg, value)
    loss = alg.update()

    assert torch.isfinite(torch.tensor(loss["behavior"]))
    assert torch.isfinite(torch.tensor(loss["reconstruction"]))
    assert loss["reconstruction"] == 0.0


class _FakeFppoAlgoCfg:
    def __init__(self) -> None:
        self.class_name = "FPPO"
        self.learning_rate = 3.0e-4
        self.dagger_update_freq = 1
        self.reconstruction_loss_coef = 0.25

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "learning_rate": self.learning_rate,
            "dagger_update_freq": self.dagger_update_freq,
            "reconstruction_loss_coef": self.reconstruction_loss_coef,
        }


class _FakeDaggerAlgoCfg(_FakeFppoAlgoCfg):
    def __init__(self) -> None:
        super().__init__()
        self.class_name = "DAgger"
        self.dagger_update_freq = 10
        self.dagger_buffer_size = 1_048_576
        self.dagger_batch_size = 16_384
        self.dagger_min_buffer_size = 262_144
        self.dagger_batches_per_update = 32

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "dagger_buffer_size": self.dagger_buffer_size,
                "dagger_batch_size": self.dagger_batch_size,
                "dagger_min_buffer_size": self.dagger_min_buffer_size,
                "dagger_batches_per_update": self.dagger_batches_per_update,
            }
        )
        return data


class GalileoStudentRunnerCfg:
    def __init__(self) -> None:
        self.algorithm = _FakeFppoAlgoCfg()


def _install_fake_galileo_builder(monkeypatch) -> None:
    def _package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        return module

    for name in (
        "crl_tasks",
        "crl_tasks.tasks",
        "crl_tasks.tasks.galileo",
        "crl_tasks.tasks.galileo.config",
        "crl_tasks.tasks.galileo.config.agents",
    ):
        monkeypatch.setitem(sys.modules, name, _package(name))

    shared_module = types.ModuleType("crl_tasks.tasks.galileo.config.agents._shared_runner_cfg")

    def _build_algorithm_cfg(role: str, algo_key: str | None = None):
        assert role == "student"
        if algo_key == "dagger":
            return _FakeDaggerAlgoCfg()
        return _FakeFppoAlgoCfg()

    shared_module.build_algorithm_cfg = _build_algorithm_cfg
    monkeypatch.setitem(
        sys.modules,
        "crl_tasks.tasks.galileo.config.agents._shared_runner_cfg",
        shared_module,
    )


def test_cli_algo_override_rebuilds_typed_dagger_cfg(monkeypatch) -> None:
    _install_fake_galileo_builder(monkeypatch)
    agent_cfg = GalileoStudentRunnerCfg()
    args_cli = argparse.Namespace(algo="dagger")

    cli_args.apply_rsl_rl_algo_override(agent_cfg, args_cli, apply_profile=True)

    assert isinstance(agent_cfg.algorithm, _FakeDaggerAlgoCfg)
    assert agent_cfg.algorithm.class_name == "DAgger"
    assert agent_cfg.algorithm.dagger_update_freq == 10
    assert agent_cfg.algorithm.dagger_buffer_size == 1_048_576


def test_cli_algo_reapply_preserves_existing_common_values(monkeypatch) -> None:
    _install_fake_galileo_builder(monkeypatch)
    agent_cfg = GalileoStudentRunnerCfg()
    agent_cfg.algorithm.learning_rate = 1.0e-4
    agent_cfg.algorithm.dagger_update_freq = 7
    args_cli = argparse.Namespace(algo="dagger")

    cli_args.apply_rsl_rl_algo_override(agent_cfg, args_cli, apply_profile=False)

    assert isinstance(agent_cfg.algorithm, _FakeDaggerAlgoCfg)
    assert agent_cfg.algorithm.class_name == "DAgger"
    assert agent_cfg.algorithm.learning_rate == 1.0e-4
    assert agent_cfg.algorithm.dagger_update_freq == 7
    assert agent_cfg.algorithm.dagger_buffer_size == 1_048_576
