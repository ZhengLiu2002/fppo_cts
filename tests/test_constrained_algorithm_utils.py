from __future__ import annotations

import torch

from scripts.rsl_rl.algorithms.omnisafe_utils import (
    Lagrange,
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_parameters,
)
from scripts.rsl_rl.algorithms.fppo import FPPO
from scripts.rsl_rl.algorithms.registry import get_algorithm_spec


def test_conjugate_gradients_matches_linear_solve() -> None:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    vector = torch.tensor([1.0, 2.0])

    solution = conjugate_gradients(lambda tensor: matrix @ tensor, vector, num_steps=8)
    expected = torch.linalg.solve(matrix, vector)

    assert torch.allclose(solution, expected, atol=1e-5)


def test_flat_parameter_round_trip() -> None:
    layer = torch.nn.Linear(2, 2)
    params = list(layer.parameters())
    flat_params = get_flat_params_from(params)
    new_values = torch.arange(flat_params.numel(), dtype=flat_params.dtype)

    set_param_values_to_parameters(params, new_values)
    assert torch.allclose(get_flat_params_from(params), new_values)

    layer(torch.ones(1, 2)).sum().backward()
    flat_grads = get_flat_gradients_from(params)
    assert flat_grads.shape == new_values.shape


def test_lagrange_state_round_trip() -> None:
    lagrange = Lagrange(
        cost_limit=1.0,
        lagrangian_multiplier_init=0.0,
        lambda_lr=0.5,
        lambda_optimizer="SGD",
    )
    lagrange.update_lagrange_multiplier(2.0)
    assert lagrange.lagrangian_multiplier.item() > 0.0

    state_dict = lagrange.state_dict()
    restored = Lagrange(
        cost_limit=1.0,
        lagrangian_multiplier_init=0.0,
        lambda_lr=0.5,
        lambda_optimizer="SGD",
    )
    restored.load_state_dict(state_dict)

    assert torch.allclose(restored.lagrangian_multiplier, lagrange.lagrangian_multiplier)


def test_registry_accepts_omnisafe_style_aliases() -> None:
    assert get_algorithm_spec("ppolag").class_name == "PPOLagrange"
    assert get_algorithm_spec("ppo_lag").class_name == "PPOLagrange"
    assert get_algorithm_spec("focops").class_name == "FOCOPS"


def _make_curriculum_fppo() -> FPPO:
    fppo = FPPO.__new__(FPPO)
    fppo.device = "cpu"
    fppo.cost_limit = 0.0
    fppo.constraint_limits = torch.tensor([0.15, 0.18, 0.05])
    fppo.constraint_limits_start = torch.tensor([0.225, 0.27, 0.075])
    fppo.constraint_limits_final = torch.tensor([0.15, 0.18, 0.05])
    fppo.constraint_limits_current = fppo.constraint_limits_start.clone()
    fppo.adaptive_constraint_curriculum = True
    fppo.constraint_names = ["prob_joint_pos", "prob_com_height", "prob_gait_pattern"]
    fppo.constraint_curriculum_names = ["prob_joint_pos", "prob_com_height"]
    fppo.constraint_curriculum_ema_decay = 0.95
    fppo.constraint_curriculum_check_interval = 2
    fppo.constraint_curriculum_alpha = 0.8
    fppo.constraint_curriculum_shrink = 0.97
    fppo._constraint_curriculum_ema = None
    fppo._constraint_curriculum_updates = 0
    fppo._constraint_curriculum_tighten_count = 0
    return fppo


def test_fppo_curriculum_tightens_when_gate_constraints_are_stable() -> None:
    fppo = _make_curriculum_fppo()

    first_metrics = fppo._update_constraint_limit_curriculum(torch.tensor([0.10, 0.12, 0.02]))
    second_metrics = fppo._update_constraint_limit_curriculum(torch.tensor([0.10, 0.12, 0.02]))

    expected_limits = torch.tensor([0.225, 0.27, 0.075]) * 0.97
    assert first_metrics["curriculum_tighten_triggered"] == 0.0
    assert second_metrics["curriculum_tighten_triggered"] == 1.0
    assert second_metrics["curriculum_tighten_count"] == 1.0
    assert torch.allclose(fppo.constraint_limits_current, expected_limits, atol=1e-6)


def test_fppo_curriculum_holds_limits_when_gate_constraint_is_not_ready() -> None:
    fppo = _make_curriculum_fppo()
    start_limits = fppo.constraint_limits_current.clone()

    fppo._update_constraint_limit_curriculum(torch.tensor([0.10, 0.24, 0.02]))
    second_metrics = fppo._update_constraint_limit_curriculum(torch.tensor([0.10, 0.24, 0.02]))

    assert second_metrics["curriculum_tighten_triggered"] == 0.0
    assert second_metrics["curriculum_gate_ready"] == 0.0
    assert torch.allclose(fppo.constraint_limits_current, start_limits, atol=1e-6)
