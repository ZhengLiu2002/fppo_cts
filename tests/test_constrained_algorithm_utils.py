from __future__ import annotations

import math

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


def test_fppo_curriculum_perf_gate_blocks_tightening_when_learning_collapses() -> None:
    fppo = _make_curriculum_fppo()
    start_limits = fppo.constraint_limits_current.clone()

    fppo._update_constraint_limit_curriculum(
        torch.tensor([0.10, 0.12, 0.02]),
        reward_return_mean=10.0,
        predictor_stop_rate=0.0,
    )
    second_metrics = fppo._update_constraint_limit_curriculum(
        torch.tensor([0.10, 0.12, 0.02]),
        reward_return_mean=0.0,
        predictor_stop_rate=1.0,
    )

    assert second_metrics["curriculum_gate_ready"] == 1.0
    assert second_metrics["curriculum_perf_gate_ready"] == 0.0
    assert second_metrics["curriculum_tighten_triggered"] == 0.0
    assert torch.allclose(fppo.constraint_limits_current, start_limits, atol=1e-6)


def test_fppo_corrector_step_adaptation_does_not_change_predictor_lr() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.step_size_adaptive = True
    fppo.step_size = 2.0e-4
    fppo.predictor_lr = 1.0e-3
    fppo.learning_rate = 1.0e-3
    fppo._corrector_step_up = 1.01
    fppo._corrector_step_down = 0.85
    fppo._corrector_step_min = 1.0e-4
    fppo._corrector_step_max = 8.0e-4
    fppo._corrector_target_accept_rate = 0.7
    fppo._corrector_margin_threshold = 0.01

    fppo._adapt_corrector_step_size(accept_rate=0.0, mean_cost_margin=-0.1)

    assert abs(fppo.step_size - 1.7e-4) < 1.0e-12
    assert fppo.predictor_lr == 1.0e-3
    assert fppo.learning_rate == 1.0e-3


def test_fppo_projection_rejects_nonfinite_kl() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.device = "cpu"
    fppo.delta_kl = 0.01
    fppo.desired_kl = 0.01
    fppo.max_corrections = 3
    fppo.backtrack_coeff = 0.5

    theta_anchor = torch.tensor([0.0, 0.0])
    theta_predictor = torch.tensor([1.0, -1.0])
    state = {"theta": theta_anchor.clone()}

    def _project_safe_set(theta_prime, theta_ref, a_mat, b_budget):
        return theta_prime, 0

    def _set_actor_param_vector(theta):
        state["theta"] = theta.detach().clone()

    def _actor_param_vector():
        return state["theta"]

    def _evaluate_candidate(_projection_batch):
        return torch.tensor(float("nan"))

    fppo._project_safe_set = _project_safe_set
    fppo._set_actor_param_vector = _set_actor_param_vector
    fppo._actor_param_vector = _actor_param_vector
    fppo._evaluate_candidate = _evaluate_candidate

    metrics = fppo._run_projection_corrector(
        projection_batch={"d_tight": torch.zeros(0), "j_cost": torch.zeros(0)},
        theta_anchor=theta_anchor,
        theta_predictor=theta_predictor,
        a_mat=torch.zeros(theta_anchor.numel(), 0),
    )

    assert metrics["accept_rate"] == 0.0
    assert metrics["infeasible_batch_rate"] == 1.0
    assert metrics["effective_step_ratio"] == 0.0
    assert torch.allclose(state["theta"], theta_anchor)


def test_fppo_projection_corrector_respects_step_size_budget() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.device = "cpu"
    fppo.delta_kl = 1.0
    fppo.desired_kl = 1.0
    fppo.max_corrections = 1
    fppo.backtrack_coeff = 0.5
    fppo.step_size = 0.2

    theta_anchor = torch.tensor([0.0, 0.0])
    theta_predictor = torch.tensor([3.0, 4.0])
    state = {"theta": theta_anchor.clone()}

    def _project_safe_set(theta_prime, theta_ref, a_mat, b_budget):
        return theta_prime, 0

    def _set_actor_param_vector(theta):
        state["theta"] = theta.detach().clone()

    def _actor_param_vector():
        return state["theta"]

    def _evaluate_candidate(_projection_batch):
        return torch.tensor(0.0)

    fppo._project_safe_set = _project_safe_set
    fppo._set_actor_param_vector = _set_actor_param_vector
    fppo._actor_param_vector = _actor_param_vector
    fppo._evaluate_candidate = _evaluate_candidate

    metrics = fppo._run_projection_corrector(
        projection_batch={"d_tight": torch.zeros(0), "j_cost": torch.zeros(0)},
        theta_anchor=theta_anchor,
        theta_predictor=theta_predictor,
        a_mat=torch.zeros(theta_anchor.numel(), 0),
    )

    expected_step = fppo.step_size * math.sqrt(theta_anchor.numel())
    expected_theta = theta_anchor + theta_predictor / torch.norm(theta_predictor) * expected_step
    assert torch.allclose(state["theta"], expected_theta, atol=1e-6)
    assert abs(metrics["effective_step_size"] - expected_step) < 1.0e-6
    assert abs(metrics["effective_step_ratio"] - (expected_step / torch.norm(theta_predictor).item())) < 1.0e-6


def test_fppo_predictor_uses_separate_kl_limits() -> None:
    class _DummyPolicy:
        def __init__(self, param: torch.nn.Parameter):
            self._param = param
            self.action_mean = torch.zeros(1, 1)
            self.action_std = torch.ones(1, 1)
            self.entropy = torch.ones(1)

        def act(self, obs_batch, masks=None, hidden_states=None):
            self.action_mean = obs_batch + self._param.view(1, 1)
            self.action_std = torch.ones_like(obs_batch)
            self.entropy = torch.ones(obs_batch.shape[0], device=obs_batch.device)

        def get_actions_log_prob(self, actions_batch):
            return self._param.expand(actions_batch.shape[0])

    batch = (
        torch.zeros(1, 1),
        None,
        torch.zeros(1, 1),
        None,
        torch.ones(1),
        None,
        None,
        None,
        None,
        torch.zeros(1),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        (None, None, None),
        None,
    )

    fppo = FPPO.__new__(FPPO)
    actor_param = torch.nn.Parameter(torch.tensor(0.0))
    fppo.policy = _DummyPolicy(actor_param)
    fppo.num_learning_epochs = 1
    fppo.num_mini_batches = 4
    fppo.predictor_lr = 1.0e-3
    fppo._predictor_lr_min = 1.0e-4
    fppo._predictor_lr_max = 3.0e-3
    fppo.predictor_desired_kl = 0.01
    fppo.predictor_kl_hard_limit = 0.02
    fppo.schedule = "fixed"
    fppo.desired_kl = 0.004
    fppo.normalize_advantage_per_mini_batch = False
    fppo.use_clipped_surrogate = False
    fppo.entropy_coef = 0.0
    fppo.is_multi_gpu = False
    fppo.gpu_global_rank = 0
    fppo.max_grad_norm = 1.0
    fppo._actor_params = [actor_param]
    fppo.actor_optimizer = torch.optim.SGD(fppo._actor_params, lr=fppo.predictor_lr)
    fppo.kl_hard_abs = 0.05
    fppo.kl_hard_ratio = 4.0

    kl_values = iter([torch.tensor(0.005), torch.tensor(0.005), torch.tensor(0.005), torch.tensor(0.005)])
    fppo._mini_batch_generator = lambda: iter([batch, batch, batch, batch])
    fppo._sanitize_tensor = lambda tensor, **_kwargs: tensor
    fppo._all_reduce_mean = lambda tensor: tensor
    fppo._safe_ratio = lambda new, old: torch.exp(new - old)
    fppo._safe_kl = lambda *args, **kwargs: next(kl_values).expand(1)
    fppo._set_predictor_learning_rate = FPPO._set_predictor_learning_rate.__get__(fppo, FPPO)

    metrics = fppo._run_ppo_predictor()

    assert metrics["updates"] == 4.0
    assert abs(metrics["update_ratio"] - 1.0) < 1.0e-12
    assert abs(metrics["stop_rate"] - 0.0) < 1.0e-12


def test_fppo_predictor_kl_skip_uses_continue_not_break() -> None:
    class _DummyPolicy:
        def __init__(self, param: torch.nn.Parameter):
            self._param = param
            self.action_mean = torch.zeros(1, 1)
            self.action_std = torch.ones(1, 1)
            self.entropy = torch.ones(1)

        def act(self, obs_batch, masks=None, hidden_states=None):
            self.action_mean = obs_batch + self._param.view(1, 1)
            self.action_std = torch.ones_like(obs_batch)
            self.entropy = torch.ones(obs_batch.shape[0], device=obs_batch.device)

        def get_actions_log_prob(self, actions_batch):
            return self._param.expand(actions_batch.shape[0])

    batch = (
        torch.zeros(1, 1),
        None,
        torch.zeros(1, 1),
        None,
        torch.ones(1),
        None,
        None,
        None,
        None,
        torch.zeros(1),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        (None, None, None),
        None,
    )

    fppo = FPPO.__new__(FPPO)
    actor_param = torch.nn.Parameter(torch.tensor(0.0))
    fppo.policy = _DummyPolicy(actor_param)
    fppo.num_learning_epochs = 1
    fppo.num_mini_batches = 4
    fppo.predictor_lr = 1.0e-3
    fppo._predictor_lr_min = 1.0e-4
    fppo._predictor_lr_max = 3.0e-3
    fppo.predictor_desired_kl = 0.01
    fppo.predictor_kl_hard_limit = 0.015
    fppo.schedule = "fixed"
    fppo.desired_kl = 0.004
    fppo.normalize_advantage_per_mini_batch = False
    fppo.use_clipped_surrogate = False
    fppo.entropy_coef = 0.0
    fppo.is_multi_gpu = False
    fppo.gpu_global_rank = 0
    fppo.max_grad_norm = 1.0
    fppo._actor_params = [actor_param]
    fppo.actor_optimizer = torch.optim.SGD(fppo._actor_params, lr=fppo.predictor_lr)
    fppo.kl_hard_abs = 0.05
    fppo.kl_hard_ratio = 4.0

    kl_values = iter([torch.tensor(0.0), torch.tensor(0.02), torch.tensor(0.0), torch.tensor(0.0)])
    fppo._mini_batch_generator = lambda: iter([batch, batch, batch, batch])
    fppo._sanitize_tensor = lambda tensor, **_kwargs: tensor
    fppo._all_reduce_mean = lambda tensor: tensor
    fppo._safe_ratio = lambda new, old: torch.exp(new - old)
    fppo._safe_kl = lambda *args, **kwargs: next(kl_values).expand(1)
    fppo._set_predictor_learning_rate = FPPO._set_predictor_learning_rate.__get__(fppo, FPPO)

    metrics = fppo._run_ppo_predictor()

    assert metrics["updates"] == 3.0
    assert abs(metrics["update_ratio"] - 0.75) < 1.0e-12
    assert abs(metrics["stop_rate"] - 0.25) < 1.0e-12
