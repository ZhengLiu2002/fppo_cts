from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from scripts.rsl_rl.algorithms.fppo import FPPO
from scripts.rsl_rl.algorithms.omnisafe_utils import (
    Lagrange,
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_parameters,
)
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


def test_fppo_constraint_cost_stats_use_env_axis() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.device = "cpu"
    fppo._all_reduce_mean = lambda tensor: tensor
    fppo._sanitize_tensor = lambda tensor, **_kwargs: tensor

    env_costs = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    j_cost, sigma_j = FPPO._estimate_constraint_cost_stats(fppo, env_costs)

    expected_mean = torch.tensor([2.0, 4.0])
    expected_sigma = env_costs.std(dim=0, unbiased=False) / math.sqrt(2.0)
    assert torch.allclose(j_cost, expected_mean)
    assert torch.allclose(sigma_j, expected_sigma)


def test_fppo_robust_margin_matches_paper_formula() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.desired_kl = 0.02
    fppo.cost_confidence = 1.5
    fppo.gradient_confidence = 2.0
    fppo.curvature_proxy = 0.25
    fppo.max_margin_abs = None
    fppo.max_margin_ratio = None
    fppo.projection_eps = 1.0e-8

    sigma_j = torch.tensor([0.20, 0.10])
    sigma_a = torch.tensor([0.05, 0.08])

    margin = FPPO._compute_robust_margin(fppo, sigma_j, sigma_a)

    radius = math.sqrt(2.0 * fppo.desired_kl)
    expected = (
        fppo.cost_confidence * sigma_j
        + fppo.gradient_confidence * sigma_a * radius
        + 0.5 * fppo.curvature_proxy * radius * radius
    )
    assert torch.allclose(margin, expected, atol=1e-6)


def test_fppo_robust_margin_respects_caps() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.desired_kl = 0.02
    fppo.cost_confidence = 10.0
    fppo.gradient_confidence = 10.0
    fppo.curvature_proxy = 0.0
    fppo.max_margin_abs = 0.2
    fppo.max_margin_ratio = 0.5
    fppo.projection_eps = 1.0e-8

    margin = FPPO._compute_robust_margin(
        fppo,
        sigma_j=torch.tensor([1.0]),
        sigma_a=torch.tensor([1.0]),
        j_cost=torch.tensor([0.1]),
        d_limits=torch.tensor([0.3]),
    )

    assert torch.allclose(margin, torch.tensor([0.1]), atol=1e-6)


def test_fppo_metric_projection_matches_single_constraint_closed_form() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.projection_eps = 1.0e-8
    fppo.exact_qp_max_constraints = 8
    fppo.qp_max_iters = 64
    fppo.qp_tol = 1.0e-8

    delta_nom = torch.tensor([3.0, 1.0])
    a_mat = torch.tensor([[2.0], [1.0]])
    budgets = torch.tensor([1.0])
    fisher_diag = torch.tensor([4.0, 1.0])

    delta_proj, active, predicted_violation, qp_condition = FPPO._project_step(
        fppo,
        delta_nom=delta_nom,
        a_mat=a_mat,
        budgets=budgets,
        fisher_diag=fisher_diag,
    )

    inv_metric = 1.0 / fisher_diag
    direction = inv_metric * a_mat[:, 0]
    violation = torch.dot(a_mat[:, 0], delta_nom) - budgets[0]
    coeff = torch.clamp(violation / torch.dot(a_mat[:, 0], direction), min=0.0)
    expected = delta_nom - coeff * direction

    assert active == 1
    assert abs(predicted_violation - violation.item()) < 1.0e-6
    assert qp_condition == 1.0
    assert torch.allclose(delta_proj, expected, atol=1.0e-6)


def test_fppo_small_qp_exact_solver_handles_multiple_constraints() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.projection_eps = 1.0e-8
    fppo.exact_qp_max_constraints = 8
    fppo.qp_max_iters = 128
    fppo.qp_tol = 1.0e-10

    delta_nom = torch.tensor([3.0, 4.0])
    a_mat = torch.eye(2)
    budgets = torch.tensor([1.0, 2.0])
    fisher_diag = torch.tensor([2.0, 5.0])

    delta_proj, active, predicted_violation, qp_condition = FPPO._project_step(
        fppo,
        delta_nom=delta_nom,
        a_mat=a_mat,
        budgets=budgets,
        fisher_diag=fisher_diag,
    )

    assert active == 2
    assert abs(predicted_violation - 2.0) < 1.0e-6
    assert qp_condition >= 1.0
    assert torch.allclose(delta_proj, torch.tensor([1.0, 2.0]), atol=1.0e-6)


def test_fppo_projection_radius_cap_uses_metric_norm() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.desired_kl = 0.02
    fppo.projection_radius_cap = 1.0
    fppo.projection_radius_mode = "kl"
    fppo.projection_eps = 1.0e-8

    delta_proj = torch.tensor([1.0, 1.0])
    fisher_diag = torch.tensor([4.0, 1.0])

    capped, scale = FPPO._apply_projection_radius_cap(fppo, delta_proj, fisher_diag)
    expected_radius = math.sqrt(2.0 * fppo.desired_kl)
    metric_norm = math.sqrt(float((delta_proj.pow(2) * fisher_diag).sum().item()))
    expected_scale = expected_radius / metric_norm

    assert abs(scale - expected_scale) < 1.0e-6
    assert torch.allclose(capped, delta_proj * expected_scale, atol=1.0e-6)


def test_fppo_gradient_uncertainty_reuses_cache() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.device = "cpu"
    fppo.gradient_uncertainty_mode = "shards"
    fppo.gradient_uncertainty_shards = 4
    fppo.uncertainty_update_interval = 4
    fppo.uncertainty_ema_decay = 0.9
    fppo.projection_eps = 1.0e-8
    fppo._update_counter = 2
    fppo._sigma_a_cache = torch.tensor([0.3, 0.4])
    fppo._uncertainty_cache_age = 0

    called = {"count": 0}

    def _compute_constraint_gradients(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("cache path should not recompute shard gradients")

    fppo._compute_constraint_gradients = _compute_constraint_gradients

    sigma = FPPO._estimate_gradient_uncertainty(
        fppo,
        projection_batch={"obs": torch.zeros(8, 2)},
        a_mat=torch.zeros(3, 2),
        fisher_diag=torch.ones(3),
    )

    assert called["count"] == 0
    assert fppo._uncertainty_cache_age == 1
    assert torch.allclose(sigma, torch.tensor([0.3, 0.4]))


def test_fppo_projection_corrector_falls_back_on_nonfinite_kl() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.device = "cpu"
    fppo.desired_kl = 0.01
    fppo.backtrack_coeff = 0.5
    fppo.max_backtracks = 2
    fppo.projection_radius_cap = None
    fppo.projection_radius_mode = "none"

    theta_anchor = torch.tensor([0.0, 0.0])
    theta_predictor = torch.tensor([1.0, -1.0])
    state = {"theta": theta_anchor.clone()}

    def _set_actor_param_vector(theta: torch.Tensor):
        state["theta"] = theta.detach().clone()

    fppo._set_actor_param_vector = _set_actor_param_vector
    fppo._evaluate_candidate_kl = lambda *_args, **_kwargs: torch.tensor(float("nan"))
    fppo._apply_projection_radius_cap = lambda delta_proj, fisher_diag: (delta_proj, 1.0)

    metrics = FPPO._run_projection_corrector(
        fppo,
        projection_batch={},
        projection_state={
            "a_mat": torch.zeros(theta_anchor.numel(), 0),
            "budgets": torch.zeros(0),
            "fisher_diag": torch.ones(theta_anchor.numel()),
        },
        theta_anchor=theta_anchor,
        theta_predictor=theta_predictor,
    )

    assert metrics["accept_rate"] == 0.0
    assert metrics["accepted_backtrack_factor"] == 0.0
    assert torch.allclose(state["theta"], theta_anchor)


def test_fppo_projection_corrector_accepts_after_backtracking() -> None:
    fppo = FPPO.__new__(FPPO)
    fppo.device = "cpu"
    fppo.desired_kl = 0.30
    fppo.backtrack_coeff = 0.5
    fppo.max_backtracks = 3
    fppo.projection_radius_cap = None
    fppo.projection_radius_mode = "none"

    theta_anchor = torch.tensor([0.0, 0.0])
    theta_predictor = torch.tensor([1.0, 0.0])
    state = {"theta": theta_anchor.clone()}

    def _set_actor_param_vector(theta: torch.Tensor):
        state["theta"] = theta.detach().clone()

    def _evaluate_candidate_kl(_projection_batch, theta_candidate: torch.Tensor) -> torch.Tensor:
        return theta_candidate.pow(2).sum()

    fppo._set_actor_param_vector = _set_actor_param_vector
    fppo._evaluate_candidate_kl = _evaluate_candidate_kl
    fppo._apply_projection_radius_cap = lambda delta_proj, fisher_diag: (delta_proj, 1.0)

    metrics = FPPO._run_projection_corrector(
        fppo,
        projection_batch={},
        projection_state={
            "a_mat": torch.zeros(theta_anchor.numel(), 0),
            "budgets": torch.zeros(0),
            "fisher_diag": torch.ones(theta_anchor.numel()),
        },
        theta_anchor=theta_anchor,
        theta_predictor=theta_predictor,
    )

    assert metrics["accept_rate"] == 1.0
    assert abs(metrics["accepted_backtrack_factor"] - 0.5) < 1.0e-6
    assert abs(metrics["backtrack_steps"] - 1.0) < 1.0e-6
    assert torch.allclose(state["theta"], torch.tensor([0.5, 0.0]), atol=1.0e-6)


def test_fppo_predictor_hard_limit_stops_and_adapts_lr() -> None:
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

    actor_param = torch.nn.Parameter(torch.tensor(0.0))
    fppo = FPPO.__new__(FPPO)
    fppo.policy = _DummyPolicy(actor_param)
    fppo.num_learning_epochs = 1
    fppo.num_mini_batches = 4
    fppo.predictor_kl_target = 0.01
    fppo.predictor_kl_hard_limit = 0.02
    fppo.predictor_adaptive_lr = True
    fppo.predictor_lr = 1.0e-3
    fppo.predictor_lr_min = 1.0e-4
    fppo.predictor_lr_max = 3.0e-3
    fppo.normalize_advantage_per_mini_batch = False
    fppo.clip_param = 0.2
    fppo.entropy_coef = 0.0
    fppo.is_multi_gpu = False
    fppo.max_grad_norm = 1.0
    fppo._actor_params = [actor_param]
    fppo.actor_optimizer = torch.optim.SGD(fppo._actor_params, lr=fppo.predictor_lr)
    fppo._mini_batch_generator = lambda: iter([batch, batch, batch, batch])
    fppo._sanitize_tensor = lambda tensor, **_kwargs: tensor
    fppo._all_reduce_mean = lambda tensor: tensor
    fppo._safe_ratio = lambda new, old: torch.exp(new - old)
    fppo._history_reconstruction_loss = (
        lambda *_args, **_kwargs: (torch.tensor(0.0), torch.tensor(0.0))
    )
    fppo._policy_act = lambda obs_batch, **kwargs: fppo.policy.act(obs_batch, **kwargs)
    fppo._set_predictor_learning_rate = FPPO._set_predictor_learning_rate.__get__(fppo, FPPO)

    kl_values = iter([torch.tensor(0.0), torch.tensor(0.03)])
    fppo._safe_kl = lambda *args, **kwargs: next(kl_values).expand(1)

    metrics = FPPO._run_ppo_predictor(fppo)

    assert metrics["updates"] == 1.0
    assert abs(metrics["update_ratio"] - 0.25) < 1.0e-12
    assert abs(metrics["stop_rate"] - 0.75) < 1.0e-12
    assert fppo.predictor_lr < 1.0e-3
