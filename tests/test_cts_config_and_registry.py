from __future__ import annotations

from pathlib import Path

from scripts.rsl_rl.algorithms.contracts import (
    CTSRuntimeContract,
    resolve_cts_runtime_contract,
)
from scripts.rsl_rl.algorithms.registry import (
    get_algorithm_class,
    get_algorithm_spec,
    list_algorithm_names,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GALILEO_INIT = REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "__init__.py"
EXPORTER_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "exporter.py"
STORAGE_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "storage" / "rollout_storage.py"
RUNNER_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "modules" / "on_policy_runner_with_extractor.py"
RUNNER_CFG_FILE = (
    REPO_ROOT
    / "crl_tasks"
    / "crl_tasks"
    / "tasks"
    / "galileo"
    / "config"
    / "agents"
    / "rsl_cts_cfg.py"
)
DEFAULTS_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "defaults.py"
)
SCENE_CFG_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "scene_cfg.py"
)
RSL_RL_CFG_FILE = (
    REPO_ROOT
    / "crl_tasks"
    / "crl_tasks"
    / "tasks"
    / "galileo"
    / "config"
    / "agents"
    / "rsl_rl_cfg.py"
)
MDP_CFG_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "mdp_cfg.py"
)
CTS_ENV_CFG_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "cts_env_cfg.py"
)
TRAIN_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "train.py"
EFFECTIVE_PARAMS_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "effective_params.py"
DUMP_EFFECTIVE_PARAMS_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "dump_effective_params.py"


def test_cts_registry_entry_uses_dedicated_training_type() -> None:
    spec = get_algorithm_spec("cts")

    assert spec.class_name == "CTS"
    assert spec.training_type == "cts"
    assert spec.config_family == "cts"


def test_registered_algorithms_expose_cts_runtime_contract() -> None:
    for algo_name in list_algorithm_names():
        contract = resolve_cts_runtime_contract(get_algorithm_class(algo_name))
        assert isinstance(contract, CTSRuntimeContract)


def test_fppo_declares_named_constraint_runtime_hook() -> None:
    assert resolve_cts_runtime_contract(get_algorithm_class("ppo")).inject_constraint_names is False
    assert resolve_cts_runtime_contract(get_algorithm_class("fppo")).inject_constraint_names is True


def test_galileo_registers_dedicated_cts_tasks() -> None:
    source = GALILEO_INIT.read_text(encoding="utf-8")

    assert 'id="Isaac-Galileo-CTS-v0"' in source
    assert 'id="Isaac-Galileo-CTS-Eval-v0"' in source
    assert 'id="Isaac-Galileo-CTS-Play-v0"' in source
    assert "cts_env_cfg.__name__" in source
    assert "rsl_cts_cfg:GalileoCTSBenchmarkRunnerCfg" in source
    assert "Isaac-Galileo-CRL-Teacher-v0" not in source
    assert "Isaac-Galileo-CRL-Student-v0" not in source


def test_cts_runner_defaults_to_fppo_benchmark_profile() -> None:
    runner_source = RUNNER_CFG_FILE.read_text(encoding="utf-8")
    defaults_source = DEFAULTS_FILE.read_text(encoding="utf-8")

    assert "algorithm = build_cts_algorithm_cfg()" in runner_source
    assert "get_algorithm_spec(selected_algo)" in runner_source
    assert 'framework_type = "cts"' in runner_source
    assert 'selected_algo == "fppo"' not in runner_source
    assert 'name: str = "fppo"' in defaults_source
    assert "class_name_map" not in defaults_source


def test_fppo_cts_config_inherits_shared_cts_algorithm_fields() -> None:
    source = RSL_RL_CFG_FILE.read_text(encoding="utf-8")

    assert "class CRLRslRlFppoAlgorithmCfg(CRLRslRlPpoAlgorithmCfg):" in source
    assert "student_group_ratio: float = 0.25" in source
    assert "velocity_estimation_loss_coef: float = 0.05" in source
    assert "reconstruction_learning_rate: float = 3e-4" in source
    assert "num_reconstruction_epochs: int = 1" in source
    assert 'roa_teacher_reg_coef_end: float = 0.05' in source
    assert 'roa_teacher_reg_scope: str = "teacher"' in source


def test_cts_runner_uses_algorithm_contracts_for_runtime_hooks() -> None:
    source = RUNNER_FILE.read_text(encoding="utf-8")

    assert "resolve_cts_runtime_contract" in source
    assert 'alg_class_name.lower() == "fppo"' not in source


def test_exporter_recognizes_cts_teacher_augmented_history_layout() -> None:
    source = EXPORTER_FILE.read_text(encoding="utf-8")

    assert "can_split_cts_history" in source
    assert 'name.startswith("teacher_")' in source
    assert "student_inferred" in source
    assert "_load_frozen_run_cfgs" not in source
    assert "EXPORT_PROFILE_GALILEO" not in source


def test_cts_defaults_keep_auxiliary_supervision_in_shared_profile() -> None:
    source = DEFAULTS_FILE.read_text(encoding="utf-8")

    assert "shared_cts_framework = dict(" in source
    assert source.count("velocity_estimation_loss_coef=0.05") == 1
    assert source.count("roa_teacher_reg_coef_end=0.05") == 1
    assert "constraint_adapter_base = dict(" in source
    assert "constraint_adapter_per_algo = {}" in source


def test_cts_defaults_and_commands_match_policy_template_targets() -> None:
    defaults_source = DEFAULTS_FILE.read_text(encoding="utf-8")
    mdp_source = MDP_CFG_FILE.read_text(encoding="utf-8")

    assert "decimation = 8" in defaults_source
    assert "render_interval = 8" in defaults_source
    assert "num_actor_obs = (" not in defaults_source
    assert "num_critic_obs = (" not in defaults_source
    assert "obs_history_length = 5" not in defaults_source
    assert "action_history_length = 3" not in defaults_source
    assert "clip_actions = 100.0" not in defaults_source
    assert "clip_obs = 100.0" not in defaults_source
    assert "random_difficulty = False" not in defaults_source
    assert '"FL_hip_joint": 0.0' in defaults_source
    assert '"FL_thigh_joint": 0.8' in defaults_source
    assert "velocity_x_backward_scale: float = 1.0" in defaults_source
    assert "velocity_y_scale: float = 0.5" in defaults_source
    assert "velocity_yaw_scale: float = 1.0" in defaults_source
    assert "max_velocity: tuple[float, float, float] = (1.2, 0.5, 1.5)" in defaults_source
    assert "max_lin_x_level: float = 5.0" in defaults_source
    assert "max_ang_z_level: float = 5.0" in defaults_source
    assert "heading_control_stiffness: float = 0.5" in defaults_source
    assert "lin_vel_x = (-0.5, 0.5)" in defaults_source
    assert "lin_vel_y = (-0.5, 0.5)" in defaults_source
    assert "ang_vel_z = (-0.25, 0.25)" in defaults_source
    assert "standing_command_prob=0.05" in defaults_source
    assert "heading_command_prob=0.7" in defaults_source
    assert '"plane_run": dict(' in defaults_source
    assert "max_curriculum_lin_x=(-1.0, 1.0)" in defaults_source
    assert "velocity_x_backward_scale=GalileoDefaults.command.velocity_x_backward_scale" in mdp_source
    assert "velocity_y_scale=GalileoDefaults.command.velocity_y_scale" in mdp_source
    assert "velocity_yaw_scale=GalileoDefaults.command.velocity_yaw_scale" in mdp_source
    assert "max_velocity=GalileoDefaults.command.max_velocity" in mdp_source
    assert "heading_control_stiffness=GalileoDefaults.command.heading_control_stiffness" in mdp_source


def test_galileo_env_exposes_config_summary_and_effective_param_dump() -> None:
    defaults_source = DEFAULTS_FILE.read_text(encoding="utf-8")
    cts_source = CTS_ENV_CFG_FILE.read_text(encoding="utf-8")
    train_source = TRAIN_FILE.read_text(encoding="utf-8")
    effective_source = EFFECTIVE_PARAMS_FILE.read_text(encoding="utf-8")
    dump_source = DUMP_EFFECTIVE_PARAMS_FILE.read_text(encoding="utf-8")

    assert "ConfigSummary = GalileoDefaults" in defaults_source
    assert "self.config_summary = GalileoDefaults" in cts_source
    assert 'hasattr(env_cfg, "apply_experiment_overrides")' in train_source
    assert "env_cfg.apply_experiment_overrides()" in train_source
    assert "write_effective_config_summary(log_dir, env_cfg, agent_cfg)" in train_source
    assert 'filename: str = "effective_summary.json"' in effective_source
    assert "build_effective_config_summary" in effective_source
    assert '"plane_split": {' in effective_source
    assert '"terrain_ranges": _to_serializable(ranges)' in effective_source
    assert '"randomization_ranges": {' in effective_source
    assert "AppLauncher.add_app_launcher_args(parser)" in dump_source
    assert "env.reset(seed=args_cli.seed)" in dump_source
    assert '"config_summary": build_effective_config_summary(unwrapped.cfg, agent_cfg)' in dump_source
    assert 'choices=["summary", "per_shape", "none"]' in dump_source


def test_cts_scene_preserves_shared_robot_actuator_overrides() -> None:
    source = SCENE_CFG_FILE.read_text(encoding="utf-8")

    rebuild_idx = source.index("self.robot = build_galileo_robot_cfg()")
    super_idx = source.index("super().__post_init__()")
    assert rebuild_idx < super_idx


def test_shared_galileo_scene_clears_inherited_asset_actuators_before_adding_base_legs() -> None:
    source = DEFAULTS_FILE.read_text(encoding="utf-8")

    clear_idx = source.index("self.robot.actuators = {}")
    base_legs_idx = source.index('self.robot.actuators["base_legs"] = CRLDCMotorCfg(')
    assert clear_idx < base_legs_idx


def test_rollout_storage_treats_cts_as_reinforcement_learning() -> None:
    source = STORAGE_FILE.read_text(encoding="utf-8")

    assert 'training_type in {"rl", "cts"}' in source
    assert 'self.training_type not in {"rl", "cts"}' in source
