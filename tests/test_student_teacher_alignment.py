from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MDP_CFG_FILE = REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "mdp_cfg.py"
DEFAULTS_FILE = REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "defaults.py"
STUDENT_ENV_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "student_env_cfg.py"
)
REWARDS_FILE = REPO_ROOT / "crl_isaaclab" / "envs" / "mdp" / "rewards.py"
OBSERVATIONS_FILE = REPO_ROOT / "crl_isaaclab" / "envs" / "mdp" / "observations.py"
RSL_CFG_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "agents" / "rsl_rl_cfg.py"
)
RUNNER_FILE = (
    REPO_ROOT / "scripts" / "rsl_rl" / "modules" / "on_policy_runner_with_extractor.py"
)


def _module_and_source(path: Path) -> tuple[ast.Module, str]:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source), source


def _class_node(container: ast.Module | ast.ClassDef, name: str) -> ast.ClassDef:
    for node in container.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found in {container}.")


def _function_node(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found in {module}.")


def _class_assignments(node: ast.ClassDef) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            assignments[stmt.target.id] = stmt.value
    return assignments


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError("Failed to recover source segment.")
    return segment


def test_feet_slide_has_fallback_for_broad_asset_body_selection() -> None:
    module, source = _module_and_source(REWARDS_FILE)
    feet_slide = _function_node(module, "feet_slide")
    segment = _source_segment(source, feet_slide)

    assert "sensor_body_count" in segment
    assert "isinstance(foot_body_ids, slice)" in segment
    assert "len(foot_body_ids) != sensor_body_count" in segment
    assert segment.count("foot_body_ids = sensor_body_ids") >= 2


def test_student_and_teacher_rewards_keep_only_nonzero_weight_terms() -> None:
    module, source = _module_and_source(MDP_CFG_FILE)
    student = _class_assignments(_class_node(module, "StudentRewardsCfg"))
    teacher = _class_assignments(_class_node(module, "TeacherRewardsCfg"))

    student_terms = {name for name in student.keys() if name != "only_positive_rewards"}
    teacher_terms = {name for name in teacher.keys() if name != "only_positive_rewards"}
    removed_zero_weight_terms = {
        "base_height_l2_fix",
        "joint_vel_l2",
        "joint_power_distribution",
        "load_sharing",
        "gait_contact_symmetry",
        "fore_hind_contact_balance",
        "foot_clearance",
    }

    assert removed_zero_weight_terms.isdisjoint(student_terms)
    assert removed_zero_weight_terms.isdisjoint(teacher_terms)
    assert student_terms == teacher_terms | {"action_smoothness_l2"}
    assert "action_smoothness_l2" not in teacher_terms
    assert "flat_orientation_l2" in student_terms
    assert "flat_orientation_l2" in teacher_terms
    assert _source_segment(source, student["only_positive_rewards"]) == "True"
    assert "weight=-0.5" in _source_segment(source, student["flat_orientation_l2"])
    assert '"flat_terrain_name": "crl_flat"' in _source_segment(source, student["flat_orientation_l2"])
    assert "weight=-1.0" in _source_segment(source, teacher["flat_orientation_l2"])
    assert '"flat_terrain_name": "crl_flat"' in _source_segment(source, teacher["flat_orientation_l2"])
    assert "weight=0.1" in _source_segment(source, student["trot_phase_reward"])
    assert "weight=0.35" in _source_segment(source, teacher["trot_phase_reward"])
    assert "weight=-0.0" in _source_segment(source, student["action_smoothness_l2"])
    assert '"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")' in _source_segment(
        source, student["feet_slide"]
    )


def test_student_curriculum_is_teacher_aligned_but_more_conservative() -> None:
    module, source = _module_and_source(MDP_CFG_FILE)
    student_curriculum = _class_assignments(_class_node(module, "StudentCurriculumCfg"))
    teacher_curriculum = _class_assignments(_class_node(module, "CurriculumCfg"))

    assert student_curriculum.keys() == teacher_curriculum.keys()
    assert "command_warmup_steps * 1.5" in _source_segment(
        source, student_curriculum["lin_vel_x_command_threshold"]
    )
    assert "command_min_progress_steps * 1.25" in _source_segment(
        source, student_curriculum["lin_vel_x_command_threshold"]
    )
    assert "max(float(GalileoDefaults.curriculum.ang_min_lin_x_level), 0.45)" in _source_segment(
        source, student_curriculum["ang_vel_z_command_threshold"]
    )

    student_env_module, student_env_source = _module_and_source(STUDENT_ENV_FILE)
    student_env = _class_assignments(_class_node(student_env_module, "GalileoStudentCRLEnvCfg"))
    assert "StudentCurriculumCfg()" in _source_segment(student_env_source, student_env["curriculum"])


def test_student_constraint_curriculum_names_are_joint_feasibility_terms() -> None:
    module, source = _module_and_source(DEFAULTS_FILE)
    defaults_cls = _class_node(module, "GalileoDefaults")
    algorithm_cls = _class_node(defaults_cls, "algorithm")
    algorithm_assignments = _class_assignments(algorithm_cls)
    student_override_segment = _source_segment(source, algorithm_assignments["student_override"])

    assert '"prob_joint_pos"' in student_override_segment
    assert '"prob_joint_vel"' in student_override_segment
    assert '"prob_joint_torque"' in student_override_segment


def test_student_proprio_history_scales_match_policy_observation_terms() -> None:
    module, source = _module_and_source(MDP_CFG_FILE)
    student_observations = _class_node(module, "StudentObservationsCfg")
    policy_cfg = _class_node(student_observations, "PolicyCfg")
    policy_assignments = _class_assignments(policy_cfg)
    history_segment = _source_segment(source, policy_assignments["proprio_history"])

    assert '"scales"' in history_segment
    assert '"base_ang_vel": 0.25' in history_segment
    assert '"projected_gravity": 1.0' in history_segment
    assert '"joint_vel": 0.05' in history_segment
    assert '"last_action": 0.25' in history_segment
    assert '"commands": (2.0, 2.0, 0.25)' in history_segment


def test_dagger_teacher_action_schedule_defaults_are_teacher_guided() -> None:
    rsl_module, rsl_source = _module_and_source(RSL_CFG_FILE)
    dagger_cfg = _class_assignments(_class_node(rsl_module, "CRLRslRlDAggerAlgorithmCfg"))

    assert _source_segment(rsl_source, dagger_cfg["teacher_action_ratio_start"]) == "1.0"
    assert _source_segment(rsl_source, dagger_cfg["teacher_action_ratio_end"]) == "0.0"
    assert _source_segment(rsl_source, dagger_cfg["teacher_action_ratio_decay_steps"]) == "8000"
    assert _source_segment(rsl_source, dagger_cfg["dagger_buffer_size"]) == "1_048_576"
    assert _source_segment(rsl_source, dagger_cfg["dagger_min_buffer_size"]) == "262_144"

    defaults_module, defaults_source = _module_and_source(DEFAULTS_FILE)
    defaults_cls = _class_node(defaults_module, "GalileoDefaults")
    algorithm_cls = _class_node(defaults_cls, "algorithm")
    algorithm_assignments = _class_assignments(algorithm_cls)
    per_algo_segment = _source_segment(defaults_source, algorithm_assignments["per_algo"])

    assert 'teacher_action_ratio_start=1.0' in per_algo_segment
    assert 'teacher_action_ratio_end=0.0' in per_algo_segment
    assert 'teacher_action_ratio_decay_steps=8000' in per_algo_segment
    assert 'dagger_buffer_size=1_048_576' in per_algo_segment
    assert 'dagger_min_buffer_size=262_144' in per_algo_segment


def test_runner_keeps_history_enabled_for_history_only_student_rollouts() -> None:
    source = RUNNER_FILE.read_text(encoding="utf-8")

    assert "policy_requires_history_rollout" in source
    assert 'getattr(actor_module, "num_hist", 0) > 0' in source
    assert 'getattr(actor_module, "num_priv_latent", 0) or 0) <= 0' in source
    assert "policy_hist_encoding = history_encoder_update_due or policy_requires_history_rollout" in source


def test_policy_history_supports_per_term_scaling() -> None:
    module, source = _module_and_source(OBSERVATIONS_FILE)
    policy_history = _class_node(module, "PolicyHistory")
    segment = _source_segment(source, policy_history)

    assert "def _apply_scale" in segment
    assert 'scale_cfg = dict(scales or {})' in segment
    assert 'scale_cfg.get("base_ang_vel", 1.0)' in segment
    assert 'scale_cfg.get("commands", 1.0)' in segment
