from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MDP_CFG_FILE = REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "mdp_cfg.py"
REWARDS_FILE = REPO_ROOT / "crl_isaaclab" / "envs" / "mdp" / "rewards.py"


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


def test_dof_error_l2_supports_command_speed_scaling() -> None:
    module, source = _module_and_source(REWARDS_FILE)
    dof_error = _function_node(module, "dof_error_l2")
    segment = _source_segment(source, dof_error)

    assert "low_speed_threshold: float = 0.1" in segment
    assert "high_speed_threshold: float = 0.8" in segment
    assert "low_speed_scale: float = 1.0" in segment
    assert "high_speed_scale: float = 1.0" in segment
    assert "_command_speed_scale(" in segment
    assert "return reward * speed_scale" in segment


def test_galileo_dof_error_reward_uses_speed_scaling_params() -> None:
    module, source = _module_and_source(MDP_CFG_FILE)
    cts = _class_assignments(_class_node(module, "CTSRewardsCfg"))

    segment = _source_segment(source, cts["dof_error_l2"])
    assert '"command_name": "base_velocity"' in segment
    assert '"low_speed_threshold": 0.1' in segment
    assert '"high_speed_threshold": 0.35' in segment
    assert '"low_speed_scale": 1.2' in segment
    assert '"high_speed_scale": 0.15' in segment


def test_flat_orientation_reward_supports_terrain_scaling() -> None:
    module, source = _module_and_source(REWARDS_FILE)
    flat_orientation = _function_node(module, "flat_orientation_l2")
    segment = _source_segment(source, flat_orientation)

    assert "terrain_scales: dict[str, float] | None = None" in segment
    assert "default_scale: float = 0.0" in segment
    assert "_terrain_name_scale(" in segment
    assert "return reward * terrain_scale" in segment


def test_galileo_flat_orientation_reward_uses_terrain_scaling_params() -> None:
    module, source = _module_and_source(MDP_CFG_FILE)
    cts = _class_assignments(_class_node(module, "CTSRewardsCfg"))

    segment = _source_segment(source, cts["flat_orientation_l2"])
    assert '"default_scale": 1.5' in segment
    assert '"terrain_scales": {' in segment
    assert '"plane_run": 1.5' in segment
    assert '"plane_yaw": 1.5' in segment
    assert '"plane_stand": 1.5' in segment
    assert '"random_rough": 1.5' in segment
    assert '"boxes": 1.2' in segment
    assert '"pyramid_stairs": 0.25' in segment
    assert '"pyramid_stairs_inv": 0.25' in segment
    assert '"hf_pyramid_slope": 0.25' in segment
    assert '"hf_pyramid_slope_inv": 0.25' in segment


def test_foot_clearance_reward_supports_terrain_relative_height() -> None:
    module, source = _module_and_source(REWARDS_FILE)
    foot_clearance = _function_node(module, "foot_clearance")
    segment = _source_segment(source, foot_clearance)

    assert "terrain_sensor_cfg: SceneEntityCfg | None = None" in segment
    assert "_foot_heights_relative(" in segment


def test_galileo_foot_clearance_reward_uses_height_scanner() -> None:
    module, source = _module_and_source(MDP_CFG_FILE)
    cts = _class_assignments(_class_node(module, "CTSRewardsCfg"))

    segment = _source_segment(source, cts["foot_clearance"])
    assert '"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")' in segment
    assert '"terrain_sensor_cfg": SceneEntityCfg("height_scanner")' in segment


def test_galileo_uses_lightweight_stance_shaping_without_trot_phase_cost() -> None:
    module, source = _module_and_source(MDP_CFG_FILE)
    cts = _class_assignments(_class_node(module, "CTSRewardsCfg"))

    hip = _source_segment(source, cts["hip_pos_l2"])
    assert '"high_speed_scale": 0.8' in hip

    symmetry = _source_segment(source, cts["gait_contact_symmetry"])
    assert "weight=0.05" in symmetry
    assert '"command_name": None' in symmetry

    trot = _source_segment(source, cts["trot_phase_reward"])
    assert "weight=0.0" in trot
    assert '"min_command_speed": 0.12' in trot
    assert '"low_speed_threshold": 0.35' in trot
    assert '"max_abs_yaw_cmd": 0.3' in trot
