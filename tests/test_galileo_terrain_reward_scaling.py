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
    assert '"high_speed_threshold": 0.8' in segment
    assert '"low_speed_scale": 1.5' in segment
    assert '"high_speed_scale": 0.5' in segment
