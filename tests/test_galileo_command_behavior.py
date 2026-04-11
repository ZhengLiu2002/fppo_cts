from __future__ import annotations

import ast
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "defaults.py"
)
COMMAND_CFG_FILE = (
    REPO_ROOT / "crl_isaaclab" / "envs" / "mdp" / "crl_commands" / "crl_command_cfg.py"
)
UNIFORM_COMMAND_FILE = (
    REPO_ROOT / "crl_isaaclab" / "envs" / "mdp" / "crl_commands" / "uniform_crl_command.py"
)
KEYBOARD_PLAY_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "play_keyboard.py"


def _module_and_source(path: Path) -> tuple[ast.Module, str]:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source), source


def _class_node(container: ast.Module | ast.ClassDef, name: str) -> ast.ClassDef:
    for node in container.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found in {container}.")


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


def test_rough_terrain_ranges_keep_a_nonzero_standing_probability() -> None:
    module, source = _module_and_source(DEFAULTS_FILE)
    defaults_cls = _class_node(module, "GalileoDefaults")
    command_cls = _class_node(defaults_cls, "command")
    command_assignments = _class_assignments(command_cls)
    ranges_segment = _source_segment(source, command_assignments["ranges"])

    assert ranges_segment.count("standing_command_prob=0.10") == 7
    assert re.search(r"standing_command_prob=0\\.0(?!\\d)", ranges_segment) is None


def test_training_command_pipeline_keeps_small_command_deadzone_enabled() -> None:
    command_cfg_source = COMMAND_CFG_FILE.read_text(encoding="utf-8")
    uniform_source = UNIFORM_COMMAND_FILE.read_text(encoding="utf-8")

    assert "small_commands_to_zero: bool = True" in command_cfg_source
    assert "self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= standing_prob" in uniform_source
    assert "xy_norm = torch.norm(self.vel_command_b[env_ids, :2]" in uniform_source
    assert "torch.abs(self.vel_command_b[:, 2]) > self.cfg.clips.ang_vel_clip" in uniform_source


def test_keyboard_play_script_uses_manual_command_injection_for_one_robot() -> None:
    source = KEYBOARD_PLAY_FILE.read_text(encoding="utf-8")

    assert "args_cli.num_envs = 1" in source
    assert "subscribe_to_keyboard_events" in source
    assert 'get_term("base_velocity")' in source
    assert "TerrainGeneratorCfg(" in source
    assert 'params.get("command_name") != "base_velocity"' in source
    assert '"\\r[CMD] "' in source
    assert "def _move_towards" in source
    assert "def step(self, dt: float" in source
