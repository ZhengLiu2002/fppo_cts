from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "defaults.py"
)
CTS_ENV_CFG_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "cts_env_cfg.py"
)
SCENE_CFG_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "scene_cfg.py"
)
TERRAIN_PROFILES_FILE = (
    REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "terrain_profiles.py"
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


def test_extreme_load_terrain_ranges_keep_reference_command_modes() -> None:
    module, source = _module_and_source(DEFAULTS_FILE)
    defaults_cls = _class_node(module, "GalileoDefaults")
    command_cls = _class_node(defaults_cls, "command")
    command_assignments = _class_assignments(command_cls)
    ranges_segment = _source_segment(source, command_assignments["ranges"])

    assert ranges_segment.count("heading_command_prob=0.7") == 6
    assert "yaw_command_prob=0.5" in ranges_segment
    assert ranges_segment.count("standing_command_prob=0.05") == 7
    assert '"plane_run": dict(' in ranges_segment
    assert '"plane_yaw": dict(' in ranges_segment
    assert '"plane_stand": dict(' in ranges_segment


def test_flat_profile_keeps_extreme_load_plane_command_modes_split() -> None:
    defaults_source = DEFAULTS_FILE.read_text(encoding="utf-8")
    scene_source = SCENE_CFG_FILE.read_text(encoding="utf-8")
    cts_source = CTS_ENV_CFG_FILE.read_text(encoding="utf-8")
    terrain_profiles_source = TERRAIN_PROFILES_FILE.read_text(encoding="utf-8")

    assert 'flat_subterrain_names = ("plane_run", "plane_yaw", "plane_stand")' in defaults_source
    assert '"plane_run": 0.5' in defaults_source
    assert '"plane_yaw": 0.25' in defaults_source
    assert '"plane_stand": 0.25' in defaults_source
    assert "terrain_mode_groups = {" in defaults_source
    assert '"plane_run": ("plane_run",)' in defaults_source
    assert '"plane_yaw": ("plane_yaw",)' in defaults_source
    assert '"plane_stand": ("plane_stand",)' in defaults_source
    assert "restrict_terrain_generator_to_named_subterrains" in scene_source
    assert "restrict_terrain_generator_to_named_subterrains" in cts_source
    assert 'f"debug_{terrain_name}"' not in cts_source
    assert "debug_{terrain_name}" not in cts_source
    assert "preserving their public names" in terrain_profiles_source


def test_training_command_pipeline_keeps_small_command_deadzone_enabled() -> None:
    command_cfg_source = COMMAND_CFG_FILE.read_text(encoding="utf-8")
    uniform_source = UNIFORM_COMMAND_FILE.read_text(encoding="utf-8")

    assert "small_commands_to_zero: bool = True" in command_cfg_source
    assert "standing_prob" in uniform_source
    assert "self.is_standing_env[" in uniform_source
    assert "r.uniform_(0.0, 1.0) <= standing_prob" in uniform_source
    assert "xy_norm = torch.norm(self.vel_command_b[" in uniform_source
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
