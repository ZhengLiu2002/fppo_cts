from __future__ import annotations

from pathlib import Path
from scripts.rsl_rl.algorithms.registry import get_algorithm_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
GALILEO_INIT = REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "__init__.py"
EXPORTER_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "exporter.py"
STORAGE_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "storage" / "rollout_storage.py"
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


def test_cts_registry_entry_uses_dedicated_training_type() -> None:
    spec = get_algorithm_spec("cts")

    assert spec.class_name == "CTS"
    assert spec.training_type == "cts"


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
    assert 'name: str = "fppo"' in defaults_source


def test_exporter_recognizes_cts_teacher_augmented_history_layout() -> None:
    source = EXPORTER_FILE.read_text(encoding="utf-8")

    assert "can_split_cts_history" in source
    assert 'name.startswith("teacher_")' in source
    assert "student_inferred" in source


def test_rollout_storage_treats_cts_as_reinforcement_learning() -> None:
    source = STORAGE_FILE.read_text(encoding="utf-8")

    assert 'training_type in {"rl", "cts"}' in source
    assert 'self.training_type not in {"rl", "cts"}' in source
