from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.rsl_rl.experiment_manager import (
    ExperimentPresetError,
    apply_experiment_preset,
    available_experiment_presets,
    load_experiment_preset,
    write_experiment_metadata,
)


class ExperimentManagerTest(unittest.TestCase):
    def test_load_experiment_preset_supports_extends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            experiments_dir = repo_root / "experiments" / "galileo"
            experiments_dir.mkdir(parents=True)
            (experiments_dir / "base.json").write_text(
                '{"meta": {"description": "base"}, "agent": {"algorithm": {"cost_limit": 0.0}}}',
                encoding="utf-8",
            )
            (experiments_dir / "child.json").write_text(
                (
                    '{"extends": "galileo/base", '
                    '"meta": {"description": "child"}, '
                    '"agent": {"algorithm": {"cost_limit": 0.1}}, '
                    '"env": {"scene": {"num_envs": 512}}}'
                ),
                encoding="utf-8",
            )

            preset = load_experiment_preset(selection="galileo/child", root=repo_root)

        assert preset is not None
        self.assertEqual(preset.name, "galileo/child")
        self.assertEqual(preset.agent_overrides["algorithm"]["cost_limit"], 0.1)
        self.assertEqual(preset.env_overrides["scene"]["num_envs"], 512)
        self.assertEqual(len(preset.source_chain), 2)

    def test_apply_experiment_preset_updates_nested_objects(self) -> None:
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(num_envs=4096),
            terrain=SimpleNamespace(difficulty_range=(0.0, 1.0)),
        )
        agent_cfg = SimpleNamespace(
            max_iterations=100000,
            algorithm=SimpleNamespace(cost_limit=0.0, class_name="FPPO"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            experiments_dir = repo_root / "experiments" / "galileo"
            experiments_dir.mkdir(parents=True)
            (experiments_dir / "preset.json").write_text(
                (
                    '{"env": {"scene": {"num_envs": 1024}, "terrain": {"difficulty_range": [0.3, 0.9]}}, '
                    '"agent": {"max_iterations": 2000, "algorithm": {"cost_limit": 0.05}}}'
                ),
                encoding="utf-8",
            )

            preset = load_experiment_preset(selection="galileo/preset", root=repo_root)
            assert preset is not None
            apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=preset)

        self.assertEqual(env_cfg.scene.num_envs, 1024)
        self.assertEqual(env_cfg.terrain.difficulty_range, (0.3, 0.9))
        self.assertEqual(agent_cfg.max_iterations, 2000)
        self.assertEqual(agent_cfg.algorithm.cost_limit, 0.05)

    def test_available_experiment_presets_reports_description(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            experiments_dir = repo_root / "experiments" / "galileo"
            experiments_dir.mkdir(parents=True)
            (experiments_dir / "preset.json").write_text(
                '{"meta": {"description": "hello world"}}', encoding="utf-8"
            )

            entries = available_experiment_presets(root=repo_root)

        self.assertEqual(entries[0]["name"], "galileo/preset")
        self.assertEqual(entries[0]["description"], "hello world")

    def test_write_experiment_metadata_persists_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            experiments_dir = repo_root / "experiments" / "galileo"
            experiments_dir.mkdir(parents=True)
            preset_file = experiments_dir / "preset.json"
            preset_file.write_text('{"meta": {"description": "meta"}}', encoding="utf-8")
            preset = load_experiment_preset(selection="galileo/preset", root=repo_root)
            assert preset is not None

            metadata_path = write_experiment_metadata(
                repo_root / "logs" / "run_1",
                preset,
                args=Namespace(exp="galileo/preset", run_name="teacher"),
            )

            self.assertTrue(metadata_path.exists())
            self.assertIn("preset_name", metadata_path.read_text(encoding="utf-8"))

    def test_unknown_override_key_raises(self) -> None:
        env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=4096))
        agent_cfg = SimpleNamespace(max_iterations=1000)

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            experiments_dir = repo_root / "experiments" / "galileo"
            experiments_dir.mkdir(parents=True)
            (experiments_dir / "bad.json").write_text(
                '{"agent": {"does_not_exist": 1}}', encoding="utf-8"
            )
            preset = load_experiment_preset(selection="galileo/bad", root=repo_root)
            assert preset is not None

            with self.assertRaises(ExperimentPresetError):
                apply_experiment_preset(env_cfg=env_cfg, agent_cfg=agent_cfg, preset=preset)


if __name__ == "__main__":
    unittest.main()
