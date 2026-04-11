from __future__ import annotations

from argparse import Namespace
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.rsl_rl.runtime import (
    bootstrap_repo_paths,
    build_evaluation_output_path,
    configure_safe_play_args,
    create_run_directory_name,
    dump_pickle,
    resolve_task_variant,
    resolve_checkpoint_path,
)


class RuntimeUtilsTest(unittest.TestCase):
    def test_bootstrap_repo_paths_adds_repo_and_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            script_dir = repo_root / "scripts" / "rsl_rl"
            tasks_dir = repo_root / "crl_tasks"
            script_dir.mkdir(parents=True)
            tasks_dir.mkdir(parents=True)
            anchor_file = script_dir / "train.py"
            anchor_file.write_text("", encoding="utf-8")

            original_sys_path = list(sys.path)
            sys.path = ["existing-entry"]
            try:
                resolved_root = bootstrap_repo_paths(anchor_file, repo_root=repo_root)
            finally:
                updated_sys_path = list(sys.path)
                sys.path = original_sys_path

        self.assertEqual(resolved_root, repo_root.resolve())
        self.assertEqual(updated_sys_path[0], str(repo_root.resolve()))
        self.assertEqual(updated_sys_path[1], str(tasks_dir.resolve()))

    def test_create_run_directory_name_uses_env_prefix(self) -> None:
        original_prefix = os.environ.get("LOG_RUN_NAME")
        os.environ["LOG_RUN_NAME"] = "fppo"
        try:
            directory_name, needs_exact_name_log = create_run_directory_name(
                "cts",
                timestamp=datetime(2026, 3, 6, 12, 0, 0),
            )
        finally:
            if original_prefix is None:
                os.environ.pop("LOG_RUN_NAME", None)
            else:
                os.environ["LOG_RUN_NAME"] = original_prefix

        self.assertEqual(directory_name, "fppo_2026-03-06_12-00-00_cts")
        self.assertFalse(needs_exact_name_log)

    def test_create_run_directory_name_includes_experiment_slug(self) -> None:
        directory_name, needs_exact_name_log = create_run_directory_name(
            "cts",
            timestamp=datetime(2026, 3, 6, 12, 0, 0),
            experiment_slug="cost-limit-relaxed",
        )

        self.assertEqual(
            directory_name,
            "2026-03-06_12-00-00_cost-limit-relaxed_cts",
        )
        self.assertTrue(needs_exact_name_log)

    def test_dump_pickle_appends_suffix_and_creates_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "artifacts" / "env"
            written_path = dump_pickle(file_path, {"seed": 42})

            self.assertEqual(written_path, Path(temp_dir) / "artifacts" / "env.pkl")
            self.assertTrue(written_path.exists())

    def test_configure_safe_play_args_forces_headless_without_display(self) -> None:
        original_display = os.environ.pop("DISPLAY", None)
        original_wayland = os.environ.pop("WAYLAND_DISPLAY", None)
        original_livestream = os.environ.pop("LIVESTREAM", None)
        args = Namespace(force_gui=False, headless=False, livestream=-1, kit_args="")
        try:
            configure_safe_play_args(args)
        finally:
            if original_display is not None:
                os.environ["DISPLAY"] = original_display
            if original_wayland is not None:
                os.environ["WAYLAND_DISPLAY"] = original_wayland
            if original_livestream is not None:
                os.environ["LIVESTREAM"] = original_livestream

        self.assertTrue(args.headless)
        self.assertIn("--disable omni.physx.ui", args.kit_args)
        self.assertIn("--disable omni.kit.window.drop_support", args.kit_args)

    def test_configure_safe_play_args_respects_force_gui(self) -> None:
        original_headless = os.environ.pop("HEADLESS", None)
        args = Namespace(force_gui=True, headless=True, livestream=-1, kit_args="")
        try:
            configure_safe_play_args(args)
            self.assertFalse(args.headless)
            self.assertEqual(args.kit_args, "")
            self.assertEqual(os.environ["HEADLESS"], "0")
        finally:
            if original_headless is None:
                os.environ.pop("HEADLESS", None)
            else:
                os.environ["HEADLESS"] = original_headless

    def test_resolve_checkpoint_path_accepts_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "fppo_2026-03-18_teacher"
            run_dir.mkdir(parents=True)
            (run_dir / "model_100.pt").write_text("", encoding="utf-8")
            (run_dir / "model_1200.pt").write_text("", encoding="utf-8")

            resolved = resolve_checkpoint_path(
                task_name="Isaac-Galileo-CTS-Play-v0",
                log_root_path=temp_dir,
                load_run=".*",
                load_checkpoint="model_.*.pt",
                checkpoint=str(run_dir),
                algo_name="fppo",
            )

        self.assertEqual(resolved, str(run_dir / "model_1200.pt"))

    def test_resolve_checkpoint_path_filters_experiment_root_by_algo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_root = Path(temp_dir) / "galileo_algo_compare_cts_fair"
            fppo_run = experiment_root / "fppo_2026-03-14_cts"
            ppo_run = experiment_root / "ppo_2026-03-14_cts"
            fppo_run.mkdir(parents=True)
            ppo_run.mkdir(parents=True)
            (fppo_run / "model_500.pt").write_text("", encoding="utf-8")
            (fppo_run / "model_1500.pt").write_text("", encoding="utf-8")
            (ppo_run / "model_9000.pt").write_text("", encoding="utf-8")

            resolved = resolve_checkpoint_path(
                task_name="Isaac-Galileo-CTS-Play-v0",
                log_root_path=temp_dir,
                load_run=".*",
                load_checkpoint="model_.*.pt",
                checkpoint=str(experiment_root),
                algo_name="fppo",
            )

        self.assertEqual(resolved, str(fppo_run / "model_1500.pt"))

    def test_resolve_task_variant_prefers_registered_eval_task(self) -> None:
        registered = {
            "Isaac-Galileo-CTS-v0",
            "Isaac-Galileo-CTS-Eval-v0",
        }

        resolved = resolve_task_variant(
            "Isaac-Galileo-CTS-v0",
            variant="eval",
            registered_tasks=registered,
        )

        self.assertEqual(resolved, "Isaac-Galileo-CTS-Eval-v0")

    def test_resolve_task_variant_keeps_original_when_no_match_exists(self) -> None:
        registered = {"Isaac-Galileo-CTS-v0"}

        resolved = resolve_task_variant(
            "Isaac-Galileo-CTS-v0",
            variant="eval",
            registered_tasks=registered,
        )

        self.assertEqual(resolved, "Isaac-Galileo-CTS-v0")

    def test_build_evaluation_output_path_nests_task_checkpoint_and_tag(self) -> None:
        path = build_evaluation_output_path(
            "/tmp/run",
            "Isaac-Galileo-CTS-Eval-v0",
            "/tmp/run/model_100.pt",
            summary_tag="seed-1",
        )

        self.assertEqual(
            path,
            Path("/tmp/run/evaluations/isaac_galileo_cts_eval_v0/model_100/seed_1/summary.json"),
        )


if __name__ == "__main__":
    unittest.main()
