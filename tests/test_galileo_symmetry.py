from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "crl_tasks"
for path in (REPO_ROOT, TASKS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

if importlib.util.find_spec("torch") is not None:
    import torch
else:
    torch = None

if torch is not None:
    from scripts.rsl_rl.algorithms.ppo import PPO
else:
    PPO = None

if torch is not None:
    SYMMETRY_FILE = REPO_ROOT / "crl_tasks" / "crl_tasks" / "tasks" / "galileo" / "config" / "symmetry.py"
    _symmetry_spec = importlib.util.spec_from_file_location("galileo_symmetry", SYMMETRY_FILE)
    if _symmetry_spec is None or _symmetry_spec.loader is None:
        raise RuntimeError(f"Failed to load symmetry module from: {SYMMETRY_FILE}")
    _symmetry_module = importlib.util.module_from_spec(_symmetry_spec)
    _symmetry_spec.loader.exec_module(_symmetry_module)
    galileo_teacher_left_right_augmentation = _symmetry_module.galileo_teacher_left_right_augmentation
else:
    galileo_teacher_left_right_augmentation = None


@unittest.skipIf(torch is None, "torch is not available in the current test environment")
class GalileoSymmetryTest(unittest.TestCase):
    def test_ppo_resolves_string_augmentation_entry(self) -> None:
        func = PPO._resolve_data_augmentation_func(
            "galileo_teacher_left_right_augmentation"
        )
        self.assertTrue(callable(func))

    def test_teacher_left_right_augmentation_mirrors_obs_and_actions(self) -> None:
        obs = torch.zeros(1, 185)
        obs[0, 0:3] = torch.tensor([1.0, 2.0, 3.0])
        obs[0, 3:6] = torch.tensor([4.0, 5.0, 6.0])
        obs[0, 6:9] = torch.tensor([7.0, 8.0, 9.0])
        obs[0, 9:21] = torch.arange(1.0, 13.0)
        obs[0, 21:33] = torch.arange(21.0, 33.0)
        obs[0, 33:45] = torch.arange(41.0, 53.0)
        obs[0, 45:48] = torch.tensor([0.5, 0.6, 0.7])
        obs[0, 48:180] = torch.arange(132.0)
        obs[0, 180:183] = torch.tensor([0.1, 0.2, 0.3])
        obs[0, 183:185] = torch.tensor([0.4, 0.5])

        actions = torch.arange(1.0, 13.0).unsqueeze(0)

        obs_aug, act_aug = galileo_teacher_left_right_augmentation(obs, actions)

        self.assertEqual(obs_aug.shape, (2, 185))
        self.assertEqual(act_aug.shape, (2, 12))
        self.assertTrue(torch.equal(obs_aug[0], obs[0]))
        self.assertTrue(torch.equal(act_aug[0], actions[0]))

        expected_actions = torch.tensor([-2.0, -1.0, -4.0, -3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0])
        self.assertTrue(torch.equal(act_aug[1], expected_actions))

        self.assertTrue(torch.equal(obs_aug[1, 0:3], torch.tensor([1.0, -2.0, 3.0])))
        self.assertTrue(torch.equal(obs_aug[1, 3:6], torch.tensor([-4.0, 5.0, -6.0])))
        self.assertTrue(torch.equal(obs_aug[1, 6:9], torch.tensor([7.0, -8.0, 9.0])))
        self.assertTrue(torch.equal(obs_aug[1, 45:48], torch.tensor([0.5, -0.6, -0.7])))
        self.assertTrue(torch.equal(obs_aug[1, 180:183], torch.tensor([0.1, -0.2, 0.3])))
        self.assertTrue(torch.equal(obs_aug[1, 183:185], torch.tensor([0.4, 0.5])))

        mirrored_scan = obs_aug[1, 48:180].reshape(11, 12)
        original_scan = obs[0, 48:180].reshape(11, 12)
        self.assertTrue(torch.equal(mirrored_scan, original_scan.flip(0)))


if __name__ == "__main__":
    unittest.main()
