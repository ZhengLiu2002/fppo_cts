"""Train Galileo with the main CTS benchmark preset."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


DEFAULT_ARGS = [
    "--task",
    "Isaac-Galileo-CTS-v0",
    "--algo",
    "fppo",
    "--exp",
    "galileo/benchmark/cts_main",
    "--run_name",
    "cts_main",
    "--headless",
]


def main() -> None:
    train_script = Path(__file__).with_name("train.py")
    sys.argv = [str(train_script), *DEFAULT_ARGS, *sys.argv[1:]]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
