# Style Guide

This guide standardizes code style across task configs and scripts.

## Naming

- Modules: `snake_case.py` (e.g. `cts_env_cfg.py`).
- Classes: `PascalCase` (e.g. `GalileoCTSCRLEnvCfg`).
- Constants: `UPPER_SNAKE_CASE` (e.g. `GALILEO_USD_PATH`).
- Config containers: keep stable public names and provide aliases when renaming.

## Config Files

- Keep task defaults in `defaults.py` and reference them from other configs.
- Prefer `GalileoDefaults` (alias of `ConfigSummary`) for shared constants.
- Avoid deep cross-imports between config files; keep dependencies one-directional.

## Docs and Comments

- Use module/class docstrings to describe purpose.
- Comments should explain *why* (not obvious *what*).
- Keep bilingual comments only when necessary for domain-specific context.

## Formatting

- Use `black` with the repo `pyproject.toml` settings.
- Use shared helpers in `scripts/rsl_rl/runtime.py` for script bootstrap, log paths, and checkpoint resolution.
- Format everything with:
```bash
python -m black .
```

## Validation

- Prefer adding small unit tests for utility helpers before touching simulator-heavy flows.
- For lightweight checks introduced in this repo, run:
```bash
./scripts/run_tests.sh -q
```

## Compatibility

- Avoid breaking Gym registration IDs to preserve experiment reproducibility.
