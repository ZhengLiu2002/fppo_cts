# Algorithms

This repo standardizes constrained-RL algorithm integration for the Galileo CTS benchmark.

## Registry

Algorithms are registered in:
- `scripts/rsl_rl/algorithms/registry.py`

Each algorithm declares:
- `name`: canonical CLI name (e.g. `fppo`)
- `module`: module path (e.g. `scripts.rsl_rl.algorithms.fppo`)
- `class_name`: class to instantiate (e.g. `FPPO`)
- `training_type`: default rollout mode used outside CTS-specific runners
- `config_family`: CTS config schema family used by the Galileo benchmark builder
- `extra_cfg_keys`: optional keys allowed beyond the class `__init__` signature

## CTS Framework

On `Isaac-Galileo-CTS-v0`, CTS is the shared teacher-student framework rather
than a separate optimizer choice:

- the runner sets `framework_type="cts"`
- `--algo ppo`, `--algo fppo`, `--algo ppo_lagrange`, etc. reuse the same
  teacher/student rollout partition and latent-alignment supervision
- the optimizer class only changes the policy update rule

Runner-facing CTS differences are declared by algorithm contracts rather than
hard-coded algorithm-name branches. The shared contract lives in:

- `scripts/rsl_rl/algorithms/contracts.py`

Algorithms inherit the default no-op CTS contract from `PPO`. If an optimizer
needs extra CTS runtime wiring, it can override `cts_runtime_contract` on the
class while the runner remains generic.

The legacy `cts` algorithm name remains as a compatibility alias for
PPO-on-CTS, but new experiments should use the real optimizer name directly.

## Validation

Before instantiation, algorithm configs are validated against the algorithm
constructor signature. Unknown keys raise warnings by default. To make it strict:

```bash
export CRL_ALGO_STRICT=1
```

## Adding a New Algorithm

1. Create the algorithm implementation in `scripts/rsl_rl/algorithms/`.
2. If needed, declare `cts_runtime_contract` on the algorithm class.
3. Register it in `scripts/rsl_rl/algorithms/registry.py` via `register_algorithm`
   and choose the correct `config_family`.
4. Add default hyperparameters in
   `crl_tasks/crl_tasks/tasks/galileo/config/defaults.py` under `GalileoDefaults.algorithm`.
5. Use `--algo <name>` on `Isaac-Galileo-CTS-v0` or set `GalileoDefaults.algorithm.name`.

This keeps CLI choices, config validation, and runner behavior in sync.
