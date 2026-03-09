# Contributing

感谢你为 `fppo_ts` 做改进。为了让仓库更适合长期维护和开源协作，请尽量遵循下面的约定。

## Development Setup

```bash
pip install -e .
pip install -e ".[dev]"

cd crl_tasks
pip install --no-build-isolation -e .
```

## Style

- Python 文件名使用 `snake_case`
- 类名使用 `PascalCase`
- 常量使用 `UPPER_SNAKE_CASE`
- 优先修改根因，不做临时性补丁
- 公开 API 变更尽量保持向后兼容

## Local Checks

```bash
python -m black scripts/rsl_rl/runtime.py scripts/rsl_rl/train.py scripts/rsl_rl/play.py \
  scripts/rsl_rl/evaluation.py scripts/rsl_rl/demo.py list_envs.py tests/test_runtime_utils.py

python -m unittest tests.test_runtime_utils -v
```

## Pull Requests

- 描述问题背景、方案和验证方式
- 如果修改训练/评估脚本，说明命令行接口是否变化
- 如果重命名配置或类，保留兼容别名并在说明中注明
