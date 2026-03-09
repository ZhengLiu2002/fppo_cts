# 实验版本管理与 Git 工作流

这份文档对应当前仓库已经实现的实验版本系统，目标是把两类“版本”分开管理：

- **代码版本**：奖励函数、观测结构、算法实现、训练流程改动。
- **实验版本**：在同一套代码上改超参数、terrain、扰动、约束阈值、训练规模等。

结论先说：

- **只改参数**：新增 `experiments/` 下的 preset 文件。
- **改代码逻辑**：新建 git 分支。
- **同时跑多个代码版本**：用 `git worktree`，不要复制整仓库。

---

## 一、当前仓库里已经有什么

现在训练脚本支持：

- `--exp <name>`：从 `experiments/` 里加载 preset。
- `--exp-file <path>`：加载显式 preset 文件。
- `--list-exp`：列出当前可用 preset。

训练时会：

- 先加载基线配置。
- 再应用算法选择（如 `--algo fppo`）。
- 再应用 experiment preset。
- 最后重新应用强优先级 CLI 参数（如 `--run_name`、`--experiment_name`、`--seed`）。

同时训练目录下还会额外保存：

- `params/env.yaml`
- `params/agent.yaml`
- `params/experiment.json`

这样你回头看日志时，能知道那次实验到底改了什么。

---

## 二、推荐目录规范

推荐按“实验主题”组织 preset：

```text
experiments/
  galileo/
    base.json
    fppo_smoke.json
    cost_limit_relaxed.json
    terrain_hard.json
```

推荐命名方式：

- `cost_limit_relaxed`
- `cost_limit_strict`
- `reward_no_slip_penalty`
- `terrain_hard`
- `student_delay_hist12`

不要把文件名起成 `test1`、`test2`、`final2_real` 这种后期根本看不懂的名字。

---

## 三、最常用的训练方式

### 1. 查看可用实验 preset

```bash
python scripts/rsl_rl/train.py --list-exp
```

### 2. 用 preset 跑 teacher

```bash
LOG_RUN_NAME=fppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --exp galileo/cost_limit_relaxed \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_fppo
```

### 3. 用 preset 跑 student

```bash
LOG_RUN_NAME=fppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Student-v0 \
  --exp galileo/cost_limit_relaxed \
  --run_name student \
  --headless \
  --logger wandb \
  --log_project_name galileo_fppo
```

### 4. CLI 和 preset 冲突时谁优先？

优先级从低到高：

1. 基线配置
2. `--algo` 触发的算法默认 profile
3. experiment preset
4. 显式 CLI 参数（如 `--run_name`、`--experiment_name`、`--seed`）

所以：

- 你可以在 preset 里写 `agent.algorithm.cost_limit`。
- 也可以在命令行上继续指定 `--run_name teacher`。

---

## 四、什么时候该用 Git 分支，什么时候只用 preset

### 只用 preset 的情况

如果你改的是这些内容，**不要开新代码分支**：

- `num_envs`
- `max_iterations`
- `cost_limit`
- `step_size`
- terrain 难度
- 随机扰动范围
- 课程学习阈值
- 某个 reward/cost 系数

这些都应该变成 `experiments/*.json`。

### 要开分支的情况

如果你改的是这些内容，就该开分支：

- 新 reward term
- 改 observation 维度
- 改 actor/critic 结构
- 改算法更新公式
- 改 rollout / storage / advantage 逻辑
- 改训练入口流程

---

## 五、给你的最实用 Git 教程

下面只讲你现在最需要的命令。

### 1. 先看当前状态

```bash
git status
```

你要养成习惯：开始改代码前和准备提交前，都先看一次。

### 2. 看自己改了什么

```bash
git diff
```

如果只想看某个文件：

```bash
git diff scripts/rsl_rl/train.py
```

### 3. 提交当前改动

```bash
git add scripts/rsl_rl/train.py experiments/galileo/cost_limit_relaxed.json
git commit -m "Add cost-limit relaxed experiment preset"
```

建议 commit message 用英文短句，方便以后查历史。

### 4. 给当前基线打一个 tag

如果你确认“现在这套代码就是基线”，建议立刻打 tag：

```bash
git tag baseline/2026-03-08
```

以后你就能随时回到这个点。

查看所有 tag：

```bash
git tag
```

### 5. 新建一个代码实验分支

比如你想试一个新的 reward 设计：

```bash
git switch -c exp/reward-slip-penalty
```

这条命令的意思是：

- 新建一个分支
- 名字叫 `exp/reward-slip-penalty`
- 并立即切换过去

### 6. 切回基线分支

先看看当前分支：

```bash
git branch --show-current
```

切回主分支（如果你的主分支叫 `main`）：

```bash
git switch main
```

如果主分支叫别的名字，就换成对应分支名。

### 7. 放弃尚未提交的改动

如果只是想撤销某个文件的本地修改：

```bash
git restore scripts/rsl_rl/train.py
```

如果想撤销所有未提交改动：

```bash
git restore .
```

这个命令会丢掉本地未提交修改，执行前一定先 `git status`。

### 8. 查看提交历史

```bash
git log --oneline --decorate -n 15
```

这条命令非常好用，能快速看到最近 15 次提交。

---

## 六、为什么你应该学会 `git worktree`

如果你同时有多个“代码版本”要跑，不要复制三份仓库。

你应该这样做：

### 1. 在当前仓库里创建一个新分支

```bash
git switch -c exp/new-reward-v2
```

### 2. 回到主分支

```bash
git switch main
```

### 3. 创建一个独立工作目录

```bash
git worktree add ../fppo_ts_new_reward exp/new-reward-v2
```

这会在上一级目录创建一个新文件夹：

- `../fppo_ts_new_reward`

这个文件夹对应：

- 同一个 git 仓库
- 但独立的工作目录
- 固定在分支 `exp/new-reward-v2`

优点：

- 不用复制一大堆日志和缓存
- 不会把两个实验分支改混
- 可以同时开两个终端分别跑不同版本

查看 worktree：

```bash
git worktree list
```

删除 worktree（先确保不再使用）：

```bash
git worktree remove ../fppo_ts_new_reward
```

---

## 七、你接下来最推荐的实际工作流

### 参数实验

1. 保持主代码不动。
2. 在 `experiments/galileo/` 新建一个 preset。
3. 用 `--exp` 跑实验。
4. 结果好就保留 preset，不好就删 preset。

### 代码实验

1. 先打 baseline tag。
2. 新建 `exp/...` 分支。
3. 如果要长期并行跑，用 `git worktree add`。
4. 每完成一个小逻辑就 commit 一次。

---

## 八、我建议你立刻执行的命令

把当前代码当基线的话，我建议你现在就执行：

```bash
git status
git add .
git commit -m "Add experiment preset management system"
git tag baseline/2026-03-08-exp-system
```

然后以后：

- 参数试验：只加 `experiments/*.json`
- 代码试验：`git switch -c exp/<name>`

这样你的实验会很快变得非常整洁。
