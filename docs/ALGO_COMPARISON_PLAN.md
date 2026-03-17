# Galileo 算法对比实验计划

这份文档把当前研究计划整理成可直接运行的 preset 和命令。

## 1. 总体结构

当前实验分成两条主线：

- **Teacher 对比实验**
- **Student 对比实验**

---

## 2. Teacher 对比实验

### 2.1 公平对比主组

使用同一套 **多奖励、少约束** Teacher 配置的算法：

- `ppo`
- `fppo`
- `np3o`
- `ppo_lagrange`
- `cpo`
- `pcpo`
- `focops`

对应 preset：

- `experiments/galileo/studies/algo_compare_teacher_fair.json`

这套 preset 的含义：

- 使用 Teacher 侧共享奖励项（所有算法共用同一套 gait-tuned reward 参数）
- 只保留三项共享约束：
  - `joint_pos_prob_constraint`
  - `joint_vel_prob_constraint`
  - `joint_torque_prob_constraint`
- 因此 PPO / FPPO / 其他对比算法的 reward / cost 参数保持一致，保证公平对比

### 2.2 NP3O 保留组

为了保留你原来那套“多约束、少奖励”的 NP3O 设置，单独保留：

- `experiments/galileo/studies/algo_compare_teacher_np3o_baseline.json`

它只给 `np3o` 用，方便你比较：

- `np3o` 在公平配置下的表现
- `np3o` 在原始 baseline 配置下的表现

### 2.3 Teacher 训练命令

公平对比主组模板：

```bash
LOG_RUN_NAME=<algo> python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo <algo> \
  --exp galileo/studies/algo_compare_teacher \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: fppo
```bash
LOG_RUN_NAME=fppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo fppo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: ppo
```bash
LOG_RUN_NAME=ppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo ppo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: pcpo
```bash
LOG_RUN_NAME=pcpo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo pcpo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: np3o
```bash
LOG_RUN_NAME=np3o python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo np3o \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```



NP3O baseline：

```bash
LOG_RUN_NAME=np3o_baseline python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo np3o \
  --exp galileo/studies/algo_compare_teacher_np3o_baseline \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

---

## 3. Student 对比实验

### 3.1 Student 公平对比 preset

对应 preset：

- `experiments/galileo/studies/algo_compare_student_fair.json`

适用算法：

- `ppo`
- `fppo`
- `np3o`
- `ppo_lagrange`
- `cpo`
- `pcpo`
- `focops`
- `distillation`

### 3.2 Student 为什么不需要额外 reward/cost 覆写

因为 Student 默认配置本身就是你要的形式：

- `rewards: StudentRewardsCfg = StudentRewardsCfg()`
- `costs: StudentCostsCfg = StudentCostsCfg()`

对应位置：

- `crl_tasks/crl_tasks/tasks/galileo/config/student_env_cfg.py`
- `crl_tasks/crl_tasks/tasks/galileo/config/mdp_cfg.py`

也就是说，Student 默认就是：

- 相对密集的 reward
- 只有三项 joint feasibility constraints

所以这个 preset 的主要作用是：

- 固定实验命名
- 给 student 算法对比统一归档
- 避免以后你自己手动写很多重复命令

### 3.3 Student 训练命令

Student RL 算法模板：

```bash
LOG_RUN_NAME=<algo> python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Student-v0 \
  --algo <algo> \
  --exp galileo/studies/algo_compare_student_fair \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name student \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

例如 `fppo`：

```bash
LOG_RUN_NAME=fppo_student python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Student-v0 \
  --algo fppo \
  --exp galileo/studies/algo_compare_student_fair \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name student \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

蒸馏版（更适合当前 blind student 跟踪微调）：

```bash
LOG_RUN_NAME=fppo_student_distill python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Student-v0 \
  --algo distillation \
  --exp galileo/studies/algo_compare_student_distill_tuned \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name student_distill \
  --headless \
  --checkpoint logs/rsl_rl/galileo_algo_compare_teacher_fair/fppo_2026-03-14_14-40-40_galileo-studies-algo-compare-teacher-fair_teacher/model_15000.pt \
  --logger wandb \
  --log_project_name galileo_teacher

```



### 3.4 Distillation 说明

`distillation` 也可以复用这个 Student preset，但它和普通 RL 算法不同：

- 需要 teacher checkpoint
- 训练时需要配合 `--load_run` / `--checkpoint` 使用
- 如果你想专门改善 student 的指令跟踪，推荐使用
  `experiments/galileo/studies/algo_compare_student_distill_tuned.json`
  这个 tuned preset。它只改 student distillation 相关参数，不影响 teacher preset。

因此更推荐把它看成 Student 侧的单独子实验。

---

## 4. 实验开始前怎么检查 preset 是否真的生效

训练时脚本会：

1. 先读取 task 默认配置
2. 再应用 `--algo`
3. 再应用 `--exp`
4. 最后把最终结果写入日志目录

训练后重点看：

- `logs/rsl_rl/<experiment>/<run>/params/env.yaml`
- `logs/rsl_rl/<experiment>/<run>/params/agent.yaml`
- `logs/rsl_rl/<experiment>/<run>/params/experiment.json`

其中 `env.yaml` 是本次训练真正生效的 reward/cost 配置。

---

## 5. 推荐执行顺序

### 第一步：先 smoke test

先把 `max_iterations` 改成 `1000`，每个算法快速跑一下，确认：

- 没有配置报错
- 日志目录命名正常
- `params/env.yaml` 和 `params/experiment.json` 正常生成

### 第二步：再跑正式实验

正式实验统一使用：

- 相同 task
- 相同 `num_envs`
- 相同 `max_iterations`
- 相同 wandb project
- 不同 `LOG_RUN_NAME`

### 第三步：补多 seed

建议对表现最好的几种算法再补 `3` 个 seed。
