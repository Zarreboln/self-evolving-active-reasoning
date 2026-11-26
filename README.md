# Harness 的部分共享:让新任务更快适应

**Zining Liu** — 麻省理工学院

围绕一个冻结的大模型,外面那层代码(harness,包括提示词、对话历史怎么组织、控制流程、工具等)可以自动改进。
现有方法在多个任务之间共享 harness 时,共享范围都是人事先定好的:要么所有任务共用一份,要么每个 benchmark 各用一份。
这个项目研究的是:**让数据来决定 harness 的哪些组件在哪些任务之间共享**,从而让新任务,包括来自没见过的 benchmark 的任务,适应得更快、花费更少、更不容易过拟合。

## 当前进度

- **官方默认基线**:每个任务都用它所在 benchmark 自己的默认智能体(agent),代码原样运行。结果见 [docs/official_baselines.md](docs/official_baselines.md)。
- **基线方法**(进行中):Meta-Harness(按 benchmark 分组、所有任务共用一份,两种用法)、Harness²、ACE、Learning Meta-Skills,都用各自公开的官方代码。
- **我们的方法**(进行中):把 harness 分成固定的几类组件;每个任务对每个组件有一个连续的权重,决定它用哪些组件、改动的功劳怎么分;进化和共享交替进行。公式和出处见 [notes/method_formulas.md](notes/method_formulas.md)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `multibench/mb/benches/` | 17 个 benchmark、101 个子任务的统一接口(游戏规则和判分照各 benchmark 官方) |
| `multibench/mb/official/` | 运行各 benchmark 官方默认智能体的脚本(只加了适配代码) |
| `multibench/mb/evolve/` | harness 自动改进所需的代码:候选在隔离的子进程里运行、静态检查、防止抄答案、评测和结果导出 |
| `sandbox/` | 负责修改代码的模型(proposer,即 Claude Code)所用的 Docker 隔离环境:只能看到自己的工作目录,只能访问 Anthropic 的接口 |
| `notes/` | 预注册、实验结果、方法公式、官方智能体核查、进度安排、论文阅读笔记 |
| `meta/arbench_mh/` | AR-Bench 的 DC、SP 环境(适配代码会导入) |

## 之前的项目

MAS.S62 课程的主动推理项目(在 AR-Bench 上用 Meta-Harness,与 Yangzi Sun 合作)完整保存在标签
[`mas-s62-ar-report`](../../tree/mas-s62-ar-report) 里。

## 第三方代码

见 [THIRD_PARTY.md](THIRD_PARTY.md)。
