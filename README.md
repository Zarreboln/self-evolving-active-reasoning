# Harness 的部分共享:跨任务族的快速适应

**Zining Liu** — Massachusetts Institute of Technology

围绕冻结模型的代码(harness)可以自动进化,但现有方法共享 harness 的边界都是人定的:要么所有任务共用一份,要么按 benchmark 各自一份。
这个项目研究的是:**让数据决定 harness 的哪些组件在哪些任务之间共享**,从而让新任务(包括来自没见过的 benchmark 的任务)适应得更快、更便宜、更不容易过拟合。

## 当前进度

- **官方默认 baseline**:每个任务用它所在 benchmark 自己的默认 agent,原样运行。结果见 [docs/official_baselines.md](docs/official_baselines.md)。
- **对照方法**(进行中):Meta-Harness(按 benchmark 分组、全局两种)、Harness²、ACE、Learning Meta-Skills,都用各自的官方代码。
- **我们的方法**(进行中):组件 = 固定槽位里的内容;任务对组件的隶属度连续可调;交替进化与提取。公式见 [notes/method_formulas.md](notes/method_formulas.md)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `multibench/mb/benches/` | 17 个 benchmark、101 个子任务的统一接口适配器(环境协议照各 benchmark 官方) |
| `multibench/mb/official/` | 各 benchmark 官方默认 agent 的运行脚本(只加接入层) |
| `multibench/mb/evolve/` | harness 进化的基础设施:候选在隔离子进程里运行、静态检查、防抄答案、评测与导出 |
| `sandbox/` | proposer(Claude Code)的 Docker 隔离沙箱:只挂载工作区,只放行 Anthropic API |
| `notes/` | 预注册、实验结果、方法公式、官方 agent 核查、时间表、论文精读笔记 |
| `meta/arbench_mh/` | AR-Bench DC、SP 环境(适配器导入用) |

## 之前的项目

MAS.S62 课程的主动推理项目(Meta-Harness on AR-Bench,与 Yangzi Sun 合作)完整保存在标签
[`mas-s62-ar-report`](../../tree/mas-s62-ar-report) 里。

## 第三方代码

见 [THIRD_PARTY.md](THIRD_PARTY.md)。
