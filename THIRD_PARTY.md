# 第三方代码

## 子模块(固定到具体 commit)

```bash
git clone --recurse-submodules https://github.com/Zarreboln/self-evolving-active-reasoning.git
```

| 目录 | 仓库 | 许可证 | 我们的改动 |
|---|---|---|---|
| `AR-Bench/` | tmlr-group/AR-Bench | 顶层无许可证文件,所以不复制代码,只用子模块拉取 | 无 |
| `meta_harness_official/` | stanford-iris-lab/meta-harness | MIT | 无 |

## 复制进仓库的代码

| 位置 | 来源 | 许可证 |
|---|---|---|
| `multibench/mb/vendor/balrog_babyai/` | balrog-ai/BALROG 的 BabyAI 文本包装 | MIT(文件夹内附) |

## 运行时使用、不在仓库里的

官方 baseline 运行时还用到这些项目的代码(安装在服务器上,或运行时下载):
TextArena、BALROG、ALFWorld、ScienceWorld、ysymyth/ReAct(提示文件)、SwiftSage `science_world` 分支(提示文件和 `eval_utils.py`)。
各自的用法和改动见 `multibench/mb/official/` 里每个脚本的开头说明,以及 `notes/official_initial_agents.md`。

之前 MAS.S62 主动推理项目用到的其他子模块(BALAR、TTHE、RRSI、EvoHarness-RL、meta-harness-tbench2-artifact),保留在标签 `mas-s62-ar-report` 里。
