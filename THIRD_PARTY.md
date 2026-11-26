# 第三方代码

## 子模块(固定到具体的提交)

```bash
git clone --recurse-submodules https://github.com/Zarreboln/self-evolving-active-reasoning.git
```

| 目录 | 仓库 | 许可证 | 我们的改动 |
|---|---|---|---|
| `AR-Bench/` | tmlr-group/AR-Bench | 仓库根目录没有许可证文件,所以不复制代码,只用子模块拉取 | 无 |
| `meta_harness_official/` | stanford-iris-lab/meta-harness | MIT | 无 |

## 复制进本仓库的代码

| 位置 | 来源 | 许可证 |
|---|---|---|
| `multibench/mb/vendor/balrog_babyai/` | balrog-ai/BALROG 中 BabyAI 的文字版包装 | MIT(许可证文件在该文件夹内) |

## 运行时用到、但不在本仓库里的

运行官方基线时还用到下面这些项目的代码(装在服务器上,或运行时下载):
TextArena、BALROG、ALFWorld、ScienceWorld、ysymyth/ReAct(提示词文件)、SwiftSage 的 `science_world` 分支(提示词文件和 `eval_utils.py`)。
各自怎么用、改了什么,见 `multibench/mb/official/` 里每个脚本开头的说明,以及 `notes/official_initial_agents.md`。

之前 MAS.S62 主动推理项目用到的其他子模块(BALAR、TTHE、RRSI、EvoHarness-RL、meta-harness-tbench2-artifact)保留在标签 `mas-s62-ar-report` 里。
