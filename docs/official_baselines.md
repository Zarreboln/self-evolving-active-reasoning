# 官方默认 baseline 结果

更新于 2026-11-26。模型:Qwen3.5-9B,关闭思考模式(vLLM,`enable_thinking: false`)。
每个任务都用**它所在 benchmark 自己的默认 agent**,代码原样运行;我们只加了接入层(指向我们的模型、用我们的数据划分)。
接入层的每一处改动都在对应脚本的开头写明,见 `multibench/mb/official/`。

## test 集结果(9 个任务)

| 任务 | 默认 agent | test 集 | 局数 | 得分 | 按输计入的局 |
|---|---|---|---|---|---|
| GN(猜数字) | AR-Bench `zero_shot` | 官方 test | 100 | **0.02** | — |
| DC(找凶手) | AR-Bench `zero_shot` | 官方 test | 100 | **0.45** | — |
| Wordle basic5 | TextArena 标准 agent | 我们的划分 | 60 | **0.43** | — |
| Mastermind basic | TextArena 标准 agent | 我们的划分 | 142 | **0.57** | — |
| Hangman basic | TextArena 标准 agent | 我们的划分 | 200 | **0.22** | 50 局:模型输出失控,上下文超长 |
| 20Q basic-things | TextArena 标准 agent(出题方为 Jev) | 我们的划分 | 49 | **0.06** | 12 局:出题方 Jev 报错,**待重跑**(见下) |
| BabyAI GoToLocal | BALROG `naive` | 我们的划分(固定种子) | 200 | **0.82** | — |
| ALFWorld pick_and_place_simple | ReAct | 官方 valid_unseen | 24 | **0.58** | — |
| ScienceWorld test-conductivity | SwiftSage 的 ReAct baseline | 官方 test 变体 | 225 | **0.33** | — |

得分是各 benchmark 自己的官方指标,映射到 [0, 1]。

## 说明

- **数据划分**:GN、DC、ALFWorld、ScienceWorld 的 test 集是官方的;TextArena 和 BabyAI 没有官方划分,用的是我们按答案哈希或固定种子划出来的。
- **计分规则**:agent 自己造成的失败(比如输出失控、格式错误)按输计入;基础设施故障(服务器、网络、第三方服务)重跑。
- **20Q 的 12 局**:最初以为是请求过长被 Jev 拒绝,后来发现 JevHub 在并发请求下会间歇性返回 `400 Unknown model`,短请求也会。这属于基础设施问题,已加上指数退避重试,将重跑后更新。
- **Hangman 的 50 局**:TextArena 标准 agent 调用模型时不设 `max_tokens`,模型偶尔陷入重复输出、写满 64K 上下文。这是官方默认设置下 agent 的真实失败,按输计入。
- **接入层的改动**:
  - ALFWorld:去掉对话模型多写的开头 `>`(原版用补全模型);把 ReAct 提示里旧版 ALFWorld 的 `put X in/on Y` 转成 ALFWorld 0.4 的 `move X to Y`。
  - ReAct 和 SwiftSage:原版调用的模型(text-davinci-002、gpt-3.5)已下线,改为调用我们的模型,其余设置照原版。
  - SwiftSage 的 act-only 版本:把 SayCan 的句向量匹配换成了和 ReAct 一样的词重叠匹配。
  - 20Q:TextArena 原版写死 GPT-4o 当出题方,我们用 Jev。
- **参考结果(不作 baseline)**:AR-Bench 的推理版 `proactive_cot` 跑过一部分后停止:GN 79/79 全对,DC 36 局中 10 局对。

## 进行中

42 个候选任务的官方 baseline(搜索集 20 局用于挑任务,test 集用于报告)正在 bath 上运行,完成后补进这份表。
