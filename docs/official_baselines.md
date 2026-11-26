# 官方默认基线结果

更新于 2026-11-26。模型:Qwen3.5-9B,关闭思考模式(vLLM,`enable_thinking: false`)。
每个任务都用**它所在 benchmark 自己的默认智能体**,代码原样运行;我们只加了适配代码,让它调用我们的模型、使用我们的数据划分。
每一处适配都写在对应脚本的开头,见 `multibench/mb/official/`。

## 测试集结果(9 个任务)

| 任务 | 默认智能体 | 测试集来源 | 局数 | 得分 | 记为失败的局 |
|---|---|---|---|---|---|
| GN(猜数字) | AR-Bench `zero_shot` | 官方测试集 | 100 | **0.02** | — |
| DC(找凶手) | AR-Bench `zero_shot` | 官方测试集 | 100 | **0.45** | — |
| Wordle basic5 | TextArena 标准智能体 | 我们的划分 | 60 | **0.43** | — |
| Mastermind basic | TextArena 标准智能体 | 我们的划分 | 142 | **0.57** | — |
| Hangman basic | TextArena 标准智能体 | 我们的划分 | 200 | **0.22** | 50 局:模型输出失控,超出上下文长度 |
| 20Q basic-things | TextArena 标准智能体(出题方为 Jev) | 我们的划分 | 49 | **0.06** | 12 局:出题方 Jev 服务报错,**待重跑**(见下) |
| BabyAI GoToLocal | BALROG `naive` | 我们的划分(固定随机种子) | 200 | **0.82** | — |
| ALFWorld pick_and_place_simple | ReAct | 官方 valid_unseen | 24 | **0.58** | — |
| ScienceWorld test-conductivity | SwiftSage 中的 ReAct 基线 | 官方测试集 | 225 | **0.33** | — |

得分是各 benchmark 自己的官方指标,换算到 0 到 1 之间。

## 说明

- **数据划分**:GN、DC、ALFWorld、ScienceWorld 的测试集是官方的;TextArena 和 BabyAI 没有官方划分,我们按答案的哈希值或固定的随机种子划分。
- **失败怎么算**:智能体自己造成的失败(例如输出失控、格式错误)记为失败;服务器、网络、第三方服务出的问题,重新运行。
- **20Q 的 12 局**:最初以为是请求太长被 Jev 拒绝;后来发现 Jev 的服务在同时收到很多请求时会间歇性报错(`400 Unknown model`),短请求也会。这属于第三方服务的问题,已改为失败后等待更久再重试,重跑后更新。
- **Hangman 的 50 局**:TextArena 的标准智能体调用模型时不限制输出长度,模型偶尔会不停重复,写满 64K 的上下文。这是官方默认设置下智能体真实的失败,记为失败。
- **适配代码做了哪些改动**:
  - ALFWorld:原版用的是续写式模型,我们的是对话模型,会在回复开头多写一个 `>`,所以把它去掉;ReAct 示例里旧版 ALFWorld 的 `put X in/on Y`,转换成 ALFWorld 0.4 中对应的 `move X to Y`。
  - ReAct 和 SwiftSage:原版调用的模型(text-davinci-002、gpt-3.5)已经停止服务,改为调用我们的模型,其余设置照原版。
  - SwiftSage 的不推理版本:原版用句向量匹配合法动作,我们改成和推理版本一样的按词重叠匹配。
  - 20Q:TextArena 原版固定用 GPT-4o 当出题方,我们用 Jev。
- **仅供参考的结果(不作为基线)**:AR-Bench 的推理版 `proactive_cot` 跑了一部分后停止:GN 完成的 79 局全对,DC 完成的 36 局对 10 局。

## 进行中

42 个候选任务的官方基线正在运行:每个任务先在搜索集上跑 20 局,用来挑选做进化实验的任务;再跑完整的测试集,用来报告。完成后补进这张表。
