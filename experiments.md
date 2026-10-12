# Stage 0 实验记录

> 2026-10-12 起指标一律用官方叫法：`circuit-wise accuracy`（论文 Table 1 唯一指标，normalization 未定义，这里是原始值）、`joint truth table correct`（整张 24 位 joint truth table 全对，论文的名词、我们对"成功"的字面读法）；uniq@end / eff / zero% 等是我们自己的诊断量。本文件是流水账，旧表只改了列名，列序未动；见 experiments/OFFICIAL_DEFINITIONS.md。

## 2026-10-06 · Stage 0 仪器搭建与零成本验证
- 起止：2026-10-06 约 10:29–10:55（本地），各脚本耗时见下
- 环境：mac 本地 python3 + numpy 2.2.1，纯 CPU，无模型调用，花费 $0
- 代码：`stage0/gn_oracle.py`（可行集 + oracle + 信息效率）、`stage0/selftest_gn.py`、`stage0/probe_synthetic.py`
- 数据：AR-Bench `data/gn/test.json`（100 条）；假设空间 |H| = 5040（4 位、数字互异、允许前导零）

### selftest_gn.py（29.0s）
- 反馈函数与 `arbench.utils.utils_gn.compare_guess` 一致：20000/20000
- 100/100 测试 secret 在空间内；其中 9 条有前导零 → 空间必须取 5040，不是 4536
  （AR-Bench 自己的 `generate_unique_four_digit_number` 强制首位非 0，与它自己的数据和 prompt 矛盾）
- **max-entropy oracle：mean K = 5.22，median 5，min 3，max 7，25 轮内 0 失败**
  直方图 {3:1, 4:13, 5:50, 6:35, 7:1}
- **random-consistent 空策略：mean 5.35，max 8，300 次全成功**

### probe_synthetic.py（29.5s，40 条 secret，seed 0）
| agent | solved | turns | dead% | eff | eff\|alive |
|---|---|---|---|---|---|
| oracle | 40/40 | 5.33 | 15.5% | 1.000 | 1.000 |
| consistent | 40/40 | 5.47 | 0.0% | 0.936 | 0.936 |
| sloppy q=.2 | 40/40 | 5.78 | 17.3% | 0.895 | 0.927 |
| sloppy q=.5 | 40/40 | 6.20 | 37.9% | 0.861 | 0.948 |
| sloppy q=.8 | 40/40 | 8.70 | 62.4% | 0.776 | 0.982 |
| sloppy q=1. | 0/40 | — | 94.7% | 0.691 | 1.000 |
| **anchored** | **0/40** | — | 96.0% | **0.265** | 1.000 |

eff = ex-ante 信息效率（所选猜测的期望比特 ÷ 最优猜测的期望比特），[0,1]，不含运气。

### 结论
1. 仪器有效：eff 对已知病理单调分离（1.000→0.265）。
2. **dead_guess 不是病理标记**——max-entropy oracle 自己有 15.5% 的不可行猜测（经典 Mastermind 分裂手）。
   只有「不可行 + 低信息量」才是病理。
3. GN 在可行猜测之间几乎无策略空间（consistent 5.47 vs oracle 5.33）→ 实例级对抗选择在 GN 上没有杠杆
   （K 只在 3–7，预算 25）。对抗者必须操纵信息结构，不能只挑 secret。
4. **唯一的「可行但低信息」模式 = anchored（逐位替换），eff 0.265、0/40 全败——而这正是 AR-Bench prompt
   第 3 步明确教的策略。** BTR 在 GN 上的具体机制大概就是这个。
5. random-consistent 100/100 → 任何在 25 轮内失败的 agent 都低于平凡空策略，失败是粗放的不是边际的。

### 待办（需要 provider key）
- 复现 T3 Table 1 的 Qwen2.5-7B 一格，验 SiliconFlow 权重/精度
- 跑真实 agent 轨迹，量 eff 分布与 anchored 模式占比

## 2026-10-06 · CircuitDecoding Stage 0（纯 CPU，$0）
- 数据：`arianhosseini/mt_puzzles` circuit_decoding/test.jsonl，300 条，CC BY 4.0（T3 用的就是这份 → 可比性成立）
- 代码：`stage0/cd_oracle.py`

### 结构（已验证）
C=3 电路 × k=3 输入 = 24 位联合真值表；每电路恰好 k-1=2 个二元门（AND/OR 消二产一），NOT 原地替换不改信号数，
故二元门恒为 6。位序 = 行 000..111、x[0] 为最高位——**900 个电路的真值表与存储逐位一致**。
只有三种组件配置，各 100 条：{AND1,OR5,NOT2} / {AND3,OR3,NOT0} / {AND3,OR3,NOT1}。

### 假设空间（枚举，瞬时）
| 配置 | \|H\| | log2\|H\| |
|---|---|---|
| AND3 OR3 NOT0 | 252 | 7.98 |
| AND1 OR5 NOT2 | 2,886 | 11.49 |
| AND3 OR3 NOT1 | 3,924 | 11.94 |

**300/300 条真值落在枚举出的假设空间内** → 枚举正确。预算 18 次单 bit 查询，信息需求最多 11.94 bit，富余约 6 次。

### 策略纵深（300 条，18 次预算）
| 策略 | 解出 | 平均查询 |
|---|---|---|
| greedy 最大熵 oracle | **300/300** | 11.60（≈ 信息下界，近最优） |
| random（可重复） | **35/300** | — |
| sequential（0..23 顺扫） | **58/300** | — |

### 仪器读数（前 60 条）
| 策略 | 解出 | 查询数 | ex-ante eff | **零信息查询占比** |
|---|---|---|---|---|
| oracle | 60/60 | 13.18 | 1.000 | 0.0% |
| random | 6/60 | 17.87 | 0.511 | **39.1%** |
| sequential | 13/60 | 17.30 | 0.583 | **32.1%** |

### 结论：CD 取代 GN 当主测试床
- **GN**：oracle 5.22 vs 空策略 5.35，两者都 100% 解出 → 可行动作之间几乎无差别，不存在"合法但无信息"的动作，对抗者也没有实例级杠杆。
- **CD**：oracle 100% vs 空策略 12% → 策略决定成败。**随机策略 39% 的查询信息量恰好为 0**——合法、不矛盾、纯浪费。
  这正是 BTR 所描述的那种动作，在 GN 里根本不存在。效率度量在这里有满量程。
- 论文里 Gemini 2.5 Pro 成功率 32.2%、GPT-4o 用满 19 轮 → 前沿模型落在 random(12%) 与 oracle(100%) 之间且偏近 random，头部空间很大。

## 2026-10-06 · 全族 Stage 0 普查（`stage0/generic.py` + `survey_families.py`，纯 CPU，$0，276s）

### 能上仪器的三族（可枚举隐藏状态 + 程序答题者）
| 族 | \|H\| | \|Q\| | 预算 | oracle | greedy-4 | random-fresh | sequential | random |
|---|---|---|---|---|---|---|---|---|
| GuessNumbers (AR-Bench, 100条) | 5040 | 5040 | 25 | 100% / 1.000 / 0% | 100% / .893 / 0.4% | 100% / .691 / 14.1% | **4%** / .157 / 71.1% | 100% / .734 / 9.2% |
| WordGuess (MTP, 400条) | 40 | 40 | 3 | 100% / 1.000 / 0% | 100% / .931 / 0.3% | 89% / .779 / 6.0% | 90% / .778 / 6.9% | 89% / .775 / 7.0% |
| **CircuitDecoding (MTP, 300条)** | ≤3924 | 24 | 18 | 100% / 1.000 / 0% | **83%** / .775 / 18.1% | **52%** / .629 / 27.7% | **19%** / .540 / 38.1% | **12%** / .480 / 44.1% |
（格子 = 解出率 / 平均 ex-ante 效率 / 零信息动作占比）

**纵深排序 CD ≫ GN > WordGuess。** CD 跨策略落差 100%→12%，零信息动作 0%→44%；
WordGuess 40 词 3 次太松（空策略也有 89%）；GN 只有完全无视反馈的 sequential 才崩（4%，零信息 71%）。

### 排除的族及原因（已读代码/数据确认，非推测）
- **word_chaining**（MTP, 400）：接龙游戏，**没有要推断的隐藏状态**，仪器不适用。
- **twenty_questions**（MTP, 400）：规则里**模型自己选秘密词并负责回答**，被测的是守住承诺状态的自洽性，
  不是查询选择。T3 也没用它。
- **movie_recommendation**（MTP, 1000）= T3 的 PreferenceEstimation：隐藏偏好是 8 维连续向量（1000 条各不相同），
  需先离散化才能当假设集。
- **AR-Bench DC**（100）：|H| = **5 个嫌疑人**（label 0-4 均匀），信息需求仅 2.32 bit 而预算 25 轮 → 信息论上平凡；
  且**答案由 LLM 生成**（`inference(response_agents[...])`），凶手的 `task` 明写 "avoid suspicion... be deceptive"
  → **答题者按设计就会撒谎，可行集过滤失效**。每案 6 条 `key_question` + `keypoint_eval_agents` 是 AR-Bench 自己的
  LLM 评判式过程分。
- **AR-Bench SP**（100）：隐藏状态是自由文本故事（`story_tree`/`bottom`/`surface`），不可枚举。

### 一个值得写进论文的观察
DC 里**对抗性答题者已经存在**（凶手被指示撒谎），但因为反馈是自由文本，**没法测量**；
CD 里可以精确测量，但对抗答题者要自己造。这正好说明"受约束对抗答题者 + 可计算信息效率"这个组合是空的。

## 2026-10-06 · 第一批真实模型结果（Engaging L40S，Qwen2.5-7B-Instruct）
- 起止：12:04 / 12:12（本地 11:50 投，vLLM 起 130s）；配置 bf16, max-model-len 4096, gpu-mem-util 0.85, temp 0.7, top_p 0.7
- 作业 25055416 `gn_gate`（我的 runner，32 并发，900 ep，**368s，0.41s/ep**）
- 作业 25056115 `gn_arb`（AR-Bench 原版 evaluator，16 分片，每 method 约 2 分钟）

### A. 我的 runner（猜中即停）
| 条件 | solved | eff | dead% | ham1% | d&dull% | tok/ep |
|---|---|---|---|---|---|---|
| bare | 6/300 | 0.439 | 95.4% | 20.0% | 53.0% | 199 |
| taught_incremental | 4/300 | **0.183** | 95.7% | 11.6% | **83.4%** | 199 |
| ~~cot~~ | ~~5/300~~ | ~~0.458~~ | — | — | — | 199 |
⚠ **cot 行作废**：AR-Bench 的 proactive_cot 走独立函数 `_run_proactive_cot_evaluation`、每轮换 user 模板；
`METHOD_DICT["proactive_cot"]` 只是占位符。我只换 system prompt，每轮 user 仍是 "do not give any other statement" → CoT 被掐死（tok/ep 与 bare 相同）。

### B. AR-Bench 原版口径（跑满 25 轮，只看最后一次猜测）+ 我的离线仪器
| log | correct | hit@any | eff | dead% | zero% |
|---|---|---|---|---|---|
| zero_shot | **0/96** | 1/96 | 0.498 | 88.9% | 32.4% |
| few_shot_inst | **1/100** | 1/100 | **0.183** | 95.0% | **69.7%** |
| *oracle* | — | — | 1.000 | 15.5% | 0.0% |
| *random* | — | — | 0.734 | — | 9.2% |
| *sequential* | — | — | 0.157 | — | 71.1% |

### 结论
1. **eff 0.183 在两套循环下完全复现** → 那条 few_shot_inst 指令的伤害与协议无关，是真实效应。
   读那条 prompt 的模型 zero% 69.7%，已贴上「完全无视反馈顺序枚举」的 71.1%。
2. **主失效 = 信念跟踪**：dead% 88.9–95.7%，且在三个条件下几乎不变（与 prompt 无关）。
3. **机制已定位**（`runner/diagnose.py`）：few_shot_inst 下汉明距离 2 占 **49.2%**（对换两位），不是逐位替换（ham1 反降）；
   重复猜测 10.1%→23.4%，每局不同猜测 22.26→18.86。抓的是 instruction 第 5 步而非第 3 步。
4. **「仅低信息（可行但无信息）」= 0.0%** → GN 里不存在 BTR 式动作，病理全在不可行类。换 CD 的决定得到真实数据支持。

### 排除的替代解释（用户质疑后实测）
**不是上下文被截断**：他们的 `propose_agent` 每轮重发完整对话；一局 77 条消息、反馈出现 25 次；
最长 transcript zero_shot 1822 / few_shot_inst 2769 tokens，max-model-len 4096，vLLM 上下文超限告警 **0 次**。
示例局 26 次猜测里 7–9 次是重复的——**历史就在眼前，模型照样重猜已排除的候选**。

### 已知缺陷
- **AR-Bench 自身 bug**：`extract_and_convert_guess` 在模型未输出 4 位数时返回空列表，`compare_guess` 随即
  `raise ValueError("Lists must have the same length")`；他们的 try 只捕 `NotNumberError`（该函数从不抛它）→ 整个分片崩。
  zero_shot 因此丢 4 条（idx 37/53/69/85）。**这会系统性删掉模型表现最差的实例，使样本被净化。** 是否打补丁待定。
- `hit@any` 只有 1/96 → 之前说"撞对了守不住"是我推过头了，两个口径下都接近零，协议差异比我说的小。

## 2026-10-09 · 判官校准 + CD 第二次真实模型结果（H100 GPU3，Qwen2.5-7B-Instruct）
> H100 这几天封存（用户 10-10 指示），后续全部走 Engaging。

### A. SP 判官准确率（TurtleBench 固定 150 条，seed 7，0-shot，`runner/turtle_judge_eval.py`）
| 判官 | 准确率 | 备注 |
|---|---|---|
| Qwen2.5-7B-Instruct（我们之前所有 SP 数字用的） | **50.5%** | 几乎不说 No（No 召回 23.8%）→ 之前 SP 的 f1_char 0.659 建立在坏反馈上，作废 |
| Llama-3.1-405B（AR-Bench 官方判官，0-shot，TurtleBench 自带预测） | 82.7% | |
| Llama-3.1-70B（2-shot） | 74.0% | |
| GPT-4o（2-shot） | 63.3% | |
| **Claude Opus 5**（每题一个独立子 agent，0-shot，`results/labels_opus.json`） | **88.0%** | 唯一弱项 Unknown 召回 47.4% |
结论：SP 要么换判官（405B-INT4 / Opus 5）要么不报。

### B. CD bare vs belief_injected，300 实例 × 1 seed，18 次预算（`results/gate_cd.jsonl`，老 runner，无时间戳）
| 条件 | joint truth table correct | uniq@end | eff | zero% | repeat% | illegal% | parsed% | alive_end |
|---|---|---|---|---|---|---|---|---|
| bare | 1/300 | 101/300（33.7%） | 0.629 | 32.6% | 4.1% | 0.3% | 86.7% | 3.04 |
| belief_injected（enum15） | 20/300 | 78/300（26.0%） | 0.613 | 32.7% | 12.1% | 2.7% | 98.7% | 3.63 |
读法：joint truth table correct 的 +19 大半来自结尾抄枚举表（parsed 86.7→98.7、alive_end 反升），每轮查询反而更差。与 10-07 L40S 那次（2→15，109→100，0.631→0.637）方向一致。
T3 代理（上一轮零信息/重复/集合不缩）在该日志上 precision 0.74 / recall 0.32（`runner/calibrate_t3_proxy.py`）。

## 2026-10-10 · Meta-Harness 搬到 CD：接口、切分、自测、首轮启动（纯 CPU + Engaging）
- 官方代码 `meta_harness_official/`（stanford-iris-lab/meta-harness @ 8123cca）；工作区 `meta/cd_evolve/`；过程流水账 `meta/journal.md`。
- 切分：`splits/cd_split.json`（seed 20261010）search 50 = 17/17/16（按库存分层），heldout 250。
- 自测（20:1x，6s，`results/cd_mock/`，7 harness × 4 脚本策略 × search 50）：
  oracle 策略下全部 harness uniq@end 100%、eff 1.000；sequential 20.0%、eff 0.535（合成参考 19.3%/0.540）；
  follow 策略只在 oracle_hint 下 100%（oracle 一致率 94.5%）；flaky 非法轮 22.2%。仪器与旧 oracle 一致。
- 与旧 runner 的两处有意差别：第 18 个答案随最终提示发出（旧 runner 从不告诉模型第 18 个答案）；每轮记录 sent + model_text。
- Engaging 作业（node3102，各 1×L40S）：
  - `cd_ladder` 25535085：19:55 起（vLLM 170s 就绪，bare 19:57:57 开跑，3.9 s/局，并发 48）；7 harness × 300 × 2 seeds。结果 → `~/results/ar_harness/cd_ladder/`。
  - `serve7b` 25535086：19:55 起，endpoint node3102:8448，供进化循环用。
- 进化循环首轮 `open1`（regime open）20:02:26 启动（`meta/cd_evolve/logs/open1/`）。
- 20:30 **协议纠错**：MTP 论文附录给出 CD 官方对话格式（`A(0, 0, 0) = 1` / `Please send your next query. 17 queries remaining.`；第 18 答随最终提示发）。此前 runner 的环境话术是自拟的，全部 CD 数字（含 gate_cd.jsonl 与 cd_ladder_oldenv）标为"旧话术"，之后一律用官方话术重跑。重提 cd_ladder、重启 open1。
- 20:41 **官方话术下 CD 基线（Engaging L40S，Qwen2.5-7B，search 50 × 2 seeds，18 次）**：
  | 条件 | uniq@end | 95% CI | joint truth table correct | eff | zero% | repeat% | illegal% | parsed% |
  |---|---|---|---|---|---|---|---|---|
  | bare | 26.0% | [18,35] | 0.0% | 0.558 | 36.9% | 1.8% | 0.3% | 86% |
  | enum15 | **53.0%** | [43,63] | 5.0% | 0.629 | 29.1% | 3.5% | 0.2% | 97% |
  同切分旧话术：bare 29%、enum15 32%（rep 8.1%、ill 1.7%）。→ 旧结论"信念注入让 uniq@end 反降"被官方协议推翻一半：手写 harness 翻倍 uniq@end，但 eff 仍在 random-fresh（0.629）水平。gate_cd.jsonl（H100，旧话术）仅作历史记录。
- 21:30 **判官校准补一行（Engaging node5100，2×H200，PP=2，vLLM 0.31）**：Llama-3.1-405B-Instruct-AWQ-INT4（hugging-quants）TurtleBench 150 0-shot **82.0%**（Correct 46/55、Incorrect 71/76、Unknown 6/19），与官方 bf16 405B 的 82.7% 持平 → **SP/DC 正式判官定为 405B-INT4**（用户 21:xx 拍板）。两个坑：TP=2 在 H200 触发 FlashInfer trtllm_mnnvl JIT 失败（nvcc 13.4 头不兼容）→ 改 PP=2；checkpoint 的 generation_config 只有 eos 128001，无 <|eot_id|> → 请求里加 stop_token_ids [128001,128008,128009]，否则答完标签继续生成"assistant\n\n## Step 1"。
- 21:36 **SP bare 基线换正式判官**（提问器 Qwen2.5-7B @ node3600，答题者+要点判官 405B-INT4 @ node5100，100 题 × 2 seeds，25 轮，`runner/sp_runner_async.py`，起止 21:29–21:36）：f1_char **0.621**，f1_word 0.244，err 0；答题者回答分布 no 48% / yes 29% / unknown 23%（旧 7B 自判时几乎不说 no）。对比旧 7B 自判 bare：0.659 / 0.242。→ SP 的旧数字作废，以后 SP 一律用这套；belief_injected（读 bottom+keypoints）不再跑，待改成只读 transcript 的版本。

## 2026-10-10 · 进化循环首轮结果（官方话术，Qwen2.5-7B @ Engaging L40S，search 切分，每候选 ×2 seeds）
### open1（regime open，v1 search 50，20:30–21:07，proposer 866s / $3.90）
| 系统 | uniq@end | 95% CI | joint truth table correct | eff | zero% | oracle% | rep% | ill% | tok_out |
|---|---|---|---|---|---|---|---|---|---|
| bare | 26.0 | [18,35] | 0.0 | 0.558 | 36.9 | 27.3 | 1.8 | 0.3 | 2084 |
| enum15 | 53.0 | [43,63] | 5.0 | 0.629 | 29.1 | 30.6 | 3.5 | 0.2 | 2879 |
| copy_query（proposer：只给一条可抄的最优查询） | **100.0** | [100,100] | 81.0 | 0.962 | 2.8 | 94.5 | 0.0 | 8.3 | 972 |
| copy_open（同上 + 未定输出清单） | 99.0 | [97,100] | 74.0 | 0.962 | 2.8 | 94.5 | 0.0 | 8.6 | 1231 |
| open_entries（enum15 的正面清单） | 32.0 | [24,41] | 47.0 | 0.577 | 34.6 | 30.0 | 0.7 | 0.3 | 1726 |
结论：决策级提示一轮到顶（模型当抄写员，oracle 一致率 94.5%）；状态级改写对信息收集无效。open1 已冻结，held-out 250×3 终评进行中。

### so1（regime state_only，v2 search 100，21:22–22:55，proposer 1346s / $6.69，critic 三候选全部通过，δ=0.080）
| 系统 | uniq@end | joint truth table correct | eff | rep% | ill% | tok_out | 选择 |
|---|---|---|---|---|---|---|---|
| bare | 19.5 | 0.0 | 0.550 | 2.2 | 0.4 | 2200 | 种子 |
| enum15 | 38.0 | 6.0 | 0.604 | 4.4 | 0.3 | 2904 | 种子 |
| veto_determined（enum15 + 打回"已确定/重复"的查询） | 93.5 | 76.5 | 0.880 | 0 | 1.0 | 4437 | 可接纳（+0.555，成本 +27%） |
| **veto_only**（bare + 只在模型要浪费查询时打回并给"未定输出"清单；终猜前列剩余配置） | **97.5** | **83.5** | 0.899 | 0 | 0.1 | 2882 | **frontier**（带内，省 49% token） |
| answer_grid（enum15 的网格呈现） | 86.5 | 82.5 | 0.746 | 1.4 | 2.3 | 3966 | 低于底线 |
结论：**不算任何信息量、不推荐任何查询**，只靠"把已确定/重复的查询打回去"这一条集合成员判断，7B 从 19.5% 到 97.5%、joint truth table correct 从 0 到 83.5%。CD 上 7B 的失败几乎全是信念跟踪（忘了自己已经知道什么），不是规划。so1 已冻结，held-out 200×3 终评进行中。
- 阶梯作业副作用：`splits` 在 12k 上下文下 61% 的局超限（425/600 时 err=259），该行作废，需用 16k 服务器重跑；enum15 4% 超限。
### open1 终评（冻结后一次性，v1 held-out 250 × 3 seeds = 750 局/系统，21:13–23:10，serve7b node3102）
| 系统 | uniq@end | 95% CI | joint truth table correct | eff | zero% | oracle% | rep% | ill% | tok_out | vs enum15（配对，uniq@end / joint truth table correct） |
|---|---|---|---|---|---|---|---|---|---|---|
| bare | 22.5 | [19.6,25.6] | 0.3 | 0.550 | 37.0 | 25.0 | 3.0 | 0.2 | 2222 | −12.5 [−16.5,−8.4] / −6.8 |
| enum15 | 35.1 | [31.7,38.3] | 7.1 | 0.609 | 30.6 | 27.0 | 4.8 | 0.3 | 2960 | — |
| copy_query | **99.7** | [99.3,100] | **80.4** | 0.962 | 2.7 | 94.5 | 0.0 | 8.3 | 1129 | +64.7 [+61.2,+68.1] / +73.3 |
| copy_open | 99.7 | [99.3,100] | 68.4 | 0.962 | 2.7 | 94.5 | 0.0 | 8.7 | 1181 | +64.7 / +61.3 |
| open_entries | 34.5 | [31.2,37.9] | 42.3 | 0.582 | 33.7 | 29.1 | 0.8 | 0.7 | 1672 | −0.5 [−4.8,+3.9] / +35.2 |
读法：held-out 复现 search 结论，无过拟合迹象；enum15 的 search 估计（53%，n=100）偏高，held-out 35.1%——50 题切分的抽样噪声，正是 v2 扩到 100 题的理由。结果文件 results/cd_heldout/open1/。

## 2026-10-11 · 阶梯作业结果（cd_ladder 25537286，node3402 1×L40S，20:36–00:28，7 harness × 300 × 2 seeds，官方话术，12k 上下文）
| harness | err% | uniq@end | 95% CI | joint truth table correct | eff | zero% | oracle% | rep% | ill% | parsed% | tok_out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.0 | 23.2 | [19.8,26.8] | 0.5 | 0.553 | 36.7 | 25.4 | 3.2 | 0.3 | 85.7 | 2389 |
| facts (a) | 0.3 | 29.0 | [25.3,32.5] | 0.5 | 0.570 | 34.8 | 26.4 | 3.4 | 0.2 | 86.0 | 2800 |
| symbolic | 1.3 | **41.2** | [37.3,45.0] | 2.2 | 0.612 | 30.1 | 27.9 | 3.9 | 0.2 | 74.3 | 2648 |
| enum15 | 3.2 | 38.7 | [34.7,42.7] | 5.5 | 0.611 | 30.3 | 28.0 | 3.6 | 0.3 | 93.2 | 2772 |
| t3_gate | 1.3 | 36.8 | [32.8,40.7] | **10.2** | 0.612 | 30.4 | 26.7 | 5.2 | 0.6 | 95.8 | 3104 |
| splits (c) | **58.7** | 27.2 | — | 4.0 | 0.720 | 18.8 | 30.9 | 5.8 | 2.2 | 39.7 | 3362 |
| oracle_hint (d) | **20.7** | 78.7 | — | 9.0 | 0.950 | 3.4 | 90.9 | 0.1 | 1.9 | 78.3 | 4077 |
口径：出错（上下文超限）的局算未收敛/未答对；轮级指标只算完成的局。splits 与 oracle_hint 两行因超限作废，已用 16k 服务器重跑（cd_ladder16k，含 enum15 以便对照）。
读法（有效行）：状态级三种（symbolic / enum15 / t3_gate）uniq@end 都在 37–41%，互相在 CI 内，比 bare 高 14–18 点；facts（只复述观测）+6 点、CI 相接；枚举列表对 uniq@end 无额外贡献，但把 joint truth table correct 从 2.2 提到 5.5，门控版 t3_gate 的 joint truth table correct 最高（10.2）。决策级 oracle_hint 即便 21% 的局超限仍到 78.7%、eff 0.95。与进化结果对照：copy_query（只给一条可抄查询，972 tok）与 veto_only（只打回已确定查询）都 ≥97.5%——阶梯里没有一级靠"多给状态"接近这个数。
### so1 终评（冻结后一次性，v2 held-out 200 × 3 seeds = 600 局/系统，22:57–01:0x，serve_7b_pre node3600，16k 上下文）
| 系统 | err% | uniq@end | 95% CI | joint truth table correct | eff | zero% | oracle% | rep% | ill% | tok_out | vs enum15（配对 uniq@end / joint truth table correct） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.0 | 25.2 | [21.7,28.7] | 0.3 | 0.556 | 36.7 | 24.9 | 3.1 | 0.3 | 2347 | −12.5 / −6.2 |
| enum15 | 0.5 | 37.7 | [34.0,41.7] | 6.5 | 0.614 | 29.9 | 26.9 | 4.7 | 0.3 | 3046 | — |
| answer_grid | 0.0 | 90.0 | [87.3,92.3] | 84.7 | 0.763 | 13.4 | 32.7 | 1.2 | 2.5 | 4071 | +52.3 / +78.2 |
| veto_determined | 1.5 | 95.3 | [93.7,96.8] | 75.8 | 0.880 | 0.0 | 35.9 | 0.0 | 1.3 | 4448 | +57.7 / +69.3 |
| **veto_only** | 0.0 | **96.7** | [95.2,98.0] | **86.2** | 0.890 | 0.1 | 43.7 | 0.0 | 0.2 | 2699 | **+59.0 [+55.2,+63.0] / +79.7 [+76.3,+82.8]** |
读法：search 上的 97.5% 在 held-out 守住（96.7%）；veto_only 的 token 与 bare 相当（2699 vs 2347）。state_only 规则下合法（只用集合成员：重复/已确定/无查询），critic 通过。answer_grid 在 held-out 上 90.0%，比 search 的 86.5% 还高——它在 search 上被噪声底线拒掉是保守选择的代价，不影响结论。
### 阶梯 16k 重跑（cd_ladder16k 25547662，01:xx–03:20，1×L40S，16k 上下文，300 × 2 seeds）
| harness | err% | uniq@end | 95% CI | joint truth table correct | eff | zero% | oracle% | tok_out |
|---|---|---|---|---|---|---|---|---|
| enum15 | 0.2 | 41.8 | [38.0,45.8] | 5.8 | 0.612 | 30.4 | 28.7 | 2892 |
| splits (c) | 10.2 | 53.7 | [49.8,57.8] | 5.7 | 0.702 | 20.8 | 32.2 | 4316 |
| oracle_hint (d) | 3.0 | **96.5** | [95.0,97.8] | 6.8 | 0.951 | 3.4 | 91.1 | 4522 |
合并后的阶梯（取各行有效版本，`results/cd_ladder/scores_best.json`，图 `ladder.png`）：bare 23.2 → facts 29.0 → symbolic 41.2 ≈ enum15 41.8 → t3_gate 36.8 → splits 53.7（仍有 10% 超限）→ oracle_hint 96.5。
读法：状态 +18、决策表 +30、决策 +73；oracle_hint 的 joint truth table correct 只有 6.8（没有终猜提示，模型抄查询不抄答案），而进化出的 copy_query（99.7 / 80.4）和 veto_only（96.7 / 86.2）同时解决了两段。

## 2026-10-11 · EvoHarness-RL 当 baseline（官方框架 + 自写 CD 适配器，Qwen2.5-7B @ serve_7b_pre，300 题 × 2 run，06:28–07:36）
框架 github.com/EvoHarnessRL/EvoHarness-RL；适配器 evoharness_rl_official/evoharness/envs/circuitdecoding/；元动作不计 18 次预算（上限 18）；累积对话；温度 0.7；他们的 I/O 格式（OBJECTIVE/OBSERVATION/ADMISSIBLE COMMANDS + <think>/<action>）。回放打分 runner/cd_score_evoharness.py。
| 模式 | run | uniq@end | 95% CI | joint truth table correct | circuit-wise accuracy | eff | zero% | oracle% | rep% | meta/局 |
|---|---|---|---|---|---|---|---|---|---|---|
| env_only（ReAct） | s0 / s1 | 29.3 / 26.0 | [24,34] / [21,31] | 0.0 / 0.0 | 4.3 / 4.9 | 0.577 / 0.569 | 33.5 / 33.9 | 20.2 / 19.3 | 4.9 / 4.1 | 0 |
| always_on（面板每轮贴：可行集计数/已确定/未确定/≤15 全表 + plan） | s0 / s1 | **71.0 / 74.7** | [66,76] / [70,79] | **62.0 / 65.0** | 74.8 / 79.0 | 0.760 / 0.763 | 14.7 / 13.8 | 38.4 / 38.1 | 0.6 / 0.4 | 0 |
| inline（EvoHarness-Base：模型自己调 track/commit/recall/note） | s0 / s1 | 30.0 / 31.7 | [25,35] / [27,37] | 0.7 / 1.7 | 5.7 / 6.9 | 0.577 / 0.586 | 33.3 / 32.7 | 21.7 / 20.8 | 3.8 / 4.6 | track 0.07 / 0.09 |
读法：① ReAct ≈ 我们的 bare（23–25%）。② EvoHarness-Base（inline，拉式）≈ ReAct：冻结的 7B 平均每局只调 0.08 次 track，harness 形同虚设——和他们论文里 Qwen3-8B 冻结 inline 只小涨、要 SFT+GRPO 才起飞一致，也是我之前"pull ≤ push"的预判。③ always_on（推式）71–75%，远高于我们 enum15（38–42%）、低于 veto_only（97%）：同样的信息换一种呈现（面板块 + 未确定清单 + think/action 短格式 + 每轮重贴 OBJECTIVE）效果差一倍，说明 state_only 这一级的"呈现"空间比我们种子显示的大得多（proposer 的 answer_grid 90% 也印证）。
