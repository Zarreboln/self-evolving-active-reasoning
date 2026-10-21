# 以往方法与官方口径参照(AR-Bench · Multi-Turn Puzzles · 2026-10)

<details>
<summary><b>官方任务定义 / 官方指标 / 官方 baseline 数字 · 判官参照</b></summary>

<dl><dd>

> 数据来源:AR-Bench(ACL'25)与 Multi-Turn Puzzles 原论文表格,2026-10-12 逐条核实。口径以仓库
> [`experiments/OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md) 为准。
> ⚠️ 指标只用官方定义与官方叫法;我们自己加的量一律标注 (ours),不得与官方指标混排。

#### 一、官方 baseline(Circuit Decoding,MTP Figure 1)

| 模型 | circuit-wise accuracy | 备注 |
|---|---|---|
| Gemini-2.5-Pro | 39.8 | 官方最强 |
| GPT-4o 档 | 15.4 | |
| 开源档 | 9.3 | |

> MTP Table 1 对 Circuit Decoding 只给一个指标:**normalized circuit-wise accuracy**(归一化方式未定义)。
> 图里的 "Success Rate (%)" 每任务定义不同、原文未写明。我们报 `circuit_wise_accuracy`(官方名词)
> 与 `joint_truth_table_correct`(官方名词 + 我们的字面读法),其余全是 (ours) 诊断量。

#### 二、官方 baseline(AR-Bench Situation Puzzles,Table 6,zero-shot,字符级 F1)

| 模型 | F1 (char) |
|---|---|
| GPT-4o | 63 |
| Llama-3.1-405B | 55 |
| Qwen-2.5-7B | 51 |
| QwQ-32B | 49 |
| Llama-3.1-70B | 54 |
| Llama-3.1-8B | 43 |
| GPT-4o-mini | 41 |

> AR-Bench 三任务官方指标:DC accuracy · SP 字符级 F1(`utils_sp.f1_score`)· GN exact match。
> 官方 referee/judge = Llama-3.1-405B。SP 的 key-question recall 是 **我们加的过程指标**,非官方。

#### 三、判官参照(TurtleBench-150,seed 7,0-shot)

| 判官 | accuracy | 来源 |
|---|---|---|
| Opus 5(每题独立上下文) | 88.0 | 我们跑,`labels_opus.json` |
| Llama-3.1-405B bf16 | 82.7 | TurtleBench 仓库归档预测 |
| **Llama-3.1-405B-INT4(我们在用)** | **82.0 / 83.3(两轮)** | 同口径复现,差异来自 temperature 0.7 采样 |
| Llama-3.1-70B(2-shot) | 74.0 | 归档预测 |
| GPT-4o(2-shot) | 63.3 | 归档预测 |
| Qwen2.5-7B-Instruct | 50.5 | 我们跑;作废早期 SP 自判数 |

> 选 TurtleBench 的理由:**AR-Bench 自己就用它交叉检验 LLM 判官(原论文 Table 13)**,且 TurtleBench
> 的 referee prompt 正是 AR-Bench SP 回答者 prompt 的来源,三分类口径与两个 few-shot 例子一致。
> 这不是我们另找的尺,是沿用原文做法。

</dd></dl>

</details>

> ---

# 实验 Log(可回溯 · 两阶段)

<details>
<summary><b>实验清单(编号 · 日期 · 耗时 · 主结果)· 口径与记录规范</b></summary>

<dl><dd>

| 编号 | 日期 | 耗时 | 主结果 |
|---|---|---|---|
| A1 | 10-12 | — | **官方口径定稿**:CD 官方指标只有 circuit-wise accuracy;joint truth table correct = 官方名词的字面读法;uniq@end/eff/zero%/oracle% 全是 (ours) 诊断。列名与 JSON key 已全仓统一 |
| A2 | 10-06 | 待补(CPU) | **18 个 query 够用**:greedy max-entropy oracle 解 300/300、均 11.60 query;greedy 采样4 250/300;random fresh 156/300;sequential 58/300;uniform random 35/300 → **任务可解,失败归因到策略不是预算** |
| **A3** | 10-10~10-14 | 见子节点 | **判官标定大树**(TurtleBench-150 固定样本)。结论:405B-INT4 **83.3** 当裁判;十一个候选逐题对照无一可平替 |
| A3.1 | 10-13 | 各约 10 min / 1×H200 | 八候选标定:Qwen3.8-Flash-Next 79.3、Qwen3-235B 74.0、Qwen3-32B 74.0、Llama-3.3-70B 73.3、Qwen3.6-35B-A3B 72.7、Qwen3.8-27B 64.0、gpt-oss-120b 62.7 |
| A3.2 | 10-13 | 10 min ×2 | 开思考:Flash-Next 79.3→**80.7**、Qwen3.8-27B 64.0→**78.7**;**均未够到 405B 的 83.3**,裁判不换 |
| **A3.3** | 10-13 | 12 s / $0.0041 | **Jev 1.13(TypeSafe 闭源 System One)当裁判 78.7**;Unknown 召回 **36.8 > 405B 的 26.3**;置信度随正确率单调(conf<0.2 正确 50.0% → conf>0.8 正确 95.6%)。**逐题不是平替**:一致率 80.7%,修好 405B 的 25 错里 10 个、弄坏它对的 125 里 17 个。**不确定性**:同一请求跑五次得 78.7/80.0/79.3/80.0/78.0,**非确定性,两分波动** |
| A3.4 | 10-13 | 各 12 s | Jev 三种官方推荐写法(结构化 criteria / 拆成两问 / 两者合用)**全部用总分换 Unknown 召回**,最好宏平均 70.9 但总分掉 3–5 分 → 与直接调概率阈值等价,没有免费午餐 |
| **A3.5** | 10-13 | — | **宏平均翻案**:金标分布偏(C 55 / I 76 / U 19),几乎不说 Unknown 的裁判在总分上不受罚。按宏平均召回重排:**Flash-Next(think)79.9 第一,405B 69.6 掉到第五**。SP 试点实测裁判对 **13.8%** 的提问答 Unknown(4500 次回答:No 58.5 / Yes 27.6 / Unknown 13.8)→ 约每七问一个落在 405B 最弱的类 |
| **A3.6** | 10-14 | 2 模型各 ~1 min | **读概率(typed read)判决**:约束到三标签读首 token 分布。**从不改变 argmax**(与 greedy 采样差 ≤1 分),只多给分布。**校准属于模型不属于机制**——Qwen2.5-7B ECE 0.56(均值 P 0.945 / 正确率 0.360,分布无信息)、Qwen3-32B 0.16、**Qwen3.8-27B 0.055 最佳且宏平均 69.3 ≈ 405B 的 69.6**。7B 另有单一失败形状:150 题答 99 次 Unknown,**禁掉 Unknown 则 36.0→68.0** |
| A4 | 10-10 | 7 min | **SP 无 harness 基线(405B 当官方级裁判)**:f1_char **0.621**;此前 0.659 是 7B 自判、作废 |
| **A5** | 10-10~11 | 待补 | **可操作性阶梯(七个手写 harness)**:bare cwa 12.9 / joint 0.5 / uniq@end 23.2 → symbolic 20.5/2.2/41.2 → enum15 23.7/5.8/41.8 → **T3-gated 31.3/10.2** → splits 22.3/5.7/**53.7** → **oracle hint 27.3/6.8/96.5**。**uniq@end 与 joint 不同步**:oracle hint 把信息收集拉满(96.5)却只把 joint 抬到 6.8 → **瓶颈在最终推断不在提问** |
| A6 | 10-11 | 1 h 08 | **EvoHarness-RL 当外部 baseline**:env_only(ReAct)cwa 4.3/4.9、inline(EvoHarness-Base,模型自己调 track/commit)5.7/6.9、**always_on(每轮贴 STATE 面板)74.8/79.0 joint 62.0/65.0** → 同样的状态信息,**面板常开 ≫ 让模型自己维护**,与 A5 同向 |
| **A7** | 10-21 | — | **指标审计:官方 F1 在量什么**(500 局存档重分析,无新推理)——**答案词数一个变量解释 F1 的 65.0% 方差(r +0.806)**,key-question recall 只有 r +0.050、控制长度后偏相关 +0.068。分桶单调无拐点(25–50 词 0.458 → 125–150 词 0.868)=**无有效长度惩罚**。**控制长度后 bare→终局 的 +15.2 缩到 +0.4…+7.2**。另:任务不强制最终答案由问答推出(**允许跳步**,degenerate 基线待测,需服务器谜底);referee 自身 83.3% → 环境自带约六分之一回答噪声。⚠️ 首次计算得 r=0.90 是把被抢占那轮的 88 局空答案算了进去,清洗后 0.806,已在文档留痕 |
| **A8** | 10-15~21 | — | **独立复核(Yangzi Sun,预注册)**:固定对话 × 两种 writer 四格,65 题 260 次生成 + 520 次盲评,加 Semantic Correctness 与 Evidence Faithfulness。**82 次 F1/语义严格排名反转(29 题)**;证据约束 writer 未提升语义正确性(−0.0308 [−0.0577, −0.0058])。我们不采纳的两点:它测的是**第一代**(其召回本就持平,终局 57.6→66.4);其数据里**两位评审都判全对的题数 evolved 15/65 对 bare 7/65**,与序数均值反向。报告在 sun-tze fork 的 `sun/c2-confirmatory-65` 分支 |
| B1 | 10-10 20:30–23:10 | 2 h 40 | 进化 `open1`(regime **open**):proposer 一轮到顶,copy_query 在 held-out 上 joint **80.4** / uniq@end 99.7 |
| B2 | 10-10 21:22–10-11 ~01:00 | ~3 h 38 | 进化 `so1`(regime **state_only**,禁止算信息增益/split/推荐 query):veto_only held-out cwa **91.3** / joint **86.2** / uniq@end 96.7,**token 与 bare 持平** → 不靠算力靠约束 |
| **B3** | 10-12 16:14–17:51 | 1 h 37 | **TTHE 试点(strict / labeled 两条流)**:strict 第 0 批**零提示下 proposer 自己重建假设空间**,无标签判官把它选中,第 1–2 批 20/20 joint 全对;流 30 题 prequential cwa 70.0 / joint 66.7。⚠️ **后续批次标签未隔离(审计查出)→ 全部作废重跑** |
| **B4** | 10-12 19:01–22:33 | 3 h 32 | **标签隔离重跑 + 冻结评测**(`strict6` / `labeled6`,控制器私有存储 + 每会话审计)。strict 流 **0/10 → 9/10 → 10/10**,29/32 会话干净、**判官会话全部干净**;冻结 200 题:bare cwa 11.8 / joint 0.0,**无标签 state-only harness 到 cwa 99.5 / joint 98.5(未见 98.2)**;labeled 终局 100/100 @215 tok/ep。判官三次否掉 10/10 的「菜单式」harness,强制「每个 query 由模型自己写」 |
| **B5** | 10-13 05:03–07:17 | 2 h 14 | **移植到 AR-Bench Situation Puzzles(strict,405B 裁判)**:30 题流 prequential F1 **63.7 → 74.5 → 79.9**;冻结 100 题(70 题未见)**bare 60.4 → 进化版 73.0–76.7**,**超过 AR-Bench Table 6 里所有 zero-shot 含 GPT-4o 63**,提问仍全部由 7B 自己写。代价:输出 token 3–8×。未选中的 b2r0g1 用一半 token 打平。21 会话 18 干净(3 次相对路径误计,已修) |
| **B6** | 10-13 | — | **给 TTHE 判官造尺(不需外部基准)**:池内 gold 判官从未看过,拿它和实际提交的候选对账。`strict6` 2/3 命中最优、平均 regret 0.011;`labeled6` 3/3、0.000;**`sp_strict1` 1/3、平均 regret 0.063、最差 0.114(b1 提交了五个里最差的)** → CD 池饱和测不出,**SP 上判官可测地有噪声** |

> 目的:每次实验都能回溯到 **条件(任务/模型/regime/signals/预算/split)→ 脚本 → 输入 → 输出/log → 结果**。
> 标注:📁本地(已核实路径)· ☁️服务器(Engaging `~/results/ar_harness`,H100 `/home/tmp_l`)· 🔒控制器私有(`~/.ar_harness_controller/<run>/`,不入 git)。
> **记录规范**:每个正式实验在 GitHub 与实验表里都把 **跑的时间** 和 **耗时** 放在实验名前面。时间记
> `YYYY-MM-DD`,跨天写 `MM-DD~MM-DD`,同条多次运行用 `日期 + 日期`;具体时刻进各实验 README 不进总表。
> 优先取 log 起止时间;没有 log 时取产物 mtime 并标明是推断。耗时没有可靠记录写 `待补`,**不硬猜**。
> 结果表末列固定为「回溯」= `[出该行数字的脚本]·[log]`。术语一律用英文(harness/proposer/judge/referee、
> prequential/transductive、regime、signals、frozen evaluation、held-out/unseen、calibration、ECE、
> macro recall、typed read、ablation、baseline、audit 等),中文只作叙述连接。
> 🔗 可回溯挂在**各小实验末尾**,只放上游/辅助脚本、产物路径与批量 log,与行级回溯零重复。

> **两阶段框架**:
> - **A1–A8 = 第一步 · 评测基座**——官方口径、任务可解性天花板、判官标定、无 harness baseline、手写
>   harness 阶梯、外部 baseline、**指标本身的审计**、**独立复核**。这一段决定「我们报的数字算不算数」,
>   而 A7 的答案是:**官方 F1 大部分在量答案长度**,所以主表要和 A7 并排读。
> - **B1–B6 = 第二步 · harness 自进化**——一次性进化、测试时进化(TTHE)、标签隔离、跨任务移植、
>   判官自身的质量。这一段决定「机制成不成立」。
>
> **指标口径(全表统一)**:
> - **circuit-wise accuracy** — CD 的**官方唯一指标**,三个电路中完整真值表正确的比例。
> - **joint truth table correct** — 24 位联合真值表全对,官方名词、我们的字面读法。
> - **f1_char** — SP **官方指标**,字符级 F1(`arbench.utils.utils_sp.f1_score`)。
> - **key-question recall** — (ours) SP 过程指标,裁判判定命中的关键问题比例,非官方。
> - **uniq@end / eff / zero% / oracle% / rep%** — (ours) 诊断量,只做机制分析,不与官方指标并列。
> - **prequential** — 新一批到来时、适应之前,**在役 harness** 的成绩(部署视角)。
> - **transductive** — 同一批上**提交后** harness 的成绩(拟合视角)。两者必须分开报。
> - **ECE** — 10 等宽桶的期望校准误差;**macro recall** — 三类召回的算术平均(治金标偏斜)。
>
> **口径阶梯(对外措辞)**:training-free **否**、data-free **也否**(有实例流)。正确词是
> **label-free / transductive**,证据是 strict ≈ labeled。真正的 data-free 要等任务族自生成。

**数据与判官基准**

| 项 | 路径 | 说明 |
|---|---|---|
| CD 实例 | ☁️ `~/code/ar_harness/mt_puzzles/` | 300 released instances,18 query + 最终猜测 |
| SP 实例 | ☁️ `~/code/ar_harness/AR-Bench/data/sp/` | 官方 split;谜底与 key question 为控制器私有 |
| TurtleBench | ☁️ `~/TurtleBench`(两台都有) | 1531 英文案例,固定抽 150(seed 7,32 个故事) |
| 判官逐题预测 | 📁 [`experiments/02_judge_calibration/turtle150_predictions/`](../experiments/02_judge_calibration/turtle150_predictions/) | 十一个候选 + Jev + typed read,每题 `pred`/`gold`(+概率) |
| 控制器私有记录 | 🔒 `~/.ar_harness_controller/<run>/` | solve_jobs、完整 opt_log、lineage;**不入 git** |
| 原始轨迹 | GitHub Release `v0.1-experiments-2026-10-11` | `cd_tthe_isolated_raw` / `cd_tthe_pilots_raw` / `sp_tthe_pilot_raw` |

</dd></dl>

</details>

> ---

# 第一阶段 · 评测基座:口径、天花板、判官、baseline

<details>
<summary><b>A3 判官标定(TurtleBench-150 · 大树)</b></summary>

<dl><dd>

> 被标定的是 **referee**(SP 里回答玩家提问的那个模型),不是 TTHE 里决定提交 harness 的 judge。
> 后者在 [B6](#) 有自己的尺。两者在文中必须分开称呼。

<details>
<summary>A3.1–A3.2 十一候选 × 思考开关(10-13,各约 10 min / 1×H200)</summary>

<dl><dd>

accuracy,后接分类召回 Correct / Incorrect / Unknown(金标 55 / 76 / 19)。

| 候选 | 作业 | 不思考 | C/I/U | 开思考 | C/I/U |
|---|---|---|---|---|---|
| **Llama-3.1-405B-INT4(在用)** | 25698041 | **83.3** | 89.1 / 93.4 / 26.3 | n/a | |
| Qwen3.8-Flash-Next-FP8 | 25718457 / 25724076 | 79.3 | 90.9 / 81.6 / 36.8 | **80.7** | 90.9 / 75.0 / 73.7 |
| Qwen3.8-27B-FP8 | 25718456 / 25724077 | 64.0 | 74.5 / 59.2 / 52.6 | **78.7** | 85.5 / 77.6 / 63.2 |
| Qwen3-235B-A22B-2507-FP8 | 25718459 | 74.0 | 83.6 / 69.7 / 63.2 | 未跑 | |
| Qwen3-32B | 25698272 | 74.0 | 80.0 / 81.6 / 26.3 | 未跑 | |
| Llama-3.3-70B-FP8-dynamic | 25698273 | 73.3 | 76.4 / 84.2 / 21.1 | 未跑 | |
| Qwen3.6-35B-A3B-FP8 | 25718458 | 72.7 | 87.3 / 69.7 / 42.1 | 未跑 | |
| gpt-oss-120b | 25698271 | 62.7 | 61.8 / 65.8 / 52.6 | 未跑 | |

**每个候选都栽在 Unknown**,而 Unknown 正是「这个问题故事根本答不了」的判断。405B 的优势全在
Incorrect(93.4,别人不过 85),它自己的 Unknown 召回 26.3 是倒数第二。开思考买回来的恰好是这一类
(Flash-Next 36.8→73.7),但没有一个总分够到 83.3。

**逐题对照(405B 对 125 错 25)**:最好的 Flash-Next(think)修好 13 个、弄坏 17 个,净亏;其余更差。
没有候选是平替,换裁判会让 SP 分数因为与 harness 无关的原因漂移 → **维持官方 405B**。

⚠️ 服务坑:H200 上 FP8 要 `VLLM_USE_DEEP_GEMM=0`(DeepGEMM JIT 撞 nvcc 13.4 头文件);
Flash-Next 的 Mamba 缓存要 `max_num_seqs<=57`;405B-INT4 要用 pipeline parallelism(TP 触发 FlashInfer JIT),
且 `generation_config` 缺 `<|eot_id|>`,必须传 `stop_token_ids [128001,128008,128009]`。

</dd></dl>
</details>

<details>
<summary>A3.3–A3.4 Jev 1.13:闭源 typed-decision 模型当裁判(10-13,12 s / $0.0041)</summary>

<dl><dd>

Jev 不吃对话,吃 `state` + 带类型的 `questions`,回传带概率的类型化答案(`POST /v1/systemone`)。
这正是裁判任务的原生形状,所以值得跑一轮。三条官方裁判规则原文搬进 `instructions`,三个标签搬进
`criteria`。⚠️ 它收不了 system prompt,**官方提示词无法逐字节原样送入**,这是此对照唯一的不对等。

| | Jev 1.13 | Llama-3.1-405B-INT4 |
|---|---|---|
| accuracy | 78.7 | 83.3 |
| C / I / U 召回 | 80.0 / 88.2 / **36.8** | 89.1 / 93.4 / 26.3 |
| 150 题耗时 | **12 s**(并发 16) | ~15 s / 2×H200,另加 270 s 加载 |
| 成本 | **$0.0041**(98,480 输入 token @ $0.042/MTok) | 两张 H200 的机时 |

**唯一给出可用分布的候选**。置信度 `1 − H(p)/ln K` 与正确率单调:

| 置信度 | n | 正确率 |
|---|---|---|
| 0.0–0.2 | 12 | 50.0% |
| 0.2–0.4 | 26 | 61.5% |
| 0.4–0.6 | 37 | 73.0% |
| 0.6–0.8 | 30 | 86.7% |
| 0.8–1.0 | 45 | 95.6% |

**三种官方推荐写法都没把它调上去**(A3.4):结构化 `what`/`not_for`/`examples` criteria 74.0、
拆成「可判定性 noul + 二选一 choice」74.7、两者合用 76.0(宏平均最高 70.9)。每一个都是拿 3–5 分
总分换 10–25 分 Unknown 召回,**和直接调概率阈值是同一笔交易**,还多一次请求。

⚠️ **Jev 非确定性**:同一请求原样跑五次得 78.7 / 80.0 / 79.3 / 80.0 / 78.0,Unknown 召回 26.3–36.8,
逐题概率也在动。**150 题上两分以内的差距单轮读不出来**。

**不采纳的三条理由**:逐题不是平替;AR-Bench 官方裁判是 405B 且报总分;它是闭源托管,换上去意味着
每则故事和每次猜测都要离开我们的机器。合适的位置是当便宜的筛子,或当第二意见用置信度挑复核对象。

</dd></dl>
</details>

<details>
<summary>A3.5 宏平均翻案:405B 总分第一、平均分第五(10-13)</summary>

<dl><dd>

accuracy 按金标分布加权(C 55 / I 76 / U 19),**一个从不说 Unknown 的裁判几乎不受罚**。按宏平均
召回与宏平均 F1 重排:

| 判官 | accuracy | macro recall | macro F1 | C / I / U |
|---|---|---|---|---|
| Qwen3.8-Flash-Next(think) | 80.7 | **79.9** | **76.0** | 90.9 / 75.0 / 73.7 |
| Qwen3.8-27B(think) | 78.7 | 75.4 | 72.5 | 85.5 / 77.6 / 63.2 |
| Qwen3-235B-A22B-2507 | 74.0 | 72.2 | 68.6 | 83.6 / 69.7 / 63.2 |
| Qwen3.8-Flash-Next | 79.3 | 69.8 | 70.1 | 90.9 / 81.6 / 36.8 |
| **Llama-3.1-405B-INT4(在用)** | **83.3** | 69.6 | 70.9 | 89.1 / 93.4 / 26.3 |
| Jev 1.13 | 78.7 | 68.3 | 69.8 | 80.0 / 88.2 / 36.8 |
| Qwen3.6-35B-A3B | 72.7 | 66.4 | 65.9 | 87.3 / 69.7 / 42.1 |
| Qwen3-32B | 74.0 | 62.6 | 63.6 | 80.0 / 81.6 / 26.3 |
| Qwen3.8-27B | 64.0 | 62.1 | 60.0 | 74.5 / 59.2 / 52.6 |
| Llama-3.3-70B | 73.3 | 60.5 | 60.8 | 76.4 / 84.2 / 21.1 |
| gpt-oss-120b | 62.7 | 60.1 | 57.7 | 61.8 / 65.8 / 52.6 |

**这件事在真实对局里不是小数**:SP 试点 180 局共 4500 次裁判回答,**No 58.5% / Yes 27.6% /
Unknown 13.8%** → 约每七个提问就有一个落在 405B 最弱的类。裁判少说 Unknown,就会把这些问题强行判成
Yes 或 No,**这是往游戏本身注入噪声,不只是指标问题**。

**处置**:裁判不换(官方口径),但**论文里总分旁边必须放宏平均**,且主结果到位后用
Flash-Next(think) 当裁判把 SP 重跑一遍做稳健性检查。〔待跑〕

</dd></dl>
</details>

<details>
<summary>A3.6 读概率(typed read):机制有效、校准属于模型(10-14,H100)</summary>

<dl><dd>

Jev 的优势不是准确率,是判决以分布形式到达。这个机制不专有:任何自己服务的模型都能被约束到三个标签、
在首 token 读出分布。[`runner/typed_judge_eval.py`](../runner/typed_judge_eval.py) 用
`structured_outputs: {"choice": [...]}` 加官方裁判提示词原文做这件事,
[`h100/typed_bench.sh`](../h100/typed_bench.sh) 把同一模型按三种协议各跑一遍。

| 模型 | 采样 T=0.7 | 采样 T=0 | 读概率 | ECE | C / I / U(读概率) | macro |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 34.7–37.3 | 34.7–35.3 | 36.0–36.7 | 0.56 | 50.9 / 17.1 / 68.4 | 45.5 |
| Qwen3-32B-FP8(no think) | 72.7–74.7 | 73.3–74.0 | 73.3 | 0.16 | 80.0 / 80.3 / 26.3 | 62.2 |
| **Qwen3.8-27B-FP8(no think)** | 64.7 | 70.0 | **70.7** | **0.055** | 76.4 / 68.4 / 63.2 | **69.3** |

**读概率从不改变判决**(与 greedy 采样差 ≤1 分,取的是同一个 argmax),它只多给分布;而分布值不值钱
**取决于模型**。32B 的 ECE 0.16,阈值因此是真旋钮:Unknown 概率 ≥0.30 则判 Unknown,召回 26.3→36.8、
总分只掉 0.6、宏平均 62.2→64.4。7B 的 ECE 0.56,平均取到 0.945 而正确率 0.360,分布无信息。
**Qwen3.8-27B 是测过校准最好的判官**:ECE 0.055、略微欠自信(均值 P 0.665 vs 正确率 0.707)、
宏平均 69.3 与 405B 的 69.6 持平,而它只占一张 H100 的一部分。它也是唯一对温度敏感的(T=0.7 低 5.3 分)。

**7B 的失败形状是单一的**:150 题答 99 次 Unknown,**禁掉 Unknown 则 36.0 → 68.0**。不是普遍不会判,
是过度弃权。

⚠️ **三次错误都产出过看起来合理的数字,全部记录在案**:①采样轮默认 T=0.7 而读概率是 0,实测值 0.7–1.4 分;
②推理模型的首 token 是思考块标记不是标签,往后扫位置时散文里的字母 "I" 被当成 Incorrect 的前缀,
得到 **50.7% 且 150 题全判 Incorrect、置信度 1.000**;③**vLLM 0.19 静默忽略 `guided_choice`**,请求照常
成功、数字与不加约束时一字不差。正确字段是 `structured_outputs`,关思考要用
`chat_template_kwargs: {"enable_thinking": false}`。判定过程留在
[`runner/probe_guided.py`](../runner/probe_guided.py)。

</dd></dl>
</details>

<details>
<summary>🔗 可回溯(A3)</summary>

<dl><dd>

- 评测脚本:[`runner/turtle_judge_eval.py`](../runner/turtle_judge_eval.py)(采样)·
  [`runner/typed_judge_eval.py`](../runner/typed_judge_eval.py)(读概率)·
  [`runner/jev_judge_eval.py`](../runner/jev_judge_eval.py) 与
  [`runner/jev_judge_variants.py`](../runner/jev_judge_variants.py)(Jev)·
  [`runner/probe_guided.py`](../runner/probe_guided.py)(约束写法探针)
- 服务脚本:☁️ `engaging/calib.sbatch` · `engaging/serve_referee.sbatch` · `engaging/judge405.sbatch` ·
  📁 [`h100/typed_bench.sh`](../h100/typed_bench.sh)
- 逐题预测与运行日志:📁 [`experiments/02_judge_calibration/turtle150_predictions/`](../experiments/02_judge_calibration/turtle150_predictions/)
- 详情页:📁 [`experiments/02_judge_calibration/README.md`](../experiments/02_judge_calibration/README.md)
- token:Jev 的 key 在 mac 钥匙串 `typesafe_token`,**值不入库**

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A2 / A4 / A5 / A6 任务天花板与 baseline</b></summary>

<dl><dd>

<details>
<summary>A2 Stage-0:假设空间、oracle 与合成策略(10-06,CPU)</summary>

<dl><dd>

| policy | solved | queries | ex-ante eff | zero-info queries |
|---|---|---|---|---|
| greedy max-entropy(oracle) | **300/300** | 11.60 | 1.000 | 0.0% |
| greedy over 4 sampled queries | 250/300 | 13.92 | 0.775 | 18.1% |
| random fresh entry | 156/300 | 16.51 | 0.629 | 27.7% |
| sequential scan | 58/300 | 17.41 | 0.540 | 38.1% |
| uniform random(允许重复) | 35/300 | 17.63 | 0.480 | 44.1% |

**18 个 query 的预算够用**:oracle 平均 11.60 个就把假设空间收敛到 1。所以后面任何失败都**不能归因到
预算不足**,只能归因到策略。这是全部后续实验的前提。

</dd></dl>
</details>

<details>
<summary>A4 SP 无 harness 基线(官方级裁判,10-10 21:29–21:36,7 min)</summary>

<dl><dd>

f1_char **0.621**。⚠️ 此前记录的 0.659 是 **7B 自己当裁判**跑出来的,已作废 —— 这正是 A3 整棵树存在的
理由。换成 405B-INT4 后,SP 的所有历史数字都要以此为基准重算。

</dd></dl>
</details>

<details>
<summary>A5 可操作性阶梯:七个手写 harness(10-10~11)</summary>

<dl><dd>

| harness(每轮回答后追加什么) | cwa | joint | uniq@end (ours) | eff (ours) | tok/ep |
|---|---|---|---|---|---|
| bare — 什么都不加 | 12.9 | 0.5 | 23.2 | 0.553 | 2389 |
| facts (a) — 把模型自己的观察复述一遍 | 16.2 | 0.5 | 29.0 | 0.570 | 2800 |
| symbolic — 剩余配置数 + 已定输出 | 20.5 | 2.2 | 41.2 | 0.612 | 2648 |
| enum15 — symbolic + 剩 ≤15 时列全表 | 23.7 | 5.8 | 41.8 | 0.612 | 2892 |
| **T3-gated** — 仅当 trap proxy 触发才 enum15 | **31.3** | **10.2** | 36.8 | 0.612 | 3104 |
| splits (c) — enum15 + 每个开放项的 0/1 split 数 | 22.3 | 5.7 | **53.7** | 0.702 | 4316 |
| oracle hint (d) — enum15 + 最大熵 query | 27.3 | 6.8 | **96.5** | 0.951 | 4522 |

**uniq@end 与 joint 不同步**,这是整条线最重要的一条观察:oracle hint 把信息收集拉到 96.5(几乎每局
都把假设空间收敛到 1),joint 却只有 6.8。**提问已经够好,瓶颈在最终把收敛后的状态读成答案**。
T3-gated 用更少的信息拿到更高的 joint,也指向同一处。

</dd></dl>
</details>

<details>
<summary>A6 EvoHarness-RL 当外部 baseline(10-11 06:28–07:36,1 h 08)</summary>

<dl><dd>

| mode | run | cwa | joint | uniq@end (ours) |
|---|---|---|---|---|
| env_only(ReAct,无 harness) | s0 / s1 | 4.3 / 4.9 | 0.0 / 0.0 | 29.3 / 26.0 |
| inline = EvoHarness-Base(模型自己调 track / commit / recall / note) | s0 / s1 | 5.7 / 6.9 | 0.7 / 1.7 | 30.0 / 31.7 |
| **always_on(每轮贴 STATE 面板)** | s0 / s1 | **74.8 / 79.0** | **62.0 / 65.0** | **71.0 / 74.7** |

同样的状态信息,**面板常开碾压让模型自己维护**(joint 62–65 对 0.7–1.7)。inline 几乎没比 ReAct 好,
说明「给模型工具让它自己记」在这个任务上不成立。与 A5 同向:harness 要替模型维护状态,但**不能替它推理**。

</dd></dl>
</details>

<details>
<summary>🔗 可回溯(A2 / A4 / A5 / A6)</summary>

<dl><dd>

- 详情页:📁 [`01_stage0_oracle`](../experiments/01_stage0_oracle/) ·
  [`07_sp_bare_405judge`](../experiments/07_sp_bare_405judge/) ·
  [`03_cd_ladder`](../experiments/03_cd_ladder/) ·
  [`06_evoharness_cd_baseline`](../experiments/06_evoharness_cd_baseline/)
- 打分:[`runner/cd_score.py`](../runner/cd_score.py) · [`runner/cd_summarize.py`](../runner/cd_summarize.py) ·
  [`runner/cd_ladder_plot.py`](../runner/cd_ladder_plot.py)
- 原始轨迹:GitHub Release `v0.1-experiments-2026-10-11`(`cd_ladder_traces` / `evoharness_cd_trajectories` / `sp405_traces`)

</dd></dl>
</details>

</dd></dl>

</details>

> ---

<details>
<summary><b>A7 指标审计:官方 F1 在量什么(10-21)</b></summary>

<dl><dd>

对象是已有的 500 局冻结评测记录(5 个 harness × 100 题),不跑新推理。
脚本 [`runner/metric_audit.py`](../runner/metric_audit.py),详情页
[`experiments/11_metric_audit/`](../experiments/11_metric_audit/)。

| 与 F1 (char) 的相关 | r |
|---|---|
| **答案词数** | **+0.806** |
| 整局输出 token | +0.403 |
| **key-question recall**(ours) | **+0.050** |

词数单独解释 R² = 0.650;加上 recall 只到 0.652(recall 系数 +0.021);控制词数后 recall 与 F1 偏相关 +0.068。
分桶 25–50 词 0.458 / 50–75 0.634 / 75–100 0.766 / 100–125 0.826 / 125–150 0.868,**单调无拐点**。

**控制长度后我们的增益还剩多少**:25–50 词 +0.005、50–75 **+0.072**、75–100 +0.043、100–125 +0.004,
而未控制的总体差是 **+0.152**。终局写 91 词、bare 写 61 词,差额大部分由此而来。长区间差异趋零,
既可能是优势小、也可能是 F1 在长答案上饱和,**单靠 F1 分不开**,要靠语义评分。

**对 benchmark 的三条判断**:指标**奖励啰嗦**,没有起作用的长度惩罚,我们的 harness 正是被这个梯度推着
把输出从 670 推到 5263 token;**允许跳步**,没有任何机制要求最终答案由问答历史推出(degenerate 基线
待测,需要服务器上的谜底);**环境自带噪声**,referee 在 TurtleBench 上 83.3%,约六分之一回答可能是错的。

⚠️ **留痕**:首轮计算得 r = 0.90,原因是把 405B 被抢占那次运行的 88 局空答案算了进去,那些点落在
(0 词, 0 分) 上。剔除后 0.806。结论方向不变,数值此前报大了。

**口径不变**:AR-Bench 的指标就是字符级 F1,主表照报(见
[`OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md)),A7 是局限分析不是替换指标。

</dd></dl>
</details>

<details>
<summary><b>A8 独立复核:Sun 的固定对话探针(10-15~21)</b></summary>

<dl><dd>

Yangzi Sun 的预注册实验,把**提问**与**最终作答**拆开:固定我们留下的两份对话(bare / 第一代),
配两种 writer(普通 / 证据约束),四格 × 65 题 = 260 次生成,520 次盲评,新增 Semantic Correctness
与 Evidence Faithfulness 两个序数指标。
[报告](https://github.com/sun-tze/self-evolving-active-reasoning/blob/sun/c2-confirmatory-65/experiments/sun_c2_confirmatory/REPORT.md)

| 条件 | F1 | 语义正确 (0-2) | 证据忠实 (0-2) | 两评审都判全对 |
|---|---|---|---|---|
| A bare / 普通 | 0.681 | 0.785 | 0.662 | 7/65 |
| B bare / 证据 | 0.683 | 0.754 | 0.669 | 8/65 |
| C 第一代 / 普通 | 0.693 | 0.754 | 0.331 | **15/65** |
| D 第一代 / 证据 | 0.695 | 0.662 | 0.400 | 12/65 |

**采纳**:F1 与语义在 29 题上出现 82 次严格排名反转,字符重合与理解因果机制可以完全脱钩;
证据约束 writer 没有提升语义正确性(−0.0308 [−0.0577, −0.0058]),大概率因为海龟汤的谜底按定义
不在证据里,把模型钉死在已确认事实上恰好压掉了必要的溯因跳跃。这与 [A7](#) 相互独立地指向同一结论。

**不采纳的三点**:①它测的是**第一代**,而第一代改的本就是答案不是提问,这一点实验 10 原文已写明;
终局在未见题上 recall 57.6→66.4。②未选中的 `b2r0g1` **用一半 token** 拿到更高 F1 且 recall +10.5,
纯长度解释不通。③它自己的数据里"两位评审都判全对"是 **15/65 对 7/65**,翻了一倍,被序数均值掩盖
——evolved 呈双峰(去掉满分后余者均值 0.38),bare 则大量拿部分分(0.64)。

**方法学边界**(报告自述):没有复现我们完整的作答机制(三份草稿 + 打分 + 三道检查),所以四格里没有
一格等于我们实际跑的系统;评审是同家族模型两次独立执行,96.9% 一致率不能证明正确性,需人工验证。
faithfulness 还可能被"可违反证据数"混淆:bare 问出的约束少,违反机会天然就少。

**下一步(未跑)**:对 **bare / 终局 / b2r0g1** 的完整 harness 答案做跨家族盲评(Opus 5,TurtleBench 88.0
对 405B 83.3),抽 30 条人工核验,同时报均值与全对率。

</dd></dl>
</details>

> ---

# 第二阶段 · harness 自进化:一次性进化 → 测试时进化 → 跨任务

<details>
<summary><b>B1 / B2 一次性进化(regime open vs state_only)</b></summary>

<dl><dd>

<details>
<summary>B1 `open1`:regime open(10-10 20:30–23:10,2 h 40)</summary>

<dl><dd>

proposer **一轮到顶**:copy_query 在 held-out 上 joint **80.4** / uniq@end 99.7。
⚠️ open regime 下 harness 可以直接替模型选 query,所以这条的高分**不能用来支持「模型自己在推理」**,
它是天花板不是主张。收口到 B2。

</dd></dl>
</details>

<details>
<summary>B2 `so1`:regime state_only(10-10 21:22–10-11 ~01:00,~3 h 38)</summary>

<dl><dd>

规则:harness 可以维护与展示状态、可以否决,但**不得计算信息增益、不得报 split 数、不得推荐 query**。
veto_only 在 held-out 上 cwa **91.3** / joint **86.2** / uniq@end 96.7,**token 与 bare 持平**。

**这是整条线的主张形状**:拿掉替模型算的部分,只留约束与状态,分数仍然上去,而且**不多花 token**。
与 A5 的 T3-gated、A6 的 always_on 三处同向。

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>B3 / B4 测试时进化(TTHE)与标签隔离 · 大树</b></summary>

<dl><dd>

> TTHE = 实例按批到来,每批之后由 proposer 提出新 harness、judge 在**不看 gold** 的前提下决定是否提交。
> 两条流:**strict**(只给运行状态)与 **labeled**(额外给结果),用来证明 label-free 不比 labeled 差。
> proposer 与 judge 都是 Claude Opus 5.5(Claude Code 内),questioner 是冻结的 Qwen2.5-7B。

<details>
<summary>B3 试点与标签泄漏(10-12 16:14–17:51,1 h 37)⚠️ 结果作废</summary>

<dl><dd>

**干净的部分**:strict 第 0 批,**零提示下 proposer 自己重建了假设空间**(写出枚举器),无标签 judge
把它选中,第 1–2 批 20/20 joint 全对。流 30 题 prequential cwa 70.0 / joint 66.7。

**作废的部分**:审计查出后续批次标签未隔离 —— gold 出现在 trace JSON、lineage 聚合与跨会话 memory
笔记里,proposer 的候选文件也落在 `agents/` 被下一轮读到。**整条流重跑(B4)**。

教训落到三处:控制器私有存储 `~/.ar_harness_controller/<run>/`;每会话自动审计(路径解析 + cwd 追踪 +
heredoc 正文 + 值扫描);Claude Code auto-memory 按仓库根 slug 隔离并每会话清空。

</dd></dl>
</details>

<details>
<summary>B4 标签隔离重跑 + 冻结评测(10-12 19:01–22:33,3 h 32)</summary>

<dl><dd>

**strict6 流**(B=10,G=3,R=2):

| batch | incoming → committed | cwa | joint | transductive joint | pool | wall |
|---|---|---|---|---|---|---|
| 0 | bare → b0r1g1 | 7 | 0 | 100 | 5 | 2577 s |
| 1 | b0r1g1 → b1r1g0 | 97 | 90 | 90 | 6 | 1080 s |
| 2 | b1r1g0 → b2r0g1 | 100 | 100 | 100 | 6 | 2188 s |

**labeled6 流**:第 0 批就到 100/100,之后两批不换(b1r0g2 连任)。

**冻结评测(200 held-out,无 proposer、无 judge)**:bare cwa 11.8 / joint 0.0;strict 终局 b2r0g1
**cwa 99.5 / joint 98.5**(未见 170 题 98.2);b1r1g0 99.3 / 98.0;labeled 终局 100/100 @ **215 tok/ep**。

**审计**:29/32 会话干净,**judge 的会话全部干净**。judge 三次否掉了 10/10 的「菜单式 / 排序式」
harness,理由是那等于 harness 替模型选 query,强制保住「**每个 query 由模型自己写**」。
这条是机制成立与否的分水岭,不是风格偏好。

**strict ≈ labeled** 是「label-free」口径的直接证据(两条流终局都到满分,strict 只多花一批)。

</dd></dl>
</details>

<details>
<summary>🔗 可回溯(B3 / B4)</summary>

<dl><dd>

- 循环:[`meta/cd_tthe/optimize.py`](../meta/cd_tthe/optimize.py) ·
  [`proposer.py`](../meta/cd_tthe/proposer.py)(含 OFF_LIMITS、`audit_session`、订阅限流重试)·
  [`solve_worker.py`](../meta/cd_tthe/solve_worker.py)(reveal 级别)· [`audit.py`](../meta/cd_tthe/audit.py)(静态审计)
- 冻结评测:[`meta/cd_tthe/evaluate.py`](../meta/cd_tthe/evaluate.py) · 轨迹图 [`trajectory.py`](../meta/cd_tthe/trajectory.py)
- 详情页:📁 [`08_cd_tthe_pilot`](../experiments/08_cd_tthe_pilot/) · [`09_cd_tthe_isolated`](../experiments/09_cd_tthe_isolated/)
- 对话实录:各详情页 `dialogues/`;判官裁决 `judge_verdicts/`;会话审计 `session_audits.jsonl`
- 原始轨迹:Release `cd_tthe_pilots_raw.tar.zst` / `cd_tthe_isolated_raw.tar.zst`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>B5 跨任务移植:AR-Bench Situation Puzzles(10-13 05:03–07:17,2 h 14)</b></summary>

<dl><dd>

换一个**没有可枚举假设空间**的任务,看同一套循环还成不成立。questioner 仍是冻结 Qwen2.5-7B,
referee 与 key-question judge 是官方 Llama-3.1-405B-INT4,协议是 AR-Bench 自己的 25 个是非问 + 最终作答。
30 题流(seed 0),B=10 / G=3 / R=2,**strict**(只给运行状态与裁判回答,不给 F1)。

**流**(prequential = 在役 harness 在新一批上、适应之前):

| batch | incoming → committed | F1 (char) | key-q recall | transductive F1 | pool | wall |
|---|---|---|---|---|---|---|
| 0 | bare → b0r1g2 | 63.7 | 55 | 79 | 7 | — |
| 1 | b0r1g2 → b1r1g0 | 74.5 | 65 | 67 | 5 | — |
| 2 | b1r1g0 → b2r1g0 | 79.9 | 72 | 69 | 6 | — |

**冻结评测(全部 100 题,其中 70 题流里没用过)**:

| harness | F1,100 | F1 未见 70 | key-q recall,100 | tok/ep |
|---|---|---|---|---|
| bare(AR-Bench 官方 zero-shot 循环) | 60.4 | 61.4 | 59.7 | 670 |
| b0 提交 | 76.7 | 75.7 | 59.0 | 4486 |
| b1 未选中(b1r0g0) | 73.0 | 73.8 | 64.3 | 4685 |
| b1 提交 | 76.2 | 75.9 | 63.2 | 4944 |
| **b2 未选中(b2r0g1)** | 76.3 | **77.5** | 68.2 | **2112** |
| b2 提交(终局) | 75.6 | 75.4 | **68.7** | 5263 |

**十条失败轨迹、无标签进化出来的 harness,让冻结的 7B 在没见过的题上从 60.4 升到 73–77**,
**超过 AR-Bench Table 6 里所有 zero-shot 模型,含 GPT-4o 的 63**,而提问仍然全部由 7B 自己写。
key-question recall 只在后两代才涨(59→68)→ **第一代改的是答案,不是提问**。代价是输出 token 3–8×;
未选中的 b2r0g1 用一半 token 打平。

**长出来的是什么**(b0 胜出者,adversarial-audit proposer):每轮给模型一条新消息,含谜面、裁判回答
按 CONFIRMED / RULED OUT / NOT IN THE STORY 分组、以及一个焦点;先查读者默认成立的前提(海龟汤的机关
通常就在某个前提为假),再维护一个**必须包含全部已确认事实**的工作故事,未确认部分逐条验证;一轮没有
Yes 之后问一个没问过的基本面。每个候选提问过格式检查(单一事实、是非问、无 "or"、不问人物想法、
不编名字、非近似重复),最终答案从已确认事实重写。B1/B2 加了重复与已否线索的否决、同一实体被否两次后
禁用、未确认猜测不许用 "the X"。

⚠️ **判官在 SP 上有噪声**:b1 提交了池里 F1 最低的(0.674 vs 在役 0.745),b2 跳过了最好的(0.802)。
10 题一批的 gold 差距本就在噪声内,**这件事本身是结论**,量化见 [B6](#)。
会话审计 18/21 干净,3 次丢弃是相对路径误计(解析到文件夹外但该路径不存在),审计已加存在性检查。

</dd></dl>

<details>
<summary>🔗 可回溯(B5)</summary>

<dl><dd>

- 任务适配:[`meta/cd_tthe/tasks.py`](../meta/cd_tthe/tasks.py)(SPECS cd/sp)·
  [`sp_bridge.py`](../meta/cd_tthe/sp_bridge.py)(SPEnv,谜底与 key question 私有)·
  [`sp_harness_base.py`](../meta/cd_tthe/sp_harness_base.py) · [`sp_signals.py`](../meta/cd_tthe/sp_signals.py) ·
  [`agents_sp/bare.py`](../meta/cd_tthe/agents_sp/bare.py)(官方 zero-shot 循环)
- 详情页:📁 [`10_sp_tthe_pilot`](../experiments/10_sp_tthe_pilot/) — 含 `judge_verdicts/`、`proposal_cards/`、
  `candidates/`、`eval_test100_scores.json`、`trajectory.png`
- 原始轨迹:Release `sp_tthe_pilot_raw.tar.zst`(17 MB)
- 成本:21 个 proposer/judge 会话,约 **$12**(订阅计价),2 h 14 min

</dd></dl>
</details>

</details>

<details>
<summary><b>B6 给 TTHE 判官造一把尺(10-13)</b></summary>

<dl><dd>

TurtleBench 量的是 **referee**。决定提交哪个 harness 的 **judge** 一直没有尺。它不需要外部基准:每次
运行都记着候选池的 gold,而 judge 从没看过;把它和实际提交的候选对账就是测量。
[`runner/judge_pick_eval.py`](../runner/judge_pick_eval.py):

| 运行 | 任务 | 提交的是池里最好的 | 平均 regret | 最差 | 平均名次 |
|---|---|---|---|---|---|
| `strict6_b10g3r2` | CD | 2/3 | 0.011 | 0.033 | 2.00 |
| `labeled6_b10g3r2` | CD | 3/3 | 0.000 | 0.000 | 1.00 |
| `sp_strict1_b10g3r2` | SP | **1/3** | **0.063** | **0.114** | 3.67 |

regret 用各任务自己的指标。CD 的池子饱和(六个候选都 1.000),这把尺在那里说明不了什么;**SP 才见真章**:
两次错判分别丢 0.075 和 0.114 的 F1,b1 那次提交的是五个里最差的。

**这个数该进论文**:无标签 judge 是我们主张的机制的一部分,而在更难的任务上它是**可测地有噪声**的。
部分原因是 10 题一批,gold 差距本在噪声内 —— 这本身也是结论。

**外部的尺(都还没跑)**:referee 做的是三分类有据推理,这在谜题游戏之外有现成基准 ——
**FEVER**(SUPPORTED / REFUTED / NOT ENOUGH INFO,标签语义一一对应,NEI 就是 Unknown)、
**ANLI / MNLI**(蕴含 / 矛盾 / 中性,无游戏腔调,能把推理与格式适应分开)、
**SQuAD 2.0 不可答子集**(单独隔离「段落判不判得了」这一步)。
**JudgeBench / RewardBench 2** 量的是另一种 judge(在候选回答间择优),精神接近但决策不同,
顶多当旁证 —— 上面这张 regret 表才是合适的工具。前三个都公开、不用登录。

</dd></dl>

</details>

> ---

# 运维与环境(踩过的坑)

<details>
<summary><b>两台机器的能力边界与已知陷阱</b></summary>

<dl><dd>

| | MIT ORCD / Engaging | 独立 H100(`ssh h100`) |
|---|---|---|
| 卡 | H200 141 GB,`mit_normal_gpu` 最多 2 张 | 4×H100 80 GB,**与他人共用**,按空闲显存现选 |
| 排队 / 抢占 | 有队列;405B 在可抢占分区,10-13 当天被抢占两次 | 无队列无抢占,起服务 1–3 min |
| 软件 | 新版 vLLM | `venv_cu128`(vLLM 0.19.1 + torch 2.11+cu128);驱动 570.195.03 = CUDA 12.8,**版本天花板 vLLM 0.26.0** |
| 装不下 | — | 405B-INT4(要两张)、Qwen3.8-Flash-Next(约 126 GB 常驻)、Qwen3-235B |

**陷阱清单**:

- **Engaging 要 Duo**,公钥 + Duo 双要求、无 token 可存,只能靠 ControlMaster 复用;断了必须人工过一次。
- **H100 上两个 venv 只有 `venv_cu128` 能用**;另一个是 torch cu130,报 "NVIDIA driver is too old"。
- **`ninja` 是推理时依赖**。带 Gated DeltaNet 的模型(Qwen3.8)在**第一次前向**才编译注意力核;缺了的表现是
  服务正常启动、`/health` 通过,然后每个请求 500、引擎死掉。
- **模型名不等于架构名**。Qwen3.8-27B 的 HF `architectures` 是 `Qwen3_5ForConditionalGeneration`,
  vLLM 0.19.1 本来就支持 —— 在注册表里搜 "3.8" 会得出错误结论。
- **`pgrep -f` / `pkill -f` 的模式不能包含自己的命令行**。犯过两次:等待循环匹配到自身而死锁;
  pkill 模式过宽把自己的 ssh 会话杀掉。
- **vLLM 0.19 静默忽略 `guided_choice`**,请求成功、什么也没约束,数字与不加约束时完全一致。
- Claude Code 的订阅 5 小时窗口会静默作废会话 → proposer 加了 LIMIT 检测与
  `--resume-from BATCH ROUND` 续跑。

详见 📁 [`h100/README.md`](../h100/README.md)。

</dd></dl>

</details>
