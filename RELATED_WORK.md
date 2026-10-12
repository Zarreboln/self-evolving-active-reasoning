# Related work — 主动推理上的 harness 自进化

**状态**：2026-10-06 第一轮收集。除标注「已精读」外，**全部只到搜索摘要级**，写进论文前必须下 PDF 核对。
本项目的三条贡献（可外部计算的信息效率度量 / T3 代理条件的校准 / 触发+注入回滚的推理时控制器）
在下面逐条标出「被占到什么程度」。

---

## 0. 直接前驱（已精读）

**T3: Reducing Belief Deviation in RL for Active Reasoning** — arXiv 2510.12264, ICLR 2026
Definition 1 给出 belief trap region（BTR），Theorem 1 在五条假设下证明信念势超过阈值后必然进入。
Definition 2 的「T3 Condition」是**逐任务手搓的代理信号**（如 CD 的「假设空间连续 3 轮不下降」）。
训练时截断进入 BTR 的轨迹尾部，Corollary 1 证明可减小优势估计偏差。只 RL 训了 Qwen2.5-7B。
作者自承局限：信念空间庞大、阈值不可测、代理条件需逐任务调。
→ **我们的切入点**：它有生成控制权却仍用启发式代理，因为 BTR 定义在 agent 信念上而 agent 信念读不出来。

**AR-Bench** — arXiv 2506.08295, ICML 2025, 代码已读
GN/SP/DC 三族。关键事实：DC 答案由 LLM 生成且凶手被指示撒谎；SP 隐藏状态是自由文本；
只 GN 用 oracle 函数。论文自承 SFT/DPO/ToT/人工指令/Proactive CoT/UoT **全都几乎没有收益**。

**Multi-Turn Puzzles** — arXiv 2508.10142，数据 CC BY 4.0，**无代码**
CD/WG/TQ/WC/MR 五族。CD = 3 电路 × 3 输入 = 24 位真值表、18 次单 bit 查询。

---

## 1. ⚠ 最该先读的两篇：可能已占住我们的位置

**Belief-State Engine: Augmenting LLMs for Principled Planning Under Partial Observability** — arXiv 2609.10036
> 「在 POMDP 模型上跑贝叶斯滤波，作为 LLM 之外的模块；每个决策步只把后验信念作为上下文交给 LLM。
> 信念是**外部的、显式的**，是有限潜在状态空间上的概率分布，而不是藏在激活里或写成一段自由文本。」
→ **这正是我们今天的实验指向的那个干预**（proactive_cot 给 42 倍推理空间无效 → 把记账搬到模型外面）。
**必须精读并明确差异**：它是方法论文、给定 POMDP 模型；我们的主张是「这个干预该不该做」由可计算的失效诊断决定，
而且 harness 要在对抗课程下自己进化出来。若差异撑不住，贡献 3 要重写。

**BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design** — arXiv 2508.21184, ICLR 2026
迭代选择最大化 EIG 的问题，用 LLM 的预测分布构造概率模型。**在 20 Questions 上成功率比纯提示基线翻倍以上。**
→ 与我们的 ex-ante 信息效率共用同一个量（EIG）。**区别必须写清楚：它用 EIG 做决策（方法），
我们用 EIG 做测量（把 agent 实际选的问题对最优问题打分）。** 测量用途下 EIG 必须精确可算，
所以我们限定可枚举假设空间；它用 LLM 估计 EIG，近似。这个区别是真的但很细，表述要小心。

---

## 2. MastermindEval —— 「直接冲突」是误报，但查出它一个真 bug

**MastermindEval** — arXiv 2503.05891, ICLR 2025 Workshop，代码 github.com/flairNLP/mastermind（已克隆、已跑）

**更正**：2026-10-06 第一轮我据搜索摘要写了「它有 feedback compliance analysis，说 LLM 大体遵守约束」，
并标为与我们 88.9–95.7% dead-guess 率直接冲突。**精读后这条不成立**：
- 论文正文：`complian` 0 次、`violat` 0 次、`feasible` 0 次、`invalid guess` 0 次。
- 代码里唯一相关的是 `Evaluator.progress()`，它算**每轮之后剩余候选集的大小**，
  **不检查模型的猜测是否在可行集内**。
→ 那段「compliance」描述是搜索引擎合成的，没有出处。**冲突解除，贡献 1 的空位仍然开着。**

**但读代码时查出一个真 bug，已复现：**
```
Evaluator.progress():  remaining_states = product(possible_colors, repeat=code_length)  → 1296
Mastermind._generate_secret_code():  random.sample(possible_colors, k=code_length)      →  360
```
密码由 `random.sample` 生成，**永远没有重复颜色**（实测 3000 个密码，0 个含重复），
但候选空间用 `product` 枚举**含重复**的组合。**72.2% 的候选永远不可能是密码。**
同一条 3 步轨迹：他们 `progress_history = [1296, 20, 2, 1]`，正确空间下是 `[360, 8, 2, 1]`。
→ 论文摘要里的「1,296 codes」本身就是错的数；`progress` 指标系统性高估剩余不确定性，早期尤甚。
   他们的 deductive 数据集「预演到只剩一个合法 code」也可能受影响（在 1296 上过滤需要更多轮才唯一）。
→ 与 AR-Bench 那个「生成器禁前导零但数据里有」同一类错误。**两个 benchmark 的假设空间定义都和自己的数据对不上。**
   这本身可以成为论文里一句话的论据：**精确的假设空间不是现成的，得自己建并对数据验证。**

## 3. 信念状态外置 / POMDP 式 agent

- **Context Gathering Decision Process (CGDP)** — arXiv 2605.07042。agentic search 的 POMDP 框架，
  持久化的**谓词式信念状态**，多跳推理提升至多 11.4%。
- **LedgerAgent** — 显式 schema 锚定的 ledger 记录已观测任务状态，保证工具调用合规（客服多轮）。
- **Active Inference as Context Acquisition for AI Agents** — arXiv 2608.19202。
- **Flexible Agent Alignment with Goal Inference from Open-Ended Dialog** — arXiv 2508.15119。
- **The Unreliable Progress Bar: Can LLM Agents Reliably Report Task Progress** — arXiv 2609.08589。
  与「agent 对自身状态的自述不可靠」同源，可支持「不能用 agent 自述的信念做度量」这一论点。

## 4. 信息增益 / 提问策略（方法侧）

- **CA-BED: Conversation-Aware Bayesian Experimental Design** — arXiv 2606.01182，ICLR 2026
  *workshop*（Agentic AI in the Wild）。维护假设分布，在模拟对话树上传播 EIG。
  **★ 唯一一篇用了我们数据集的外部工作**：评测用 AR-Bench 的 **Detective Cases**（100 场景 × 5 嫌疑人，3 个种子）
  和 20 Questions（COMMON，111 个实体）。
  → 证实了我对 DC 的判断：隐藏状态可枚举（5 个嫌疑人）。**EIG 式提问选择在 AR-Bench DC 上已经有人做了。**
  但 DC 的信息需求只有 2.32 bit / 25 轮，信息论上平凡；且答题者是 LLM 且凶手被指示撒谎，可行集过滤失效。
  我们不与它在 DC 上竞争。
- **Uncertainty-Aware Clarification in LLM Agents with Information Gain** — arXiv 2606.03135。
- **Active Task Disambiguation with LLMs** — arXiv 2502.04485。
- **Bayesian Preference Elicitation with Language Models** — arXiv 2403.05534。
- **Learning to Ask Informative Questions: ... Preference Optimization and Expected Information Gain**。
- **Proactive CoT** — Deng et al. 2023a。**AR-Bench 的 proactive_cot 基线出处**；
  原工作做对话澄清，AR-Bench 把「候选动作 + 带推理地选」手工实例化到每个任务
  （GN 六个类目、SP/DC 八个，整个仓库里只出现在 prompt.py，论文正文 0 次提及）。
  → **这些手工类目是可进化对象，不是固定环境**；而且它们全在讲「怎么选下一个问题」，
  没有一个关于「怎么维护已排除集合」——与我们测到的真实瓶颈错位。
- **REVOIR** — arXiv 2609.37588（已精读，见 memory）。assistance game + 一步 VoI；单次会话、意图固定、无记忆。

## 5. Harness / scaffold 自进化（不改权重）

- **Self-Improvements in Modern Agentic Systems: A Survey** — arXiv 2607.13104。
  给出我们一直在用的那组术语：基础模型改进 = 参数化慢回路；**scaffold 改进 = 非参数化快回路**
  （提示、记忆、工具、完整 scaffold）。**写 related work 的骨架从这篇起。**
- **SIA: Self Improving AI with Harness & Weight Updates** — arXiv 2605.27276。
  harness 与权重**联合**优化，并提出信用分配问题：失败时该改提示、改工具封装、还是算梯度。
  → 与我们「只改 harness」的约束正相对；必须说明为什么我们刻意不联合。
- **A Self-Improving Coding Agent** — arXiv 2504.15228。meta-agent 改写任务 agent 的 scaffold。
- **FlashEvolve: Accelerating Agent Self-Evolution with Asynchronous Stage Orchestration** — arXiv 2605.08520。
- 记忆机制一支（见 memory [[personvideo-bench]]）：HiPS 2608.25329、AdaMem 2606.21144、
  Sager 2604.14972、Pep 2602.15012；负面结果 2609.01615（提示空间跨用户元学习不迁移）。

### 5.1 我们直接套用的方法模板（10-10 精读，PDF/txt 在 papers/）
- **Meta-Harness: End-to-End Optimization of Model Harnesses** — arXiv 2603.28052（Lee, Nair, Zhang, Lee, Khattab, Finn）。
  任务：围绕固定底座模型的 harness（存什么、取什么、给模型看什么）。问题：手写 harness 靠试错。方法：强 coding agent
  （Claude Code + Opus 4.6）当 proposer，文件系统暴露全部历史（候选代码、分数、**原始轨迹**），每轮提候选→评测→回写。
  结果：Table 3 只给分数 34.6、分数+摘要 34.9、给原始轨迹 50.0（median）；TB2 上 Haiku 4.5 排所有 Haiku harness 第一。
  官方代码 stanford-iris-lab/meta-harness（10-02 发布）含 ONBOARDING.md、两套 SKILL.md、claude_wrapper.py——我们的
  `meta/cd_evolve/` 逐项照搬（proposer 每轮全新会话、pending_eval 合同、frontier/summary/reports、held-out 冻结后一次性评）。
  与我们的差别：他们的信号是任务级分数+轨迹；我们每轮有 oracle 信息量（更密），且 truth 在接口上拿不到。
- **RRSI: Regularized Recursive Self-Improvement of Agent Harnesses** — arXiv 2609.24972（Google Cloud AI Research，Yoonho Lee 合著；代码 google-research/rrsi 已 clone 到 rrsi_official/）。
  任务：围绕冻结模型的 harness 进化。问题：在有限 evolve 集上反复提议-选择会过拟合——evolve 集涨、OOD 不涨甚至退。方法：不缩小编辑空间，正则化搜索轨迹。
  提议侧：每候选编辑数 b_t 余弦退火（L0 式）、proposer 条件于全部编辑历史（被证伪的假设不重提）、停滞 w 轮就把部分名额留给没碰过的组件。
  选择侧：critic 评测前读 diff 筛 benchmark 专属逻辑；噪声底线 S′ ≥ S* − δ（δ 由重复评测基线估）；成本规则 ΔC ≤ β0 + β1·ΔS（Ridge 式）；无贡献组件剪枝（Lasso 式）。
  结果（Table 1/2/3/4）：Harvey LAB evolve 集 RRSI 只 +1.1（Meta-Harness +3.6），但 OOD 均值 39.7→43.6，其余四个方法 OOD 几乎不动或退；去掉选择侧正则 evolve 集涨到 91.5、OOD 掉到 41.0、token 涨一半；
  Terminal-Bench 2.1 Opus 4.8 74.2→80.2、SWE-bench Verified +1.8；Gemini 3.5 Flash 64.6→78.7；该 harness 原样换到 Gemini 3.1 Flash Lite 仍 11.2→14.6。
  → 对我们：四条选择侧规则可直接移植；按它的 δ 口径我们 100 局的噪声带约 0.13，候选评测要加量。
- **Darwin Gödel Machine** — arXiv 2505.22954。开放式存档 + 自指改写；消融：无自改退化、无存档只留最新。我们预算小，取
  Meta-Harness 的线性历史而不建存档，并在论文里说明为什么。
- **HyperAgents** — arXiv 2603.19461。连「如何改」的元过程本身也可改。属于研究过头的自指方向，不进我们主线。
- **SkillOpt** — arXiv 2605.23904。技能 = 冻结 agent 的外部状态；只接受在 held-out 上严格提升的有界编辑（add/delete/replace）。
  我们的候选接受规则（search 选、held-out 一次性验）与此同源。
- **ADAS** — arXiv 2408.08435。meta agent 用 GPT-4 搜、被搜出的 agent 用 GPT-3.5 评，且跨域跨模型迁移。
  → 「强 proposer + 弱执行模型」是本线的既定设定，不是我们的发明。
- **ALMA（Learning to Continually Learn via Meta-learning Agentic Memory Designs）** — arXiv 2602.07755。GPT-5-nano 上学、
  GPT-5-mini 上验迁移；记忆设计空间的元学习。
- 程序/图/OS 三种 harness 抽象（Probe 1 的对照组）：AFlow 2410.10762（图搜索）、DSPy 2310.03714（程序+编译）、
  MemGPT 2310.08560（OS 式分页）、Recursive LMs 2512.24601（递归调用）、AWM 2409.07429 / ACE 2510.04618 / MCE 2601.21557 /
  MemEvolve 2512.18746（记忆内容与结构进化）。它们都回答不了「模型会不会用」这个问题；我们的抽象是**外置精确信念状态估计器**
  （对标 §3 的 BSE 2609.10036 / CGDP 2605.07042）。

### 5.2 极限与分类法（Probe 2 / Probe 4 的锚）
- **Snell et al., Scaling LLM Test-Time Compute Optimally** — arXiv 2408.03314。简单题 test-time 算力占优、难题预训练占优。
  我们是它的 harness 版：GN（查表即可）被 harness 补满，CD（需推理）补不上。
- **Why Do Multi-Agent LLM Systems Fail? (MAST)** — arXiv 2503.13657。三类 14 种失效；「无或不完整验证」就是 SP 里判官噪声
  进注入文本的那条链。
- **More Agents Is All You Need** — arXiv 2402.05120。采样投票随 agent 数上升。进化 harness 必须打赢同 token 预算的多采样投票。
- **Open-Endedness is Essential for ASI** — arXiv 2406.04268。开放式 = 对观察者既新颖又可学习。只作度量参照，不追。
- **Prompt-Space Meta-Learning Does Not Transfer Across Users** — arXiv 2609.01615。frozen-LLM 负结果；元目标对打乱配对不敏感
  → 我们加 scramble 对照与 250 held-out。

### 5.3 "LLM 自己写出状态表示 / 信念跟踪代码"——从零臂的直接前驱（10-11 下载，PDF+txt 在 papers/）
- **WorldCoder** — arXiv 2402.12275（Tang, Key, Ellis）。任务：模型驱动的 agent。问题：从交互里学世界模型。方法：LLM 把世界模型写成 Python 程序，程序要解释已有交互且对可得奖励"乐观"（用程序与规划器之间的逻辑约束定义）；多个候选程序存成树，Thompson 采样选哪个让 LLM 改。结果：gridworld/任务规划上比深度 RL 样本高效、比 ReAct 省算力，可跨环境改代码迁移。
- **Code World Models for General Game Playing** — arXiv 2510.04542（DeepMind）。LLM 把规则和轨迹翻译成可执行世界模型（状态转移、合法动作枚举、终止判定），交给 MCTS/IS-MCTS；对不完全信息游戏让 LLM 合成"推断函数"（观测→可能的隐藏历史，CWM 当解码器，规则当正则）。10 个游戏里 9 个追平或超过 Gemini 2.5 Pro。→ 与 CD 最像：隐藏状态 + 枚举 + 规划器，只是他们的"隐藏状态"推断仍由 LLM 写代码完成。
- **PatchWorld** — arXiv 2605.30880。任务：文本 agent 的世界模型。问题：环境只给文本观测、不暴露真状态。方法：把离线轨迹变成可执行 Python **信念状态程序**（显式说"Wordle 的候选集"就是这种信念），用反例引导的归纳合成（CEGIS）修补，补丁只在 held-out 轨迹上重放保真度提高时接受。结果：7 个 AgentGym 环境上代码式决策最高（76.4%），不调 LLM 做前瞻。→ 最接近"从零发明信念跟踪"，但对象是环境动态，不是 harness。
- **LATM: Large Language Models as Tool Makers** — arXiv 2305.17126（ICLR'24）。强模型当"造工具者"写 Python 函数（含 BBH 的 logical deduction：枚举全排列筛约束），弱模型当"用工具者"；GPT-4 造 + GPT-3.5 用 ≈ 全程 GPT-4。→ "强者写求解器、弱者调用"的原型；CREATOR 2305.14318 同向。
- **Hypothesis Search** — arXiv 2309.05660。LLM 先生成自然语言假设再实现成程序、用样例筛（ARC）。假设空间由 LLM 提出而非枚举。
- **Text2World** — arXiv 2502.13092。LLM 从文本生成 PDDL 世界模型的基准（数百域、执行式指标）；最强模型仍有限。→ "LLM 写符号世界模型"的能力上限参考。
- **BED-LLM** — arXiv 2508.21184（已有）。用 LLM 自身采样假设估计期望信息增益；隐式信念而非枚举。→ 与我们"精确枚举"相对的一端。

### 5.4 2026 年 harness × 信念 的同期工作（10-11 下载）
- **EvoHarness-RL** — arXiv 2608.05446（UIUC + Meta）。把外部支持组织成 Belief / Progress / Experience 工作区，暴露 track / commit / recall / note 四个动作，先当推理时脚手架（+10.0 点），再用 cost-aware GRPO 学"何时用 harness"（比最强开源基线 +8.5）。→ 他们学的是策略如何调用 harness（改权重），我们固定权重进化 harness 本身；Belief 作为 harness 状态的提法与我们一致。
- **Measuring Harness-Induced Belief Divergence** — arXiv 2607.04528。同一任务、同一模型，换 harness 会改变 agent 的多步信念（belief-rollout 诊断 + 跨 harness 信念散度，分到达项与增长项）。→ 我们的 ex-ante eff / zero% 是在可枚举任务上对"信念被 harness 改变"的精确版本。
- **Towards a Belief-Based World Model for LLM Agents** — arXiv 2609.00455（UIUC/IBM）。论点：模拟接口不够，部分可观测下要把"已知/不确定"的信念直接暴露给策略；实验证明给信念能提升。→ 与我们的 symbolic/enum15 一级同论点；他们没有量"模型用没用上"。
- **Bayesian-Agent** — arXiv 2606.08348。把可复用技能当贝叶斯证据对象维护后验，跨 harness；有"从零在线进化"模式。→ 选择侧的另一种正则（后验而非噪声带）。
- Harness Engineering 源码研究 arXiv 2609.00006；Agentic World Modeling 综述 arXiv 2604.22748（只记录，未读）。

## 6. 对抗课程 / UED

- **PAIRED** — 原始的 regret-based 对抗环境生成。
- **ADD: Adversarial Environment Design via Regret-Guided Diffusion Models** — arXiv 2410.19715。
- **Stabilizing UED with a Learned Adversary** — arXiv 2308.10797。
- **ATLAS / Beyond Fixed Tasks: UED for Task-Level Pairs** — arXiv 2511.12706。
  在**任务和关卡两层**做 autocurricula，生成「可解但有挑战」的问题。
  → 与我们的「可解性证书 + 难度按 oracle 轮数 K 分带」是同一个关切，**必须对照**。
- **Beyond Simply Environment Scaling: Designing Effective Environment Distributions** — arXiv 2608.03571。
- **WebRL** — 自进化在线课程 RL 训练网页 agent。
→ 我们的 `min_H max_x [V_oracle − V_H]` 就是 minimax regret UED，**不是新框架**；
   新的是把它用在 harness（非参数）而不是策略权重上，且 oracle 可精确计算而非估计。

## 7. 机器教学（对偶视角，理论参照）

Goldman 的 teaching dimension：最优合作教师教会一个概念所需的最少样本（最坏情况）。
extended / subset / recursive teaching dimension 为变体。
→ 我们的「受约束对抗答题者」是它的对偶：**最坏可容许教师**。这个类比可以用来给
   「诚实但最省信息」的答题者一个学理位置，但**没有现成的 LLM 侧工作**，别过度声称。

---

## 待办（按优先级）

1. **精读 MastermindEval**，逐条排查 feedback compliance 与我们 dead-guess 率的口径差异。这是最近的对手。
2. **精读 Belief-State Engine**，确认贡献 3 还剩多少空间。
3. **精读 BED-LLM**，把「EIG 做决策 vs 做测量」的区别写成一段能过审的话。
4. 读 ATLAS 的可解性/难度控制，对照我们的 oracle 证书。
5. 从 Self-Improvements 那篇综述倒查 scaffold 自进化一支的完整引用链。
6. 补搜：主动推理的**诊断型**工作（不是方法、是测量），目前一篇都没搜到——若确实没有，这是贡献 1 的空位证据。
