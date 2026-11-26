# TreeMAML — Meta-Learning with MAML on Trees

- 标题:Meta-Learning with MAML on Trees(文中方法名 TreeMAML)
- 作者:Jezabel R. Garcia, Federica Freddi, Feng-Ting Liao, Jamie McGowan, Tim Nieradzik, Da-shan Shiu, Ye Tian, Alberto Bernacchia(MediaTek Research, Cambridge UK)
- arXiv:2103.04691v1(2021-03-08,cs.LG),arXiv comment:"Updated version of paper in EACL workshop: Adapt-NLP 2021"
- 正式发表版本:Adapt-NLP 2021(EACL workshop),ACL Anthology 2021.adaptnlp-1.8,题为 "Cross-Lingual Transfer with MAML on Trees"
- 本地:`~/Desktop/self_evolve_ai/papers/2103.04691.pdf` / `.txt`(pdftotext -layout)
- 篇幅:6 页,**无附录**,全文已读(含 Algorithm 1/2、Table 1/2、Fig 1-3)
- 检索说明:arXiv 上 "TreeMAML" 只命中这一篇。调用方提到的 "TreeMAML: Model-Agnostic Meta-Learning for Very Related Tasks with Hierarchical ..." 这个标题,arXiv/OpenReview 都没查到。OpenReview 上 ICLR 2022 Paper817 的作者回复里把它当作相关工作来区分(其方法用 learning trajectory 聚类后条件化全局初始化),说明这篇确实被当作 "TreeMAML" 引用。

## 任务

Few-shot meta-learning,任务之间呈**层级(树状)关系**。两个实验场景:
1. 合成:多维线性回归 y = Σ_{i=1}^{64} P_i x_i + η,任务参数 P 从 4 个层级排列的簇中采样(C1=2, C2=4),x_i ~ U[-5, 5],噪声 0.01(§4,Table 1 标题)。
2. 跨语言 NLI:XNLI,取 15 种语言中属于同一棵系统发生树的 12 种(en fr es de el bg ru vi th zh hi ur;Fig 3 的简化语言树,深度 3)。**任务 = (语言, genre) 组合**,genre 共 10 种;meta-training 每个任务 3-shot;每次把一种语言当 target,其余 11 种作 auxiliary(§5)。

## 问题

- MAML 对所有任务一视同仁,"does not diversify task relationships according to their similarity"(§1);无关任务之间共享信息会伤性能(Abstract)。
- 已有利用任务关系的 meta-learning(HSML, Yao et al. 2019;Liu et al. 2019)"are not model-agnostic"(§2);Jerfel et al. 2019、Katoch et al. 2020 改了 MAML 但只有一层聚类,"cannot be represented by a tree structure"(§2)。
- 目标:给 MAML 一个**沿树共享 inner-loop 更新**的、仍然 model-agnostic 的扩展;并在树未知时能从数据里学出树。

## 方法

**核心:inner loop 的第 k 步在第 k 层簇内平均梯度,逐层下行。**

- 第 k 步任务被分成 C_k 个簇,同簇任务用簇内平均梯度更新(Eq. 5)。例:K=3, C1=2, C2=4, C3=8(§3.1,Fig 1b)。
- 按簇写的更新(Eq. 7):θ_{c,k} = θ_{p_c,k-1} − α/(n_t|T_c|) Σ_{i∈T_c} Σ_j ∇L(θ_{p_c,k-1}; D_ij),p_c 为父簇。任务最终参数 θ_{τ_i}(ω) = θ_{c_i,K}。
- 退化:C_k = m(任务数)恒成立时即 MAML;C_k = 1 时每步都用全体平均梯度(§3.1)。
- 直觉:每个任务数据少、梯度噪声大,相似任务求和提高信噪比(§3.1)。
- Outer loop 与 MAML 相同:ω ← ω − β/(m n_v) ΣΣ ∇_ω L(θ_{c_i,K}(ω); D_ij)(Algorithm 1)。只有**一个全局初始化 ω**,簇节点上没有持久参数;θ_{c,k} 是每个 episode 内临时算出来的。
- 两个版本(§3.1):
  - **Fixed TreeMAML**:树由先验知识给定(如语言系统发生树),C_k 由树决定。
  - **Learned TreeMAML**:树未知,用层级聚类学;C_k 由聚类在每一步决定。聚类特征 = 每个任务在当前节点参数处的**平均 loss 梯度** g_ik = (1/n_t) Σ_j ∇L(θ_{c_i,k}; D_ij)(Eq. 8,依据 Task2Vec, Achille et al. 2019)。
- 聚类算法(§3.2,Algorithm 2):改自 Menon et al. 2019 的 **Online Top-Down (OTD)** 层级聚类,逐个插入任务,两处修改:
  1. 最大深度 D = inner step 数,树的每层对应一个 inner step;
  2. 非二叉近似:超参 ξ,新任务与节点平均相似度相差超过 ξ·σ_T(簇内相似度标准差的倍数)时才另立新簇,否则作为该节点的又一个孩子,所以节点可以有多于两个孩子。
  - 插入规则(Algorithm 2):A 为空或只有一个孩子→直接加入;若 ω(A∪{x}) > ω(A)→找最相似孩子 x*,未到最大深度则递归 OTD(x*, x) 生成新节点,到了就直接加入;若 ω(A∪{x}) < ω(A) − ξσ_T→当前节点与新任务成为新簇的两个孩子;否则加入孩子集合。
  - 相似度:合成实验用 cosine similarity(§4.2)。
  - 选在线聚类的理由(§3.2):测试时计算效率与可扩展性;"When a new task is evaluated, we reuse the tree structure generated for a training batch and add the new task. This process saves us from computing a new task hierarchy from scratch for every new task."
- 测试时适配(§3.1 末):"we first reconstruct the tree structure by using a batch of training data and then we introduce the new task." NLI 中 "We add the new target task to the original distribution of tasks, we apply the TreeMAML algorithm and evaluate the model on the target language test set"(§5)。
- NLI 设置:只对 Multi-BERT 顶层(layer 12)做 TreeMAML,4 个 inner step;MAML 用相同 step 数(§5)。Fixed 版本举例:target=德语时第 1 步用其余所有语言,第 2 步用 Indo-European 分支的训练语言,"The third and last steps uses only English"(§5.1)。
- 合成 Learned 版本:最大深度 2,于是 3 个 inner step,最后一步 task-specific;MAML 也设 3 步(§4.2)。

## 结果

**合成线性回归(Table 1,MSE ±95% CI,400 个 meta-testing 任务,noise 0.01):**

| Model | K=5 | K=10 | K=20 |
|---|---|---|---|
| Baseline(所有任务一起训,无 meta-learning) | 1.293 ± 0.074 | 1.055 ± 0.047 | 1.139 ± 0.061 |
| MAML | 1.025 ± 0.068 | 0.950 ± 0.048 | 0.785 ± 0.028 |
| Fixed TreeMAML | 0.621 ± 0.038 | 0.535 ± 0.024 | 0.483 ± 0.016 |
| Learned TreeMAML | 0.758 ± 0.047 | 0.510 ± 0.024 | 0.495 ± 0.018 |

- Fig 2(Fixed vs MAML vs Baseline,K=4…128):"TreeMAML outperforms MAML, especially when the number of tasks data points K is low"(§4.1),图无具体数值。
- §4.2 说 learned 在数据点多时优于 fixed(梯度噪声小→聚类准)。注意:Discussion 里写成 "when the number of points of the test task is small, the learned algorithm outperform the fixed tree",与 Table 1(K=5 时 fixed 0.621 < learned 0.758)矛盾,以表为准。

**XNLI(Table 2,准确率 %,3 次平均,std 均 <1%):**

| | en | fr | es | de | el | bg | ru | vi | th | zh | hi | ur | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Multi-BERT(两 auxiliary 语言) | 81.94 | 75.39 | 75.79 | 73.25 | 69.54 | 71.60 | 70.84 | 73.23 | 61.18 | 73.93 | 64.37 | 63.71 | 71.23 |
| XMAML(两 auxiliary 语言) | 82.71 | 75.97 | 76.51 | 74.07 | 70.66 | 72.77 | 72.12 | 73.87 | 62.5 | 74.85 | 65.75 | 64.59 | 72.20 |
| Multi-BERT(全部语言) | 83.56 | 76.22 | 76.89 | 73.11 | 72.89 | 72.89 | 71.33 | 74.67 | 57.56 | 74.89 | 63.11 | 63.33 | 71.70 |
| MAML(全部语言) | 83.11 | 78.22 | 77.11 | 73.56 | 69.33 | 71.78 | 71.33 | 74.22 | 57.33 | 75.11 | 63.33 | 63.78 | 71.52 |
| Fixed TreeMAML | 84.67 | 79.78 | 78.22 | 76.89 | 72.00 | 74.22 | 73.33 | 74.44 | 59.56 | 79.11 | 66.00 | 66.89 | **73.76** |
| Learned TreeMAML | 84.22 | 77.33 | 79.78 | 78.00 | 71.56 | 73.78 | 74.00 | 74.89 | 59.78 | 76.44 | 65.11 | 65.56 | 73.37 |

- 平均:Fixed 73.76 vs MAML 71.52(+2.24)vs XMAML 72.20(+1.56)。正文称比 XMAML "average of ∼3% better"(§5.1)和 "improvement in accuracy ∼3%"(§6),与表中均值差 1.56 对不上,论文未解释口径。
- XMAML 只用两种 auxiliary 语言,TreeMAML 用全部 11 种;Table 2 标题自己承认 "The difference in the amount of data used for training may account for a part of the difference"。
- 失败点:Greek 上 baseline 72.89 > Fixed 72.00;Thai 上 TreeMAML(59.56/59.78)低于 XMAML 62.5。作者归因为语言树过于简化("generic 'Asian' language tree")(§5.2)。
- **没有与 HSML 的数值比较**,HSML 只在 related work 里出现。

## 逐问作答

**a. 树怎么建、更新怎么沿树共享**
- 两种来源:Fixed=先验给定(语言系统发生树,Fig 3,深度 3);Learned=在线层级聚类 OTD 的改版(Algorithm 2),特征=各任务在当前节点参数处的平均 loss 梯度(Eq. 8),合成实验用 cosine similarity。
- 深度固定:最大深度 D = inner step 数(§3.2);分叉数不固定,由 ξ 控制非二叉(§3.2)。
- 更新规则:第 k 个 inner step 中,每个簇 c 从父簇参数 θ_{p_c,k-1} 出发,用簇内所有任务梯度的平均走一步(Eq. 7);越往下簇越细,最后一层可为单任务。

**b. 树固定还是训练中重建;软/硬归属**
- Learned 版**每个 batch、每个 inner step 都重建**:"The relationships between languages and genres is learned at each step of gradient descent, for each batch of data. Therefore, the tree for one batch can be different from the tree for the next batch"(§5.2)。Algorithm 1 在每步 k 后 "regroup tasks into C_k clusters"。Fixed 版整个训练中不变。
- **硬归属**:每任务在第 k 步属于唯一簇 c_i^k,簇内等权平均(Eq. 7);Discussion 把 "weighted averaging of the gradients" 列为 future work(§6)。

**c. 任务族的采样平衡**
- 论文未说明。Algorithm 1 只写 "sample batch of i = 1 : m tasks {τ_i} ∼ p(τ)";NLI 只说每任务 3-shot。没有按族/按语言做重采样或加权的描述。

**d. 增量/在线**
- 机制本身支持把单个新任务插进已有树:OTD 是在线的,"we reuse the tree structure generated for a training batch and add the new task. This process saves us from computing a new task hierarchy from scratch for every new task"(§3.2);测试时 "first reconstruct the tree structure by using a batch of training data and then we introduce the new task"(§3.1)。
- 所以在 meta-test 阶段新任务(包括 meta-training 没见过的 target 语言)不需要重训 ω,也不需要从头重聚类;但插入要用到新任务在**每一层父节点参数处**的梯度,且插入后该路径上所有簇的平均梯度变化→该分支下游的 θ_{c,k} 都要重算(本来每个 episode 都重算)。
- 新任务族:论文没有专门讨论或实验"新族插入",也没有讨论树长期累积、族漂移或 ω 是否需要再训练。由于树本来就是 per-batch 临时结构,"永久树+增量维护"在本文里并不存在;能持续积累的只有 ω。

**e. 评测**
- 合成:400 个 meta-testing 任务,与训练任务来自同样的 4 簇(=held-out tasks,同族)。
- XNLI:target 语言整体不参与 meta-training(其余 11 种做 auxiliary)→ 某种意义上是 held-out family(语言),但该语言在固定树里的位置事先已知。
- Benchmarks:合成多维线性回归、XNLI 12 语言。主数字见上表:合成 MSE 约为 MAML 的一半(Discussion "halving the MSE");XNLI avg Fixed 73.76 / Learned 73.37 vs MAML 71.52。无 HSML 对比。

**f. 对学到的树结构的分析**
- 只有定性讨论,没有可视化,也没有和真树做定量比较:learned 有时学到与 fixed 相同的树(结果几乎一样),有时把任务分到错误分支(更差),有时可能找到同 genre 跨语言的关系(更好)(§5.2,§6)。合成实验没有报告聚类恢复真簇的准确率。

**g. 代码**
- 论文中未给出代码链接;GitHub 搜 "treemaml" 无结果。论文未说明。

## 对我们的借鉴

我们的设定:meta-learn 的对象是 LLM agent harness 的**代码组件库**,不是神经参数;组件对任务是连续 membership;outer loop 把 per-task inner-loop 演化中反复出现的 edit 提升为共享组件;约 17 个文本 benchmark、约 100 个 subtask;结构要能在线扩展、局部适配。

**能搬过来的机制**
1. **"共享更新信号沿树逐层下行"**:把 Eq. 7 换成代码版本——第 k 层节点汇总其子树内各任务 inner-loop 产生的 edit(diff/patch 或失败轨迹的修复建议),只有簇内多数任务一致的 edit 才在该节点生效,下层逐步特化,叶子是 task-specific edit。这对应 "promote recurring edits" 的分层版本:edit 在多大的子树内 recurring,就提升到哪一层,而不是只分 "全局 / 任务专属" 两档。
2. **聚类特征 = 适配信号而非任务描述**:TreeMAML 用的是梯度(=任务想把参数往哪推)。我们的对应物是任务的 edit 向量 / 失败模式签名 / 组件效用向量(各组件在该任务上的 ablation 增益),比 benchmark 名字或 prompt embedding 更贴近 "该不该共享"。
3. **退化端点做对照**:C_k=m→纯 per-task 演化;C_k=1→全局共享库。这两端正是我们需要的 baseline。
4. **深度 = 适配步数**:可以把树的层数和 inner-loop 演化预算(几轮 edit)绑定,每轮在更细的子树内做共享。

**树设计对在线扩展是支持还是阻碍**
- 支持的部分:OTD 是在线 top-down 插入,单个新任务沿树往下走、只在一条路径上局部修改,不需要全局重聚类。这正好符合我们 "新 benchmark 随时加入、结构局部适配" 的需求。
- 阻碍的部分:
  - 本文的树是 **per-batch 临时的**,每步重建,没有持久节点状态,也没有节点级参数(只有一个 ω)。我们要的是持久组件库,本文没有给出"树长期维护"的任何机制(合并、拆分、重平衡、删除过时节点都没有)。
  - **硬归属 + 等权平均**:与我们的连续 membership 冲突;一个 subtask 同时用到多个分支的组件时,树结构表达不了(这实际上要 DAG 或软层级)。
  - OTD 是**插入顺序相关**的贪心算法,早插入的任务决定上层切分,随着 benchmark 不断加入,上层会逐渐失配,论文对此没有任何讨论或修复手段。
  - 深度固定 = inner step 数,不适合 benchmark 规模不一(有的 benchmark 一个 subtask,有的十几个)。
- 如何做成增量的(建议,非论文内容):
  1. 树节点持久化,每个节点存:组件集合、成员任务的 membership 权重、该节点 edit 统计(出现次数/跨任务一致率)。
  2. 新任务:先跑短 inner loop 得到 edit 签名,用 OTD 式 top-down 插入,但改成**软插入**——对每个候选子节点算相似度,按 softmax 给出多个 membership,只在相关路径上更新统计。
  3. 只在受影响路径上做局部 promote/demote:新任务带来的 edit 若在某个节点达到 recurrence 阈值就提升为该节点组件;只重新验证该节点子树内的任务(局部 regression test),不跑全量。
  4. 周期性局部重构:监测节点内相似度方差(相当于 OTD 的 ξσ_T 判据),超阈值时只拆分/合并该节点,而不是全树重建——这是 Menon et al. 2019 在线层级聚类框架内可以做的局部旋转,TreeMAML 没用上。
  5. 用 HSML 式的软分配或 mixture(Jerfel et al. 2019 的 online mixture 本来就面向 continual learning)补上 TreeMAML 缺的连续 membership;TreeMAML 只提供"沿层级分步共享"这一骨架。
- 需要注意的证据强度:本文只有一个合成实验加一个 NLI 实验,只改 BERT 顶层,没有 HSML 对比,没有树结构质量分析,正文 "~3%" 与表格均值差 1.56 不一致。可借鉴的是机制,不能当作 "层级共享一定优于平铺共享" 的强证据引用。
