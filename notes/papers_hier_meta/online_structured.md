# 在线/增量结构的 meta-learning:OSML 与 DPM-MAML 精读

- (1) **Online Structured Meta-learning (OSML)**:Huaxiu Yao, Yingbo Zhou, Mehrdad Mahdavi, Zhenhui Li, Richard Socher, Caiming Xiong. NeurIPS 2020. arXiv **2010.11545** (v1, 2020-10-22)。PDF:`~/Desktop/self_evolve_ai/papers/2010.11545.pdf`(正文 9 页 + 附录 A–D,已全读)。
- (2) **Reconciling meta-learning and continual learning with online mixtures of tasks**:Ghassen Jerfel*, Erin Grant*, Thomas L. Griffiths, Katherine Heller. NeurIPS 2019. arXiv **1812.06080**(读的是 v3, 2019-06-19;v1 题为 "Online gradient-based mixtures for transfer modulation in meta-learning")。PDF:`~/Desktop/self_evolve_ai/papers/1812.06080.pdf`(正文 + 附录 A–D,已全读)。注:arXiv v3 版面标注 "Preprint. Work in progress.",与 NeurIPS 终稿可能有出入,以下数字均来自 arXiv v3。OSML 把它称作 DPM [15]。

---

## 论文 1:OSML

### 任务
Online meta-learning(沿 FTML [Finn et al. 2019] 设定):任务 T_t 按序到达,每到一个任务先用累积的 meta-learner 适配、再把它纳入 meta-learner,目标是最小化 regret(Eq. 3)。本文专门针对 **heterogeneous / non-stationary** 任务分布(多个 disjoint modes)。

### 问题
FTML 只有一个全局共享的 meta-learner(一套初始化),异质任务下会被某个 mode "吸走"(Meta-dataset 上 FTML 在 Aircraft 好、Fungi/Flower 差,§4.2)。已有的 modulation 类方法(HSML、MMAML、TADAM 等)依赖训练好的 task representation network,在线场景下不现实;DPM [Jerfel] 遇到不相似任务就整套新建 meta-learner,共享粒度太粗(§1, §5)。

### 方法
- 把 meta-learner 拆成 **meta-hierarchical graph**:L 层(实验中 = 4 个 conv block 层),第 l 层有 B_l 个 **knowledge block**(每个 block = conv + BN + ReLU,附录 A.1)。
- **Search(路径构建)**:每层在 B_l 个已有 block 外再加 **1 个 novel block**,按 DARTS 方式做 softmax 连续松弛(Eq. 4):g_l = Σ_{b=1}^{B_l+1} softmax(o)_b · M_t(w0^b)(g_{l-1}),其中 M_t 是在 support set 上的一步 inner update。用 query loss 同时 meta-update 所有 block 初始参数 w0 与重要性系数 o(Eq. 5)。最后每层 **hard argmax** 选 b*_l = argmax o_b,连成 meta-knowledge pathway。
- **Update**:维护 task buffer B(装所有历史任务)。对 pathway 上每个被选中的 block,从 buffer 里抽 "**也使用该 block 的历史任务**" T_k,用 first-order 近似更新该 block(Eq. 6, Alg. 1 第 6–11 行),低层到高层逐层迭代。然后用 D_supp ⊕ D_query fine-tune 这条 pathway(Eq. 7),在 held-out D_test 上评测。
- 超参:β1–β5 = 0.001, 0.01, 0.001, 0.01, 0.001;每类 support/query 各 40,test 20;最后一层 linear 每任务随机重初始化(附录 A)。

### 结果
- Rainbow MNIST(56 个任务,7 色×2 尺度×4 角度;homogeneous):OSML 92.65±4.44% / AR 1.50;FTML 91.41±5.15 / 2.67;DPM 90.80±5.38 / 3.14;HSML 90.36±4.60 / 3.41;FT 87.71;NT 85.09(Fig. 2)。最终每层 block 数 1,1,2,2。
- Multi-filtered miniImageNet(Blur/Night/Pencil 各 20 个 5-way 任务,共 60,打乱顺序):Overall OSML **63.23±2.00%** (AR 1.67) vs HSML 59.25±2.36 (2.63)、DPM 58.45±2.44 (2.85)、FTML 56.63±2.50 (3.50)、FT 50.47、NT 48.35(Fig. 3)。分项 Night 65.25 vs HSML 57.57 差最大。
- Meta-dataset(Aircraft/Flower/Fungi,各 20 个任务,共 60;以 miniImageNet 预训练权重初始化):Overall OSML **65.00±2.46%** (AR 1.87) vs HSML 61.54±2.22 (2.85)、DPM 60.07±3.02 (3.15)、FTML 57.63±2.93 (3.92)、NT 55.23、FT 53.85(Fig. 4)。
- 容量对照(Fig. 6,multi-filtered):NT-Large 42.53(变差,过拟合)、FT-Large 52.46、FTML-Large 56.85,均远低于 OSML 63.23;Meta-dataset(附录 C Fig. 9):NT-Large 60.56、FT-Large 59.63、FTML-Large 59.90 vs OSML 65.00。论文结论:提升来自结构而非容量。注:Fig. 6 表中 NT 的 Overall 写成 53.32±2.30,与 Fig. 3 的 48.35 不一致,原文笔误或口径不同,论文未说明。
- 每任务样本数(Table 1,Meta-dataset):200 样本 OSML 55.57 vs FTML 52.08 / DPM 51.60;300:61.78 vs 56.40 / 57.42;400:65.00 vs 57.63 / 60.07。
- Rainbow MNIST 达到 90% 所需样本数(附录 D Fig. 10):OSML 多数任务更少(只有曲线,无汇总数字)。

---

## 论文 2:Jerfel et al.(DPM / 非参数 MAML 混合)

### 任务
Task-agnostic continual meta-learning:任务分布随时间漂移,**不告诉模型何时切换**(task change unobserved,§1),meta-learner 须在线检测分布漂移并调整复杂度;同时对旧分布保持性能(避免 catastrophic forgetting)。

### 问题
MAML 等把所有 inductive bias 压进一套超参 θ(初始化),等于假设任务"同等相关"。异质时会负迁移;非平稳时旧任务被遗忘。有限混合又引入"组件数 L 选多少"的模型选择问题,而非平稳下 L 不能固定(§5)。

### 方法
- 基于 "MAML = hierarchical Bayes" [Grant et al. 2018]:K 步梯度下降 ≈ 后验 mode 估计 φ̂_j,query 似然 ≈ 边际似然(Eq. 1)。
- **有限混合**(§3):L 个 MAML 初始化 θ^(1..L)。对每个任务、每个 component 都做 inner adaptation 得 φ̂_j^(ℓ);**E-step**:γ_j^(ℓ) = τ-softmax_ℓ(Σ_i log p(x_ji | φ̂_j^(ℓ)))——即"从该初始化出发适配后在 support set 上的似然"做 **soft** 责任分配(Eq. 2);**M-step**:各 θ^(ℓ) 按 γ 加权的 MAML 梯度更新。只用 support set 算分配,保持 episodic。
- **非参数(DPMM/CRP)**(§5, Subroutine 3):每个 meta-batch 都从全局先验 G0 抽一个候选新簇 θ^(L+1),同样做 inner adaptation。打分 ρ_j^(ℓ) = Σ log p(x|φ̂^(ℓ)) + log n^(ℓ)(已有簇),ρ^(L+1) = Σ log p(x|φ̂^(L+1)) + log ζ(新簇);τ-softmax 得 γ。**若 γ^(L+1) > ε 则永久 spawn**(L←L+1),否则去掉新簇重新归一化。CRP 先验 Eq. 3:p(z=ℓ) ∝ n^(ℓ),新簇 ∝ ζ。
- 实践修正(附录 D):阈值 ε = 0.95·T/(L+1)(T = meta-batch 大小)(C.2.2);n^(ℓ) 用**大小为 5 的滑动窗口**计数并乘可调系数(因为 loss 量级和 log n 不匹配,否则几轮后就再也 spawn 不了或一开始狂 spawn,D.2);ζ 固定为 meta-batch 大小(D.5;原文此处把浓度参数写成 α,与正文 ζ 记号不一致);**cool-down**:spawn 后固定若干轮不允许再 spawn,并 **冻结已有簇**的更新,让新簇离开先验(D.1);task-agnostic 版 cool-down = 1k 迭代(<15% 阶段长度),task-aware(oracle)版 cool-down = 整个阶段长度。注:D.1 原文两句都写成 "task-aware",这里按语义推断第二句应为 task-agnostic(11-25 核对原文时发现)。G0:先单簇训练若干轮,用它初始化;之后周期性用各非空簇按大小加权平均(或加权 KDE 平滑,略好)更新 G0(D.6.2)。不存任务分配,分配每次用当前参数即时重算(附录 A)。未用剪枝(D.4,讨论了两种剪枝启发式但未用)。
- 实现 TensorFlow,预分配权重 + mask(D.6.1)。

### 结果
- 标准 miniImageNet 5-way(Table 1,4-layer conv):MAML 48.70±1.84 (1-shot) / 63.11±0.92 (5-shot);本文 2/3/4/5 components 1-shot 49.60/51.20/50.49/51.46,5-shot 64.60/65.00/64.78/—;不随 L 单调提升。参数量 2 comp 65,546,5 comp 163,865。
- 连续 few-shot 回归(polynomial 4000 iter → sinusoid 3000 iter → sawtooth):在 iteration 4000 和 "7000 稍后" 各 spawn 一个新簇,每簇专门化于对应函数族并保持(Fig. 7);非参数方法对旧分布的 MSE 退化明显慢于 uniform / non-uniform 参数混合(Fig. 6,只有曲线,图例中的遗忘数值文本未能提取,论文正文未给具体数)。
- 连续 miniImageNet(Blur 7500 iter → Night 7500 → Pencil):catastrophic forgetting(前两个任务在训练结束时的平均精度下降)**Task-agnostic 1.12%、Task-aware(oracle) 0.768%**,对比 MT-net 8.46%、PLATIPUS 8.60%、VERSA 7.89%、Uniform 8.83%(Fig. 8)。训练结束时两种变体在 task 1、2 上都超过所有参数化 baseline;但在**活跃阶段**参数化 baseline(容量更高)精度更好(§6.2)。绝对精度只有曲线(纵轴约 30–40%),具体数论文未给。
- 每簇在测试集上拿到某一个滤镜分布的多数责任(Fig. 9)。

---

## 问题 a–f 对照

### a. 结构单元与分配
| | OSML | DPM (Jerfel) |
|---|---|---|
| 单元 | 每层的 knowledge block(conv+BN+ReLU 的初始参数);一个任务的结构 = 跨层 pathway(每层选 1 个) | 整套网络的 MAML 初始化 θ^(ℓ)(mixture component) |
| 粒度 | 层级/模块级,可跨族混搭(低层共享、高层分化) | 整模型级,一个簇 = 一整套初始化 |
| 分配打分 | DARTS 式 softmax 权重 o,用 query loss 梯度学出来(Eq. 4–5),即"用了这个 block 适配后 query loss 降多少"的可微代理 | 从该簇初始化 inner-adapt 后 support set log-likelihood + CRP 先验 log n^(ℓ)(Eq. 2, Subroutine 3) |
| soft/hard | 搜索时 soft,最终 **hard argmax** 每层选一个 | **soft** τ-softmax 责任,M-step 加权更新所有簇(τ=1 in miniImageNet) |

### b. 何时新建、冻结什么
- **OSML**:每个任务每层都放一个 novel block 参与竞争;若 argmax 落在 novel block 上,它就成为新 block(隐含准则 = novel block 的 o 超过所有已有 block)。**没有显式阈值/惩罚项**,论文未说明 novel block 如何初始化、未选中时是否丢弃。新任务到来时,搜索阶段所有 block 都参与 Eq. 5 的 meta-update(论文写法即对 w0,t 全部更新,未说明是否只在搜索临时副本上做);正式的知识更新只针对 pathway 上被选 block,并用"同样使用该 block 的历史任务"回放混合更新。未选 block 不动。旧任务的 pathway 不重算(论文未提重新搜索旧任务)。
- **DPM**:每个 meta-batch 都试一个从 G0 抽的候选簇;**γ^(L+1) > ε = 0.95·T/(L+1)** 才永久加入;CRP 惩罚 log ζ vs log n^(ℓ)(滑动窗口 5 + 系数)抑制过度 spawn。spawn 后 cool-down:禁止再 spawn + **冻结已有簇**;cool-down 外所有簇按 γ 软更新(不冻结)。旧任务分配不存储、随时用当前参数重算(附录 A);G0 周期性重算为簇均值。没有对旧簇的整体重训。

### c. 防遗忘 / 防负迁移
- OSML:(1) 模块隔离:新族可以占新 block,不覆盖他族 block;(2) **task buffer 回放**:更新共享 block 时从 buffer 抽共用该 block 的历史任务(Alg. 1 第 8 行)。论文**没有报告**对旧任务的 backward transfer/遗忘指标,评测是每个任务到达时的即时精度。
- DPM:(1) 责任 γ 门控梯度,只有对该任务"似然高"的簇被大幅更新 → 限制负迁移;(2) 新分布 spawn 专门簇;(3) cool-down 冻结旧簇。无回放 buffer(stochastic 设定不访问过去数据)。显式报告 CF 指标(1.12% vs 7.89–8.83%)。

### d. 任务流与 benchmark
- OSML:Rainbow MNIST 56 任务(同质,顺序给出);multi-filtered miniImageNet 与 Meta-dataset(Aircraft/Flower/Fungi)各 60 任务、**全部打乱**(族交错到达,不是按族顺序)。数字见上。HSML/DPM/FTML 均为 baseline;OSML 全面领先。
- DPM:**按族顺序分阶段**到达(regression 3 段;miniImageNet Blur→Night→Pencil 各 7500 iter),task-agnostic(不告诉边界)。baseline:uniform/non-uniform 参数混合(L=3)、MT-net、PLATIPUS、VERSA;标准 miniImageNet 对 MAML。无 HSML。

### e. 学到的结构分析
- OSML:Fig. 5(Meta-dataset)/Fig. 8(multi-filtered)给每个 block 被各子数据集选中的频次:部分 block 被单一族主导,部分跨族共享;Fungi 与 Flower 更常共享(如 layer 1 block 7,解释为背景和形状相似);Blur 与 Night 比各自与 Pencil 更常共享(解释为保留更多原图信息);Rainbow MNIST 最终 1,1,2,2 个 block(低层全共享)。最终 block 数从图上读:Meta-dataset 约 layer1 9 个、layer2-3 各 6 个、layer4 8 个(图横轴,非正文数字)。
- DPM:Fig. 7/9 的 meta-test 责任随时间曲线,每簇基本对应一个族,spawn 时刻与分布切换对齐。无跨族共享分析(结构上也不支持部分共享)。

### f. 代码
- 两篇论文正文/附录均**未给代码链接**;网络检索(2026-11-25)也未找到官方仓库。论文未说明。

---

## 对我们的借鉴(harness 代码组件库,~17 benchmark / ~100 subtask,随时新增)

两篇各取一半:**OSML 的粒度**(组件级、可跨族混搭,而不是 DPM 的"整套 harness 一个簇")+ **DPM 的新建判据**(显式的 "新组件 vs 已有组件" 似然竞争 + CRP 惩罚 + 阈值 + cool-down)。OSML 自己没有新建惩罚,也没有报告遗忘,这两块都得从 DPM 和回放那边补。

1. **Block = 一个代码组件**:有接口的 harness 单元(一个 prompt 段、一个 tool wrapper、一个 parser/verifier、一个 memory policy),**带槽位(slot)**,对应 OSML 的"层"。每个 slot 下有若干竞争实现,一个任务的 harness = 每个 slot 选一个(或空)= pathway。这样既能低层共享(如通用 answer-extraction),又能高层分化。
2. **分配打分 = 实测迁移增益**(OSML 中 o 的不可微版本,DPM 中 log p(x|φ̂^(ℓ)) 的对应物):对新 subtask s 和 slot 内候选 c,score(s,c) = 在 s 的 dev/support 小批上**"插入 c 后再做 k 步 inner-loop 进化"**的分数 − 基线 harness 同等预算进化后的分数(对应 DPM"从该初始化适配后再评估",而不是零样本插入)。再加先验项 λ·log n_c(n_c = 近期窗口内采用 c 的 subtask 数,学 DPM 用滑动窗口和系数,避免数量级失配)。membership 用 τ-softmax 得连续 γ(s,c)(对应我们的连续任务成员);部署时每 slot 取 argmax(OSML 的 hard 选择),γ 只用于决定"谁的编辑会回流到 c"。
3. **新建判据**:每个 slot 永远放一个"空/novel"候选 = 该 subtask 自己 inner loop 进化出的本地版本。新组件只在以下条件同时满足时 spawn:(i) novel 候选的 γ > ε(用 DPM 的 ε = 0.95·T/(L+1) 型阈值,T = 同批评估的 subtask 数),或说本地版对最好已有组件的增益 > 噪声带(用重复评估的标准误定,比如 > 2σ);(ii) **复现性**:同一编辑模式(diff 聚类)在 ≥ m 个 subtask 的 inner loop 中独立出现,且至少跨 2 个 benchmark(否则只是该 subtask 的私有补丁,留在本地,不进库)。即"在单个任务上胜出"只算候选,"反复出现"才晋升,这正是我们 outer loop 的晋升机制,CRP 的 log ζ 在这里就是晋升门槛。
4. **冻结**:新加 benchmark 时只做局部计算:(a) 新任务对所有现有组件打分(只读);(b) 只更新新任务 argmax 选中的组件以及新 spawn 的组件;(c) spawn 后 cool-down:K 轮内不再 spawn,且**冻结已有组件**,只让新组件进化(DPM D.1)。其余组件和其他 subtask 的 pathway 不重算;旧 subtask 的分配只在其被选组件被修改时才局部重评(DPM 是每次即时重算,我们评估贵,改为"被动失效"触发)。
5. **防回归**:(a) **回放门控**(OSML Alg. 1 第 8 行的代码版):共享组件 c 的任何修改,必须在"当前使用 c 的所有 subtask"(按 γ 加权抽样,或至少每个 benchmark 一个)的冻结 dev 集上跑一遍,任一下降超过噪声带就拒绝合并,改为 fork 出新组件给新任务(即负迁移时自动 spawn,而不是覆盖);(b) 组件版本化、不可变:修改 = 新版本,旧 subtask 固定在旧版本,直到新版本在它们上不劣才迁移;(c) 报告 backward transfer 指标(学 DPM 的 CF = 旧 benchmark 在新 benchmark 加入后的平均下降),这是 OSML 缺的。
6. **剪枝/合并**(DPM D.4 只讨论未用):长期无人选的组件删;两组件在验证 subtask 上 γ 分布的 odds-ratio 很接近就合并,防止库膨胀。

需注意的差异:两篇的 inner adaptation 是几步梯度,便宜;我们的 inner loop 是 LLM 驱动的进化,每次打分都贵,所以"每任务对每候选都 inner-adapt"(DPM 的做法)不可行,需要先用便宜信号(零样本插入分数、编辑 diff 相似度)筛 top-k 候选再做完整打分。
