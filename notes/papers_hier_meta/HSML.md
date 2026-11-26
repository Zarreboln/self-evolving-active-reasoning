# HSML: Hierarchically Structured Meta-learning

- Huaxiu Yao, Ying Wei, Junzhou Huang, Zhenhui Li. ICML 2019 (PMLR 97)
- arXiv 1905.05301 (已核对 ID 与标题；本地为 v2, 2019-11-15)
- PDF/全文: `~/Desktop/self_evolve_ai/papers/1905.05301.pdf` / `.txt`(正文 + 附录 A–J 全读)
- 代码: https://github.com/huaxiuyao/HSML (静态版), https://github.com/huaxiuyao/HSML_Dynamic (continual 版)，TensorFlow 1.x，基于 cbfinn/maml。论文正文没给 URL，是从作者 GitHub 核实的；下文凡标"代码"的都是我读代码得到的，论文里没有写

---

## 任务

Gradient-based meta-learning(在 MAML 上做)，场景是任务分布**异质**：任务来自多个 family/数据集。目标是学一个 meta-learner(θ0 初始化 + 聚类/门控参数)，让新任务只用几步梯度就适配好。

- Toy regression: 4 个函数族(Sinusoids / Line / Cubic / Quadratic，参数范围见 §6.1)，5-shot / 10-shot，MSE。
- Few-shot image classification: 作者自建的 "Multi-Datasets" benchmark = Bird(CUB) / Texture(DTD) / Aircraft / Fungi，各自按类切分 meta-train/val/test(Bird/Aircraft/Fungi 64/16/20，Texture 30/7/10；附录 B)，5-way 1/5-shot，base learner 为 4-block conv。
- 另有 MiniImagenet(附录 D)作对照，以及 continual adaptation 设置(新 family 在 meta-training 中途加入)。

## 问题

- 全局共享初始化(MAML、Meta-SGD)只有一个 θ0，处理不了任务异质性和不确定性；
- 逐任务定制(MT-Net、BMAML、MUMOMAML 等)又丢掉了相近任务之间的泛化共享(§1, Fig.1)；
- 所以需要介于两者之间的**按 cluster 定制**：同一 cluster 内的任务共享知识，不同 cluster 各自定制，而且要用层次结构表达多层次的任务关系(§4.2 用基因交互多层聚类来论证"单层分组不够")；
- 任务按顺序到来时，任务关系会变化，结构也要能随之扩展(§1, §4.3 Continual Adaptation)。

## 方法

三段式(Fig.2)：task representation → hierarchical task clustering → knowledge adaptation，端到端训练。

**1. Task representation (§4.1)**
- 先用 F(x,y) 对每个样本做预嵌入(同时嵌 feature 和 label，刻画联合分布)。toy 用一层 FC；图像用 2 conv + 2 FC(附录 C)。
- 两种 aggregator：
  - PAA (pooling autoencoder)：g_ij = FC_enc(F(x,y))，g_i = Pool_j(g_ij)(max/mean)，重构损失 L_r = Σ‖FC_dec(g_ij) − F(x,y)‖²，见 Eq.(2)(3)；
  - RAA (recurrent autoencoder)：样本依次喂进 LSTM/GRU encoder，再由 decoder 反向重构；g_i = (1/n^tr) Σ_j g_ij(Eq.4–5)，对样本做随机排列来缓解顺序敏感。
- 默认用 RAA(§6；附录 G 表 9：1-shot 时 RAA 均值 49.35% 对 MPAA 46.14%，shot 越多两者越接近，8-shot 为 65.01 对 64.97)。
- task representation size：toy 40，图像 128(表 4)。

**2. Hierarchical task clustering (§4.2)**
- 每层交替做 assignment 和 update。第 0 层 h^0 = g_i，K^0 = 1；最顶层 K^L = 1。
- Assignment(Eq.6)：**soft**，对到可学习 cluster center 的 Euclidean 距离做 softmax：p^{k^l→k^{l+1}} = exp(−‖(h^{k^l} − c_{k^{l+1}})/σ^l‖²/2) / Σ exp(...)，σ^l 是每层的 scaling factor。论文给出选 soft 的两条理由：任务组之间有重叠/连续共享；可微，能端到端训练。
- Update(Eq.7)：h^{k^{l+1}} = Σ_{k^l} p^{k^l→k^{l+1}} · tanh(W^{k^{l+1}} h^{k^l} + b^{k^{l+1}})，每个 cluster 有自己的 W、b。
- cluster center 随机初始化，随训练一起学(§4.2 末，附录 C)。
- 规模设置：§6 说"高层 cluster 数取低层的一半"(沿用 DiffPool)，toy 与图像都用 3 层 4-2-1。
- [代码] `lstm_tree.py` 里只有第一层用 cluster center + 距离 softmax，σ 是常数(multidataset 取 10、toy 取 2)；更上面的层由每个父节点的可学习向量打分(tree_type 1 为线性打分，2 为距离打分)再 softmax 作 gate，最后在 root 处求和。这和 Eq.6"每层都用到 center 的距离"写法不完全一致。

**3. Knowledge adaptation (§4.3)**
- Parameter gate(Eq.8)：o_i = σ(W_g (g_i ⊕ h^L_i))，⊕ 是拼接，σ 是 sigmoid。
- 初始化调制：θ0i = θ0 ∘ o_i(逐元素乘，即对共享初始化加 mask)，再从 θ0i 出发做 inner-loop 梯度步。
- 总目标(Eq.9)：min_Θ Σ_i L(f_{θ0i − α∇L}, D^te_i) + ξ L_r(D^tr_i)，Θ 包括 θ0、聚类参数和 W_g。ξ = 0.01(表 4 中记作 γ)。
- 理论部分(§5、附录 A)：把 θ0t 一阶 Taylor 近似写成 Σ_k B̂_k θ0，再借 Kuzborskij & Lampert 2017 的 on-average stability 推出比 MAML 更紧的泛化界(Thm 1 对应 convex，Thm 2 对应 non-convex)。B̂_k = I 时退化为 MAML。

**4. Continual adaptation (§4.3, Alg.1)**
- 每 Q 个训练任务计算一次平均 loss L̄；如果 L̄_new > μ·L̄_old，就"increase the number of clusters"，新 cluster 的参数随机初始化。
- toy：μ = 1.25，Q = 1000；分类：改用平均**训练准确率**的变化来判断，μ = 0.85 < 1，Q = 100(附录 C，表 4)。
- "For continual scenario, we add one cluster every time"(附录 C)。
- [代码 HSML_Dynamic/main.py] 触发扩展时，先存下全部 trainable 变量，再 `tf.reset_default_graph()`，以 cluster_layer_0 + 1 重建整张图，同名旧变量拷回，新变量随机初始化，然后**所有参数一起继续训练，没有冻结任何部分**。扩展的只有最底层(cluster_layer_0)，上层 cluster 数(cluster_layer_1 = 2)不变。另有一个论文未写的冷却期 `online_change = 8000` 次迭代，两次扩展之间至少隔这么久。

**其他超参(表 4)**：meta-batch toy 25 / 图像 4；α = 0.001；β toy 0.001 / 图像 0.01；inner steps 5；image embedding 64。

## 结果

**Toy regression(表 1，4000 tasks，MSE ± 95% CI)**
- 5-shot：HSML 0.856±0.073，最强 baseline MUMOMAML 1.096±0.085，MAML 2.205±0.121。
- 10-shot：HSML 0.161±0.021，MUMOMAML 0.256±0.028，MAML 0.761±0.068。

**Multi-Datasets 5-way(表 2，每数据集 1000 tasks)**
- 1-shot 平均：HSML 49.35%，MUMOMAML 46.50%，BMAML 45.89%，MAML 44.77%。HSML 分项：Bird 60.98 / Texture 35.01 / Aircraft 57.38 / Fungi 44.02。
- 5-shot 平均：HSML 62.39%，MUMOMAML 59.41%，MAML 57.78%。分项：Bird 71.68 / Texture 48.08 / Aircraft 73.49 / Fungi 56.32。
- MiniImagenet 5-way 1-shot(表 5)：HSML 50.38±1.85%，与 MetaSGD 50.47、Relation Net 50.44 等持平。论文的解释是这个 benchmark 异质性不明显。

**Leave-one-dataset-out(附录 E 表 6，5-way 1-shot，4000 tasks)**
- 平均：HSML-RTG 34.41%，MUMOMAML 33.28%，MAML 32.43%。分项依次为 Bird 42.54 / Texture 30.90 / Aircraft 31.23 / Fungi 32.98。所有方法都比不留出时低很多。

**cluster 数(表 3 / 表 8，1-shot)**
- 平均：(2,2,1) 47.65%，(4,2,1) 49.35%，(6,3,1) 48.45%，(8,4,4,1) 49.08%。
- 正文把最后一档写成 "(8,4,4,1)" 和 "(8,4,2,1)" 两种，前后不一致；表 8 写的是 (8,4,4,1)。结论是太少不够，加层没有收益，只多参数。

**消融(附录 H 表 10，1-shot，以 Bird / Texture / Aircraft / Fungi 顺序)**
- A1(每个 GT 数据集单独训一个 MAML，作为 oracle hard clustering)：58.67 / 33.46 / 55.81 / 43.50，四项都低于 HSML。
- A2a(先单独预训练 task embedding 再冻住)：56.97 / 29.12 / 45.71 / 40.92，说明 meta-gradient 对 embedding 的训练是有用的。
- A3a(去掉 clustering)：58.22 / 33.30 / 55.35 / 42.68；A3b(改成 flat clustering)：58.08 / 34.26 / 56.11 / 43.38。层次结构相对 flat 只多出约 1–3 个点，而且多数项落在 CI(±1.4%)以内。
- A4a(gate 只吃 cluster 表示)：59.01 / 33.69 / 56.69 / 42.88；A4b(gate 换成 FiLM)：61.02 / 34.87 / 56.53 / 44.56，与 HSML 相当。
- A4c(直接从表示生成初始化，不用 mask θ0)：53.95 / 32.35 / 52.15 / 42.31，明显更差。论文的解释是 mask 能约束优化空间，并且在 continual 场景下给新 cluster 提供 warm start。

**Continual adaptation**
- Toy 5-shot(Fig.4)：先只有 Sin + Line，15000 轮加入 Quadratic，30000 轮加入 Cubic。MSE：HSML-D 0.869±0.072，HSML-S(10C) 0.889±0.071，HSML-S(2C) 0.933±0.074。三者 CI 有重叠。
- 分类 5-way 1-shot(Fig.7 / 附录 F 表 7)：先只有 Bird + Texture，约 15000 轮加入 Aircraft，约 25000 轮加入 Fungi。平均：HSML-Dynamic 47.96%，HSML-Static(10C) 46.63%，HSML-Static(2C) 46.56%，MUMOMAML 44.11%。提升主要来自 Aircraft(54.50 对 52.30 / 51.28)。
- HSML-D 最后长到几个 cluster：论文未说明。官方 bash 中测试命令用的是 `--cluster_layer_0=4`(从 2 起步)，可推测最终是 4，但论文本身没有写。
- [代码] continual 分类实验里，数据生成器在每次扩展时都按 `sel_num = self.clusters` 重建，而新数据集何时出现取决于 sel_num 和样本序号(`data_generator.py` L229–242)。也就是说**数据流的演化与 cluster 数在代码中是耦合的**，论文里没有提到。toy 版本则是按迭代数(15000/30000)切换的固定日程。

---

## 逐题回答

**a. 层次如何表示，如何调制初始化**
- 层数和每层 cluster 数都是预先设定的：实验中为 3 层 4→2→1，输入层 K^0 = 1 即任务表示本身(§6.1, §6.2)。
- 每层是 **soft assignment**：对可学习 center 的距离做 softmax(Eq.6)，再按权重把各 cluster 的 tanh(W h + b) 加起来，得到下一层表示(Eq.7)。
- 任务表示 g_i 来自 RAA(默认，LSTM/GRU autoencoder，样本表示取平均，Eq.4–5)或 PAA(Eq.2–3)，同时用重构损失(ξ = 0.01)和 meta-loss 训练。
- 调制方式：o_i = sigmoid(W_g[g_i ⊕ h^L_i])，θ0i = θ0 ∘ o_i(Eq.8)。实现上只有第一层有显式 center，上层用可学习 gate 向量(见方法中"代码"一条)。

**b. 结构是固定还是学出来的**
- 静态版中层数和每层 cluster 数 {K^1..K^L} 都是事先给定的超参(Alg.1 的 Require)，只有 center、W、b、σ 相关参数和 W_g 是学出来的(代码里 σ 是常数 10 或 2)。
- 定义结构的超参：层数 L、每层 K^l(经验规则为上层取下层的一半)、σ^l、表示维度 d(40 或 128)。
- Continual 版只让最底层 K^1 增长，增长由 μ、Q 控制，代码里还有冷却期 8000。

**c. 多数据集如何平衡**
- 论文只写"Each task samples classes from one of the four datasets"(§6.2)和"randomly sampled from one of the four underlying functions"(§6.1)，比例论文未说明。
- [代码] 每个任务按 `np.random.randint(4)` 均匀选数据集或函数族，也就是**按数据集等概率采样**，与数据集大小无关。

**d. 能否在训练后增量加入新任务或新 family**
- 论文有 continual adaptation 实验(§4.3, §6.1, §6.2)，但扩展发生在**持续的 meta-training 过程中**，不是训练完成后的局部插入。
- 扩展方式：每 Q 个任务比较平均 loss(或准确率)，超过阈值 μ 就在最底层加 1 个随机初始化的 cluster(center 与对应的 W、b)，旧参数保留，然后全部参数(θ0、aggregator、所有 center、W_g)**一起继续训练，什么都不冻结**。代码里的做法是整图重建再拷回旧权重。
- 结果：分类平均 47.96 对 46.63 / 46.56，toy MSE 0.869 对 0.889 / 0.933，后者 CI 有重叠。
- 论文没有讨论的薄弱处：
  1. W_g 的输入维度虽不随 cluster 数变化(h^L 维度固定)，但加入新 cluster 后，所有任务的 softmax 归一化都会改变，旧任务的 θ0i 会漂移，没有任何保护旧 cluster 的机制，论文也没测旧 family 的遗忘；
  2. 触发依据是全局平均 loss，只能判断"要不要加"，判断不了"加在哪里"，新 cluster 也没有用新任务来初始化；
  3. 只能加叶子层，上层 K 是固定的；
  4. 不支持删除或合并 cluster。
- 论文在 §7 也承认这只是"simple version for continual learning"，并把 relationship forgetting 等列为未来工作。

**e. Held-out 评测**
- 主结果(表 1、表 2)是**已见 family 中的未见任务**：toy 用新的函数参数，分类用每个数据集的 meta-test 类。
- 只有附录 E 的 leave-one-dataset-out(表 6)测了**未见 family**，平均 34.41% 对 MUMOMAML 33.28%、MAML 32.43%，绝对值下降很多。

**f. 是否分析了学到的结构**
- 有。方法是选少量任务画第一层 soft-assignment 热力图，再在每层取 top-activated 节点画出 activation path(Fig.3、Fig.5，附录 I 的 Fig.9 有 16 个 toy 任务，附录 J 的 Fig.10 有 12 个分类任务)，另外对 gated θ0i 做 t-SNE 并与 MUMOMAML 对比(Fig.6，每个数据集 1000 个任务)。
- 观察：
  - toy 按**曲线形状**聚类，而不是按函数族(例如 Cubic 2 落到 sin 所在 cluster，Quadratic 2 落到正斜率线所在 cluster)；
  - 分类中 bird→C2、texture→C4、aircraft→C1、fungi→C3；第二大激活揭示跨数据集关联，作者解读为 shape(bird–aircraft)、environment(fungi–bird)、texture(texture–fungi)，第二层左节点对应"environment"，右节点对应"surface texture"。
- 这些属于定性解读，**没有任何定量的聚类一致性指标**(如 NMI 或 purity)。

**g. 代码**
- https://github.com/huaxiuyao/HSML ，continual 版为 https://github.com/huaxiuyao/HSML_Dynamic 。数据集链接在 README 中。论文正文没有给代码 URL。

---

## 对我们的借鉴

我们的设定是：组件库由代码组件构成，任务成员关系是连续的，outer loop 把 per-task inner-loop 进化中反复出现的 edit 提升为共享组件；约 17 个 benchmark、约 100 个 subtask；要求结构能在线扩展。

**可以搬过来的机制**
1. **Soft、连续的 membership**：HSML 用 softmax 权重，不用 hard 分配，理由也是"任务共享是连续的"(§4.2，引 Kumar & Daumé 2012)。这和我们"组件对任务有连续 membership"的想法一致。在代码库里，membership 可以是组件对每个 subtask 的边际收益(ablate 前后的分数差)，或是该组件被 per-task 进化保留下来的频率，不需要可微。
2. **"共享底座 + cluster 级 mask"的调制形式**：θ0i = θ0 ∘ o_i，即"全局一份，按 cluster 打开或关闭"。代码版本是：全局组件库，每个 cluster(或任务)用一个 on/off/参数化的配置来选择启用哪些组件。消融 A4c 表明直接生成初始化比 mask 共享底座明显差(1-shot 平均差约 4 个点)，论文的解释是 mask 能约束搜索空间，并给新 cluster 提供 warm start。这条原则对我们同样适用：**新任务从"全局库 + 最近 cluster 的开关配置"起步，而不是从零开始写 harness**。
3. **任务表示 = 由任务数据导出的特征**：HSML 用 (x,y) 联合嵌入，并且消融 A2a 表明 embedding 必须对下游目标敏感，单纯的重构预训练不够。对我们来说，任务表示不应只取 benchmark 描述文本的 embedding，而应包含"哪些 edit 对它有效"这类行为签名(例如 inner-loop 被接受的 edit 集合，或在组件上的收益向量)。只看外观相似度不可靠，HSML 的 toy 实验里任务按曲线形状聚到一起，而不是按函数族。
4. **按 family 等概率采样**(代码中是 randint)：我们的 17 个 benchmark 规模差异大，outer loop 统计"recurring edit"时应先在 benchmark 内归一化再跨 benchmark 汇总，否则大 benchmark 会主导哪些组件被提升。HSML 没讨论这个问题，只是默认按数据集均匀采样。
5. **分析方式**：membership 热力图加 activation path 的可视化可以直接借来展示"哪些 subtask 共享哪些组件"。不过要补上定量指标，HSML 自己就缺这一项。

**搬不过来的机制**
1. **端到端可微的聚类**：center、W、b、W_g 都靠 meta-gradient 联合训练，这是 HSML 的核心。代码组件之间没有梯度，聚类和组件提升只能用离散的统计或搜索，例如 edit 共现、收益矩阵的双聚类、LLM 归纳。
2. **固定拓扑**：层数和 K^l 都是预设超参，消融里加层也没有帮助(表 3)，层次结构相对 flat 的增益多数在 CI 内(A3b 对 HSML)。我们不该先定死层数或 cluster 数；层次应当从组件被多少任务共享中**自然出现**，例如共享到几乎所有任务的就是全局层、只被一个 benchmark 共享的就是 benchmark 层。
3. **全局 mask 向量**：在神经网络里是逐参数的 sigmoid，在代码库里只能对应组件级开关，粒度粗得多。

**HSML 对在线扩展的支持与阻碍**
- 支持的地方有两点：一是有"loss 恶化 → 加 cluster"的触发器；二是新 cluster 依附在共享 θ0 上，mask 机制给了 warm start(A4c 的论点)。
- 阻碍的地方：
  1. **全局重训**：加 cluster 等于重建整图并联合训练全部参数，没有局部更新，这正是我们要避免的"recompute everything"；
  2. **全局 softmax 归一化**：一个新 center 会改变所有旧任务的分配，从而改变它们的初始化。在我们的设定里，对应风险是新 benchmark 进来后重新聚类或提升组件，会悄悄改变旧任务实际用到的组件组合。因此需要**旧任务回归集**和"新组件默认只对新 membership 生效"的局部规则；
  3. **触发依据是全局平均 loss**：只能判断"要不要扩"，无法判断"在哪扩"，而且论文中的 μ(1.25 / 0.85)、Q 以及代码中隐藏的冷却期 8000 都是按场景手调的；
  4. **只扩叶子、不删不并**，并且 continual 分类代码中数据日程与 cluster 数耦合，实验本身并不等价于"未知时刻到来的新 family"；
  5. **评测**：continual 实验的提升有限，toy 的 CI 重叠，也没有报告旧 family 的遗忘。
- 落到我们的设计上：
  - 扩展判据改为**局部判据**，比如新任务的 inner-loop edit 与现有任何组件的相似度都低于阈值，或现有组件在它上面的收益都低于阈值，满足时才新建组件或 cluster；
  - 新组件只以新任务的 edit 作为种子，不重算旧 membership；
  - 只有当某个 edit 在 ≥k 个任务上复现时才向上提升到更高层；
  - 每次扩展后，对旧任务抽样回归，用来替代 HSML 缺失的遗忘评测。
