# meta/journal.md — 全过程流水账（append-only）

规则：每一步都记（时间、做了什么、看到什么、为什么）。结果表只进 experiments.md；这里是过程。

## 2026-10-10

- 19:1x 用 Reading 2 四个 probe 审论文，产出计划（见会话）。结论：Probe 4 为主轴——harness 能外置"代码算得出的状态"，固定小模型只会"查表"程度地用它；SP 里状态要靠模型判断，harness 质量坍缩为判官质量。
- 19:40 Engaging 解封（ControlMaster pid 99313 复活）。login008。无作业。login 节点上没有残留下载进程。
  - 405B 三个下载（bf16→/orcd/scratch/orcd/012/ziningl/models/Llama-3.1-405B-Instruct，FP8/INT4→/orcd/pool/007/ziningl/models/...）日志停在 10-09 12:40–13:35，"Fetching N files" 早期就卡住；进程已不在。未确认已落盘字节数（待查）。
  - Qwen2.5-7B-Instruct 与 Qwen2.5-72B-Instruct-AWQ 在 POOL hf_cache（72B 快照 698703ea，已在 2×L40S 服务成功过）。
  - GPU 分区：mit_normal_gpu 6h 上限：53×l40s:4、13×h200:8、1×h100:4；mit_preemptable 2 天：a100/h100/h200 若干。
- 19:5x 用户：「解封了直接开始干」→ 开始 Phase A（纯 CPU）：harness 接口 + 原始轨迹日志 + 50/250 切分 + 评分 + 种子 harness。
  - 已建 candidates/cd/{_common.py, bare, enum15, symbolic, facts, splits, oracle_hint, t3_gate}/harness.py。接口：init_state(view) / build_feedback(view, state) / final_hint(view, state)；view 里只有 H、alive、history、game（无 truth）。
  - 接口与旧 runner 的两处差别（有意）：① 第 18 个答案随最后的猜测提示一起发给模型（旧 runner 从不告诉模型第 18 个答案）；② 每轮记录发出的文本与模型原文。
- 19:5x 用户：「记录所有中间过程 记得把meta-harness代码拿下来再开始干」→ 本文件开始；先核 Meta-Harness 代码来源，再继续。
- 19:45 405B 三个变体重新用 sbatch 在 mit_data_transfer(dtn001)续传（jobs 25534538 fp8 / 25534542 int4 / 25534545 bf16，各 4 workers）。此前只落了 2.7–5.1 GB。pool 剩 962G（1T 配额），scratch 248T。
- 19:5x 官方代码 stanford-iris-lab/meta-harness（10-02 发布，1647★）clone 到 meta_harness_official/。读完 ONBOARDING.md、text_classification/meta_harness.py、两份 SKILL.md、claude_wrapper.py。要点：proposer = 每轮一个全新 `claude -p --model opus --effort max` 会话（允许 Read/Glob/Grep/Agent/Write/Edit/Bash），技能提示词注入；proposer 不跑评测；每轮必须出 3 个候选（TB2 是 1 个）；反调参、反过拟合规则；强制先在轨迹上原型；pending_eval.json 合同；evolution_summary.jsonl + frontier_val.json + reports/ 是 proposer 能看的历史；held-out 只在冻结后一次性评。
- 19:5x 本机 `claude -p "..." --model opus` 走订阅验证可用（1.8s；opus 别名解析到 claude-opus-5-5）。
- 19:48 Phase A 代码：runner/cd_harness_run.py（harness 可插拔、每轮记 sent+model_text、第 18 个答案随最终提示发出、best_query 记录以算 oracle 一致率）、runner/cd_mock.py（sequential/random/greedy/follow/flaky 五种脚本策略）、runner/cd_make_split.py（seed 20261010，search 50=17/17/16 按库存分层，heldout 250；search 实例副本写到 meta/cd_evolve/data/cd_search.jsonl 供 proposer 原型用）、runner/cd_score.py（uniq@end/exact 的 bootstrap 95% CI，--vs 配对差）。候选按官方布局放 meta/cd_evolve/agents/<name>.py，公共函数 meta/cd_evolve/harness_lib.py。
- 19:51 CPU 自测矩阵（7 harness × 4 脚本策略 × search 50，6s）：greedy 策略下所有 harness uniq@end 100%/eff 1.000（harness 不影响 oracle，合理）；sequential 20%/eff 0.535（合成参考 19.3%/0.540，吻合）；follow 只在 oracle_hint 下起飞（100%，oracle 一致率 94.5%，首轮无提示）；flaky 非法轮 22.2%。结果在 results/cd_mock/。
- 19:51 老 CD 作业计时：600 局 1484s（1×L40S，并发 32）→ 2.5 s/局。
- 19:52 代码同步到 Engaging（runner/stage0/splits/meta/cd_evolve/engaging/mt_puzzles），远端 mock 冒烟通过；提交 cd_ladder（25535085：7 harness × 300 × 2 seeds，一个 L40S，预计 ~3h）与 serve7b（25535086：给进化循环用的 7B 服务，6h）。两者排队中（Priority）。
- 19:56 写 meta/cd_evolve/.claude/skills/meta-harness-cd/SKILL.md（proposer 先验：任务/接口/两种 regime/指标/反调参/反过拟合/四步流程/pending_eval 合同）、domain_spec.md（按官方 ONBOARDING 要求，标明 confirmed/default/unknown）、evolve.py（外循环：propose→validate(import+mock+regime 静态检查)→benchmark(远端 srun --overlap)→score→frontier/summary；--test 冻结后一次性 held-out）。
- 19:58 cd_ladder（25535085）与 serve7b（25535086）同时在 node3102 起来；vLLM 170s 就绪；bare 开跑 3.9 s/局（并发 48，比 10-07 的 2.5 s/局慢——上下文长了，第 18 答案+原文日志不影响服务端，可能是节点差异）；endpoint node3102:8448。
- 20:00 proposer 管道验证：claude_wrapper.run 走订阅、技能经 --append-system-prompt 注入、会话落盘 logs/wrapper_test/<ts>_plumbing/{meta.json,events.jsonl,response.md}，一次空跑 5s、cache 写入 21.6k token。CLI 2.1.295 支持 wrapper 用到的全部旗标（--effort 取值 low/medium/high/xhigh/max）。
- 20:00 evolve.py 纯 CPU 干跑（--stub-proposer --local-mock follow）：baselines→3 个 stub 候选→validate→mock benchmark→frontier（stub1_oracle_hint 100%）→--test 冻结后 held-out×3 seeds（750 局配对 bootstrap）→再跑被拒。全链路通。
- 20:03 下载：bf16 在 scratch 已 676G/812G；FP8/INT4 在 pool 卡在 4.3G/5.1G 不动（疑似 10-09 被杀进程留下的 .lock）。排查中。
- 20:02 srun --overlap 进 serve7b 作业验证通过（node3102，/health OK）。
- 20:02 **启动真实第一轮**：`evolve.py --run-name open1 --regime open --iterations 1 --baselines bare,enum15`（nohup，日志 meta/cd_evolve/logs/open1/{evolve.log,nohup.out}；proposer 会话落 logs/open1/claude_sessions/）。先跑 bare/enum15 在 search 50×2 seeds 上的基线，再由全新 claude -p 会话提 3 个候选。目的：按 ONBOARDING 的建议先量一轮的质量与成本再定预算。
- 20:04 FP8/INT4 下载从 pool 改到 scratch 重提（25535610 fp8 / 25535611 int4）；pool 上 4–5 GB 残片删除；bf16 已 879G、"Fetching 584 files 22%"（写入字节数超过权重总量，可能含预分配，等完成再核）。
- 20:05 文档：experiments.md 追加 10-09（判官校准表、CD H100 重算表、T3 代理）与 10-10（切分/自测/作业/首轮）两节；RELATED_WORK.md §5 新增 5.1（方法模板：Meta-Harness/DGM/HyperAgents/SkillOpt/ADAS/ALMA + 程序/图/OS 抽象对照）与 5.2（Snell/MAST/MoreAgents/开放式/2609.01615）。
- 20:06 为网格实验排了 Qwen2.5-32B-Instruct-AWQ 下载（dlhf 25535665，进 POOL hf_cache；72B-AWQ 已在）。写 runner/cd_ladder_plot.py（uniq@end 带 CI + eff 带 random-fresh/sequential 参考线），用 mock 分数试过出图。
- 20:06 校正：上面 19:48–20:06 的时间戳按 Engaging 作业日志重排过（我之前估快了约 15 分钟）。
- 20:16 open1 基线完成：bare uniq@end 32%（search 50×2）、enum15 32%/exact 7%；proposer 会话 20:16 起，20:28–20:29 写出 3 个候选（directive：把摘要换成一条可直接照抄的动作；escalate：directive + 偏离反馈；openlist：enum15 的无排序条目列表）。
- 20:30 **协议纠错**：核对 MTP 论文附录的 CD 对话样例，官方环境每轮回 `A(0, 0, 0) = 1` + `Please send your next query. 17 queries remaining.`，第 18 个答案和最终提示一起发。我们之前用的 "The output of that query is 0. Please make your next query." 是我自己编的。改 runner 为官方措辞（含剩余次数倒计时），mock 解析同步改，CPU 自测矩阵重跑通过（oracle 100%/1.000，sequential 20%/0.535）。官方未发布环境代码，模型不出查询时怎么处理仍未知（我们：重问一次、计入 18 次）。
- 20:31 为保证口径一致：杀掉 open1（proposer 候选文件移到 results/oldenv/agents_open1/ 留档，日志移到 results/oldenv/open1_logs/），取消 cd_ladder 25535085（远端结果改名 cd_ladder_oldenv_2029）。serve7b 不动。新代码 rsync 到 Engaging、远端冒烟确认新话术，重提 cd_ladder 25537086，重启 open1（20:33）。
- 20:34 教训：两个并行 Bash 调用共用一个 shell 的 cwd，一个 cd 把另一个的相对路径带偏了——journal 一次追加落空、experiments.md 写到了错误目录（已纠正）。以后写文件一律绝对路径。
- 20:33 重提的 cd_ladder 25537086 落到 node3006，vLLM 在 init_device（torch.accelerator.set_device_index）就挂，81 秒失败；判断是该节点 GPU 状态问题（同一脚本在 node3102 正常）。日志留作 vllm_25537086_failed_node3006.log。排除 node3006 重提为 25537286。
- 20:45 三个 405B 下载 20:07 同时死于 scratch "Disk quota exceeded"（per-user 约 1T；df 看不出来，`quota`/`lfs quota` 都没有 scratch 的条目）。bf16 目录 939G 里有 181G 是 `original/`（PyTorch 原版 checkpoint，vLLM 不用），hf download 默认也拉。处理：删 original/ 与 FP8 残片（scratch 1017G→799G）；bf16 残片用 rsync --remove-source-files 搬去 pool（mvbf16 25537655，边搬边释放）；INT4 优先续传（25537656，dl_405b.sbatch 已加 --exclude "original/*"）；FP8 等空间腾出再提。
- 20:45 **用户定：SP/DC 判官 = Llama-3.1-405B-Instruct-AWQ-INT4**（论文判官的 4-bit 版）。写 engaging/judge405.sbatch（2×H200 TP=2，起来先跑 TurtleBench 150 校准，再发布 serve_judge.endpoint 持续服务）；Engaging 上 clone 了 mazzzystar/TurtleBench。runner/sp_runner_async.py 加独立判官端点（答题者+要点判官走 405B，提问器走 7B）。
- 20:45 用户给出定位：**模型只是提问器，研究对象是 harness 自进化能否解决主动推理**。据此提出：撤掉"查询必须模型写"等约束（只留"读不到真值"+"环境固定"）；CD 主实验改为从零（不给 harness_lib、接口不暴露假设空间）看 proposer 能否自己发明信念跟踪；AR-Bench 三任务为主结果、对手是其五个手写方法；记录机制发现轨迹。CD 从零方案待用户点头。
- 20:41 **官方话术下的基线（search 50×2 seeds，7B）**：bare uniq@end 26% [18,35]、exact 0、eff 0.558、rep 1.8%、ill 0.3%；enum15 uniq@end **53%** [43,63]、exact 5%、eff 0.629、rep 3.5%、ill 0.2%、parsed 97%。同一切分旧话术：bare 29%、enum15 32%（rep 8.1%、ill 1.7%）。**结论要改**：此前"外置信念让 uniq@end 反降、重复和非法上升"相当一部分是我自拟环境话术造成的（官方每轮复述查询并倒计时）；在官方协议下手写 harness 把 uniq@end 翻倍，但 eff 0.629 仍只是 random-fresh 水平，离 oracle 很远。阶梯作业（300×2）会给出完整读数。
- 20:50 用户问通用 harness 进化框架。核实（gh api）：RRSI google-research/rrsi 1437★（arXiv 2609.24972，Google Cloud AI Research，Yoonho Lee 合著，PDF+txt 已入 papers/）、OpenEvolve 7518★、ShinkaEvolve 1440★、DSPy 38.6k★、GEPA 6.9k★、TextGrad 3.8k★、microsoft/Trace 763★、SuperagenticAI/metaharness 173★、harness-forge 80★、WHALE 59★。arXiv export API 这会儿连不上（SSL/代理），GEPA/Trace/Shinka 的 arXiv 号未核，只引仓库名。
- 20:52 按用户要求 clone RRSI 到 rrsi_official/（HEAD be50316，4 commits，Apache-2.0，核心约 3.1k 行）。已读 domain.py（Domain 抽象：evolve_ids/heldout_ids/smoke_ids/run/score/guards/critic_patterns/briefs）、selection.py（噪声底线 S* − δ、成本规则 ΔC ≤ β0+β1ΔS、小增益时 w_s·ΔS − w_c·ΔC + w_n·ν > 0、guards、argmax）、critic.py（正则黑名单 + LLM 审查，六类拒绝：泄漏/退化/判官作弊/未声明捆绑/运行时记忆泄漏/无界循环）。
- 20:55 RRSI 细读：rrsi.json（TB2 实例）δ=0.017、β0=0.1、β1=44.5、w_s=0、w_c=15、w_n=0.5、T=20 轮、k=2 trials、每轮 m=2 候选、编辑预算 b_t 余弦退火 1–4、停滞窗 w=3、修复轮 5、缺失>20% 判无效；calibrate.py 用 bootstrap（或重复基线评测）估 δ=z·sd_null（z=2）；llm.py 直接走 AnthropicVertex（不是 LiteLLM），critic/proposer/analyst 都要 API——移植到我们这里 critic 的 LLM 审查可包成无工具的 `claude -p`。
- 20:55 open1 第一轮 proposer（20:41 起）已写出 copy_query.py、open_entries.py，第三个未出，pending_eval.json 未写。
- 21:00 读完 RRSI 论文方法/结果两节（txt 含 NUL 字节，grep 要加 -a），条目写入 RELATED_WORK §5.1。
- 21:03 **open1 第一轮结果（official 话术，search 50×2）**：proposer 会话 866s、订阅计价 $3.90、输出 81.8k token、cache 读 12.9M；候选 copy_query（只给一条可抄的最优查询）uniq@end 比 bare +74 [+65,+82]（≈100%）、exact +81（≈81%）；open_entries（"还能问这些"正面名单）uniq +6 [−3,+15]（噪声内）、exact +47 [+38,+57]（终猜前贴候选表）。第三个候选评测中/未出。决策级提示让 7B 当抄写员就能到顶，印证 open regime 的平凡吸引子；状态级改写对信息收集无效、对终答有效。
- 21:05 阶梯：bare 600 局 1115s（1.86 s/局）；enum15 跑到 125/600 时 err=7（待查：可能上下文超限或超时）。INT4 205G/219G 接近完成；judge405 作业已挂依赖（afterok:25537656）自动起。
- 21:05 用户：先考虑单任务（多任务骨架以后再说）。单任务升级顺序：runner 钩子（打回重问、重写上下文；查询仍须模型写）→ evolve.py 接 RRSI 选择规则与 critic → search 扩到 100 题。
- 21:07 open1 第一轮结束（1565s）：第三候选 copy_open 也评完；frontier = copy_query @ uniq@end 100%。
- 21:08 阶梯 enum15 的 400 错误 = 上下文超限（max-model-len 12288，prompt 已 11089 token + 1200 输出）；7B 很啰嗦（该局输出 6081 token），enum15 文本逐轮累加。serve_7b/cd_ladder/cd_grid 三个 sbatch 的 max-model-len 改为 16384（只影响以后的作业；当前阶梯与 serve7b 仍 12288）。错误局在 cd_score 里按 err% 计、不进指标——偏向删掉最长的局，要写明。
- 21:09 INT4 下载 20:45 已完成（47 shards，205G）；judge405 提交为 25539120（2×H200，排队中）。
- 21:11 runner 新钩子通过 mock 验证：filter_action（打回重问，每轮 ≤2 次，不耗环境预算，计 n_reasks/n_calls）、compose_messages（重写上下文；不变量：system 不动、每条观测 "X(a, b, c) = b" 必须仍在、末尾是 user 轮；违反即 HarnessViolation 记错不重试）。mock 的历史重建改为直接从观测行解析（压缩后的账本也能读）。为不在运行中的阶梯作业脚下换代码，新 runner 暂不同步到 Engaging，等阶梯结束。
- 21:11 search 切分 v2：splits/cd_split_v2.json（100 题 = 34/33/33，含 v1 的 50；held-out 200），工作区副本 data/cd_search_v2.jsonl。runner/cd_score.py 加 --split-file。
- 21:12 evolve.py 接入 RRSI 选择侧：noise_delta()（bare 基线 bootstrap，δ=2·√2·se，缓存 noise.json）；admissible()：S′ ≥ S* − δ，增益 > δ 时 ΔC ≤ β0+β1·ΔS（0.1 + 2.0·ΔS，C=每局 in+out token），带内只有省 ≥5% token 才收；不可接纳的候选记入 frontier_val.json 但不成为 incumbent。critic：正则预检（truth/实例文件/文件与网络 IO/动态执行/import 白名单）+ LLM 审查（claude -p，RRSI 六类改写成我们的五类，含 regime 违规），拒绝原因写入 evolution_summary。SKILL.md 补：钩子说明、"查询必须模型写"、选择规则、critic、16k 上下文、v2 数据文件。
- 21:12 干跑验证：v2 切分下 δ=0.078（n=200）；admissible() 行为正确（带内只收省 token、超带按成本预算）。critic 正则 `\btruth\b` 误伤（harness 文本里"joint truth tables"是任务词汇），改成只拦代码级访问（truth_bits / ["truth"] / inst[...] / 实例文件字段 / IO / 动态执行 / import 白名单）。LLM critic 实测：copy_query 在 open 下 accept（16s），在 state_only 下 reject 且理由准确（best_query 推荐 + 分裂计数）。
- 21:13 启动 open1 冻结终评（`evolve.py --run-name open1 --test`）：bare/enum15/copy_query/copy_open/open_entries 在 v1 held-out 250 × 3 seeds 上跑（约 3750 局，走 serve7b），结果到 results/cd_heldout/open1/。
- 21:16 判官作业卡在 QOSMaxGRESPerUser：mit_normal_gpu 每用户 2 张卡（阶梯 + serve7b 已占满）；mit_preemptable 每用户 4 张、上限 2 天、可被抢占。判官改投 mit_preemptable（24h）：校准只要几分钟，之后的 SP 运行本身支持 JSONL 断点续跑，被抢占可接受。
- 21:21 judge405（25539226，mit_preemptable node5100）vLLM 在 CUDA graph 捕获阶段挂：TP=2 在 H200 上触发 FlashInfer trtllm_mnnvl all-reduce 的 JIT，nvcc 13.4 与其 CCCL 头不兼容（和老脚本里 sampler 的坑同源；L40S 上 72B TP=2 不走这条路所以没撞）。权重 47 shards 加载正常。加 --disable-custom-all-reduce（回落 NCCL）重提。
- 21:22 judge405 重提为 25539697（mit_preemptable，--disable-custom-all-reduce），已在 node5100 运行。第二个 7B 服务投 mit_preemptable（25539716，endpoint 文件 serve_7b_pre.endpoint，文件内含 jobid），给进化循环用；带钩子的新 runner 以 runner/cd_harness_run_v2.py 的名字放到 Engaging（阶梯作业用的旧文件不动）。evolve.py 加 --endpoint-file / --remote-runner。
- 21:23 启动 **so1**（regime state_only，v2 切分 100×2，critic llm，基线 bare/enum15，先跑 1 轮）：等 serve_7b_pre 起来后自动开始。
- 21:24 校正：21:08–21:23 的时间戳按循环日志重排过（我的估计又快了约 20 分钟；以后直接取 `date`）。
- 21:27 judge405 第二次（25539697）同样死在 FlashInfer trtllm_mnnvl JIT，--disable-custom-all-reduce 没绕开。改用 --pipeline-parallel-size 2（PP 只用点对点通信，不需要 all-reduce 核）+ --dtype float16，重提。备选：升级/卸载 conda_env 里的 flashinfer-python。
- 21:29 judge405 PP=2（25539762，node5100:9194）起来：加载 160s，校准首跑 141/150 无法解析——原因是 checkpoint 的 generation_config 只列 <|end_of_text|>，模型输出 "Incorrectassistant\n\n## Step 1…"，而解析器要求标签后有词边界。改：解析只要前导边界（\b(Incorrect|Correct|Unknown)），Llama 判官请求加 stop_token_ids；SP runner 同步。重跑：**82.0%**（0 条无法解析），与官方 405B 82.7% 持平。结果 ~/results/ar_harness/judge/turtle150_405b_int4.json。
- 21:36 SP bare（7B 提问 + 405B-INT4 答题/判官）200 局 7 分钟跑完：f1_char 0.621 / f1_word 0.244；答题分布 no/yes/unknown = 48/29/23%。结果 results/sp405/。ssh 里 `nohup srun … &` 会把 ssh 会话挂住直到 srun 结束（stdin 没重定向），下次用 `setsid … < /dev/null`。
- 22:55 **so1 第一轮**：veto_only 97.5%（exact 83.5%），veto_determined 93.5%，answer_grid 86.5%（低于底线被拒）；critic 全过；proposer 22 分钟 $6.69。机制：filter_action 只在模型问已确定/重复/无查询时打回，附"未定输出"清单；compose_messages 删掉已结案的打回对话；final_hint 把剩余配置写成 <guess> 串。state_only 规则下合法（集合成员，无信息量计算）。
- 22:57 so1 冻结并启动 held-out 终评（v2 held-out 200 × 3 seeds × 5 系统，走 serve_7b_pre）。open1 终评 4/5 系统完成。阶梯跑到 splits：12k 上下文下 61% 局超限（err=259/425），该 harness 要用 16k 重跑。
- 22:58 bf16 残片已全部搬到 pool（758G），mvbf16 作业以 exit 1 结束（rsync 尾部告警），导致 afterok 依赖的 FP8 下载被取消；scratch 只剩 INT4，重提 FP8 下载。
- 23:10 open1 终评完成（750 局/系统）：bare 22.5、enum15 35.1、copy_query 99.7（exact 80.4）、copy_open 99.7（68.4）、open_entries 34.5（42.3）。enum15 的 search 53% 是小样本偏高。
- 00:28 阶梯完成（7 harness × 600 局，近 4 小时）。splits 58.7%、oracle_hint 20.7%、enum15 3.2% 的局因 12k 上下文超限出错；这三行用 16k 重跑（cd_ladder16k，sbatch 加 LADDER_DIR/HARNESSES 覆盖）。cd_score 表加 err% 列并写明口径。结果 results/cd_ladder/，图 results/cd_ladder/ladder_12k.png。
- 01:0x so1 终评完成（600 局/系统）：veto_only 96.7% [95.2,98.0]、exact 86.2%、token≈bare；veto_determined 95.3%；answer_grid 90.0%；enum15 37.7%；bare 25.2%。两条 regime 的 held-out 都落地：decide 级 copy_query 99.7%/80.4，state 级 veto_only 96.7%/86.2。
- 03:20 阶梯 16k 重跑完成：enum15 41.8、splits 53.7（仍 10% 超限，输出 4.3k tok/局）、oracle_hint 96.5（exact 6.8）。合并图 results/cd_ladder/ladder.png。
- 03:22 FP8 下载已于 23:08 完成（454G，scratch）；三变体现状：INT4 205G + FP8 454G 在 scratch，bf16 残片 758G 在 pool（未完成，缺约 54G，需要时再续）。mit_normal_gpu 的 serve7b 到 6h 上限已结束；preemptable 上的 judge405 与 serve7b 还在（24h 上限，已用 6h）。
- 03:5x 用户问"从零发明信念跟踪"有无文献。arXiv export API 之前报 Error 是我 zsh 里 `set -- $pair` 不分词导致 id 为空，不是代理；改用 ${pair% *} 拆分后正常。下载并读了 9 篇（WorldCoder 2402.12275、CWM-GGP 2510.04542、PatchWorld 2605.30880、LATM 2305.17126、Hypothesis Search 2309.05660、Text2World 2502.13092、EvoHarness-RL 2608.05446、Belief Divergence 2607.04528、BB-WM 2609.00455、Bayesian-Agent 2606.08348），写入 RELATED_WORK §5.3/§5.4。
- 04:1x 用户要把 EvoHarness-RL 当 baseline 跑。官方代码 github.com/EvoHarnessRL/EvoHarness-RL（10-10 推送）clone 到 evoharness_rl_official/。计划：只跑 EvoHarness-Base（推理时 BPE 脚手架，不训练，和我们"权重固定"口径一致）；要给 CD 写一个 environment adapter（Belief/Progress/Experience 的落地）；EvoHarness-RL（SFT+GRPO 改权重）不在同一口径，只在有余力时作附加臂。
- 04:4x 为 EvoHarness-RL 写了 CircuitDecoding 适配器（evoharness_rl_official/evoharness/envs/circuitdecoding/：env.py 官方话术环境 + belief.py 把我们的可行集跟踪包成他们的 Belief（track [remaining]/track [A]，always_on 面板=enum15 内容）+ prompts.py 三种模式的提示词 + domain.py（accumulated 上下文，和 MTP 协议一致）+ configs/eval/circuitdecoding.yaml），注册进 envs/__init__.py；runner/cd_score_evoharness.py 回放他们的轨迹算 uniq@end/eff 等。CPU 脚本策略冒烟：env_only 顺序扫失败、always_on 抄面板表成功、inline track 一次后成功。
- 04:4x 部署到 Engaging（rsync 不含 training/，pip install -e 进 conda_env），在 serve_7b_pre（25539716）作业里 srun 跑 6 个 run：{env_only, always_on, inline} × 2 次，各 300 题，温度 0.7、1200 token、32 workers，meta-action 不计预算（上限 18）。输出 ~/results/ar_harness/evoharness_cd/<mode>_s<seed>/。
- 07:36 EvoHarness 六个 run 跑完（约 1h8m），结果同步到 results/evoharness_cd/。ReAct 26–29%；always_on 71–75%（exact 62–65）；inline 30–32%（track 0.08 次/局）。写入 experiments.md。
- 2026-10-12 15:0x 用户同意把所有实验提交到 GitHub。ar_harness 已 git init 并接上 Zarreboln/self-evolving-active-reasoning（公开，原来只有 README 骨架；合并骨架历史）。.gitignore：四个第三方 clone + mastermind、papers/、原始轨迹 jsonl、EvoHarness trajectories、mock/dry 输出、MTP 非 CD 任务数据。提交：代码+文档一笔、experiments 索引一笔、七个实验各一笔（01 stage0 / 02 判官 / 03 阶梯 / 04 open1 / 05 so1 / 06 EvoHarness / 07 SP）、README 重写一笔；已 push。原始轨迹打成 6 个 tar.gz 挂到 release v0.1-experiments-2026-10-11。
- 同时发现 Engaging 的 ControlMaster 掉了（Permission denied (keyboard-interactive)），需要用户再过一次 Duo；judge 校准的逐题文件 turtle150_405b_int4.json 还在集群上，本地先写了汇总 json。
- 2026-10-12 15:3x 用户问"proposer 读了什么种子的轨迹"。从会话日志核：open1 的 proposer 读了 search 切分上 bare 与 enum15 的全部轨迹（--list 清单、enum15 的第 1/140/275 局逐轮、对 enum15 轨迹文件做统计、错误局）和 evolution_summary/frontier；so1 的 proposer 除了 so1 的 bare/enum15 轨迹，还读了同一工作区里 open1 的 evolution_summary/frontier 和 open_entries 的轨迹、stage0/cd_oracle.py、data/cd_search_v2.jsonl。其余 5 个种子只有代码在 agents/，没有轨迹。
- **发现两个协议漏洞并修复**：① proposer 会话（claude -p，cwd=meta/cd_evolve）会读写 Claude Code 按目录的 auto-memory（~/.claude/projects/-…-meta-cd-evolve/memory/）：so1 的会话写了两条笔记（上下文预算/mock 盲区；"pending_eval 里把贵的候选排前面，因为成本规则按当时的 incumbent 算"），并被后续会话读到——这是协议之外的跨会话状态，而且第二条是在学怎么博弈选择规则。修复：evolve.py 在每个 proposer/critic 会话前把该目录清空（原件归档到 run 日志）、会话后把它写的归档并删除；so1 那两条已归档到 logs/so1/claude_sessions/iter1_memory_written_recovered/。② 选择规则实现与 RRSI 算法 2 不一致：我按评测顺序逐个更新 incumbent，同一轮的兄弟候选会互相定价。改为整轮对同一 incumbent H_t 与 S*（已接纳的最高分）判定，可接纳者取 uniq@end 最高为本轮赢家，只有赢家进 frontier；S* 不再包含未接纳候选。
- 2026-10-12 16:0x 用户：找"harness 不断进化、放进内循环"的文献。找到并下载 TTHE 2607.08124、JIT-Agent 2608.25593、RHO 2606.05922、Evo-Harness 2608.15071、AgentStream 2608.00155、Rethinking-eval 2607.12227（PDF+txt 在 papers/），另核实 EvoHarnessBench/Evo-Bench/HarnessX/AHE/Harness-updating/Learning-While-Acting/AdaMEM/Evo-Memory/EnvHarness 的 ID。写入 RELATED_WORK §5.5。要点：TTHE 的"无标签代理信号 + 测试中持续提交"正好对应 CD 里不碰真值就能算的 uniq@end/零信息率；Rethinking 要求同预算 test-time scaling 对照和 held-out。
- 2026-10-12 16:3x 用户选定 TTHE 路线（harness 放进内循环、测试时进化）。精读 TTHE 方法：流按批（B=10 最好）处理；每批 G=3 条分支 × R=3 轮，OBSERVE（跑所有分支得轨迹+代理信号）→ PROPOSE（G 个 coding-agent proposer 各改自己分支，角色：保守修复/探索/对抗）→ JUDGE（agentic 选择器，只看无标签证据，可再探测）→ 提交 Ht+1 管辖下一批；分数为 transductive（提交的 harness 在本批缓存输出上打分）；solver/proposer/judge 同一冻结模型（deepseek-v4-flash），Claude Code 只当编码界面把调用路由到该模型。代码 github.com/junnie00/TTHE（10-06 推送）已 clone 到 tthe_official/。
- 2026-10-12 17:0x 开始搭 meta/cd_tthe/（TTHE 式内循环，CD 版）：harness = 包住冻结 7B、自己跑整局的任意 Python 类（solve()），两条不变量（冻结求解器、无标签）；bridge.py 提供 CDEnv（真值私有，只经 env.query 给一位）、带磁盘缓存的 Solver（可换 mock）、流构造；agents/bare.py = 官方协议种子。TTHE 代码细节：按批 G 分支 × R 轮，proposer 三角色（保守修复/独立探索/对抗审计），judge 在"本批观测过的所有 harness（含 incumbent）"里选（回滚门），transductive 计分；solver 调用有缓存；候选在子进程里带超时跑；vLLM 有 Anthropic 兼容入口（entrypoints/anthropic），所以"7B 自己当 proposer"的 TTHE 原味臂可行。
- 2026-10-12 17:3x meta/cd_tthe/ 搭完并在 CPU 上跑通：signals.py（无标签代理：health/n_queries/uniq_end/alive_end/zero_info/repeat/oracle_agree/eff）、audit.py（冻结求解器+无标签+分臂 import 白名单；拦下读 _truth 的样例）、solve_worker.py（子进程跑一局，记轨迹/信号/金标仅存档）、proposer.py（三角色 proposer + judge；后端 claude/solver(路由到 vLLM Anthropic 入口)/stub；轨迹渲染成 md；check 脚本让 proposer 自验；记忆隔离）、optimize.py（按批 G 分支 × R 轮 → judge 在含 incumbent 的全池里选 → 提交；同时记 prequential 与 transductive）。stub+mock 干跑：2 批 × 2 分支 × 2 轮，judge 回滚门生效（stub 子代与父代同分则保留 incumbent）。
- 2026-10-12 15:35 **cd_tthe 管道测试（真 proposer，假求解器）**：scratch 臂，G=1 R=1，5 题；conservative-repair 角色的 Claude 会话 136s/$0.60，**从任务规则重新发明了假设空间枚举**（_realizable/_configurations：每电路恰 2 个二元门、NOT 任意、排除常数、跨电路分配），三种库存的空间大小与我们核过的 oracle 完全一致（252/2886/3924）；加了"已确定格子不花预算、给值后重问"的 veto、唯一时提前停、终猜前列已确定输出；它还试过"每轮都给已确定清单"并因 mock 上更差而撤回（写进 card 的 preserved_behaviors）。judge 18s/$0.09 提交了它；mock 上 uniq_end 5/5。它顺手发现 check 脚本里 json.dumps 生成的 `null` 字面量会 NameError（已改为 repr 生成）。真 7B 上的表现待 GPU。
- 2026-10-12 18:0x 用户定：proposer 只用 Claude，要看多轮内循环持续进化的轨迹。cd_tthe 加 --budget（每局查询数可配，12 次留余量）、trajectory.py（逐批：在岗 harness、是否换代、换代理由、prequential/transductive 分数 + 图）。等 Duo 后起 GPU：scratch 臂，B=10、G=3、R=2，200 题流；若首批就到顶，并行一条 budget=12 的流。
- 2026-10-12 18:2x 用户要求 proposer 起步"什么也看不到"：加 --signals strict（proposer/judge 只看原始轨迹 + 运行状态/是否产出终猜/用了几问；不给 uniq_end/alive_end/零信息/eff，不给"模型习惯"先验；judge 也只能读轨迹判断）。stub 干跑通过；启动 strict 管道测试（真 Claude proposer，假求解器，5 题，G=1 R=1），看信念跟踪在零提示下能否长出来。
- 2026-10-12 18:4x 用户去掉"full"臂（我们算的无标签信号不给 proposer 看）。现在两条臂：strict（只看轨迹+运行状态）与 labeled（strict + 每局最终结果：整表是否正确、三张对几张）；strict 的角色描述也去掉了失败类型例子。
- 2026-10-12 16:05 EDT Engaging 经用户 Duo 恢复（login010）。run_pilot.sh 提交 serve7b 25669627（mit_normal_gpu，6h，endpoint 文件 serve_7b_tthe.endpoint），等 endpoint 后开隧道 127.0.0.1:18000，再并行起 pilot_strict_b10g3r2 与 pilot_labeled_b10g3r2（各 30 题 = 3 批，G=3，R=2，proposer=Claude opus，effort high）。
- 2026-10-12 15:55 EDT **strict 管道测试结果（零提示，真 proposer，假求解器，1 批 5 题，G=1 R=1）**：proposer（保守修复，96s/$0.39）从轨迹里自己诊断出"模型按行扫 A、B，到 C 只剩 2 问，终猜里未观测格子全填 0"，提出的改法是提示层面的：每轮附各电路已问次数 + "预算覆盖不了 24 行、要用库存推断"的提示、终猜前按电路列出全部观测、终猜与观测矛盾则重问一次并把观测格子写回。它写明"可以用库存推断行"却没有实现枚举。judge（20s/$0.09）因假模型下两者输出完全相同而保留 bare（回滚门正确）。对比 full 信号下的那次（同角色一个会话就写出枚举器）：proposer 看到什么决定它发明什么。真正的检验要等 7B 上多轮。
- 2026-10-12 16:11 EDT 试点第一次起来全部崩：mac 的系统代理（HTTP_PROXY）劫持了 127.0.0.1:18000 的隧道，OpenAI 客户端拿到 502（和 memory 里 ollama 那次同一个坑）。bridge.Solver 改为 httpx.Client(trust_env=False)，直测隧道返回 OK；杀掉两条臂和 proposer 会话，清空 run 目录重启。

### 2026-10-12 16:30 EDT — official definitions pinned (user: "完全按照官方定义不要自己编")
- Wrote `experiments/OFFICIAL_DEFINITIONS.md`: MTP-CD task text (Table 1), the two official metric names
  ("Normalized circuit-wise accuracy", "Normalized % of trajectories with successful end state"; the
  normalization is not defined anywhere in the paper or dataset card), official baseline numbers
  (Fig. 1 success rate: Gemini-2.5-Pro-Exp 39.8 / GPT-4o 15.4 / Claude-3.5-sonnet 9.3; Fig. 5 Gemini-2.0
  Pro 32.2 / Flash 7.2 / Flash-lite 4.1), and the list of things the official release leaves open
  (invalid-reply handling, early stop, decoding params, normalization) — all of those are now labelled
  as our choices. AR-Bench section: DC accuracy / SP char-level F1 / GN exact match, judge 405B.
- `runner/cd_score.py` table now prints `success` (= our `exact`) and `ckt-acc` first; uniq@end, eff,
  zero%, orcl% stay as diagnostics. JSON keys unchanged so old score files still load.
- cd_tthe loop prints "prequential success=…" first, uniq_end in parentheses.
- Pilots (started 16:14/16:15, 7B on node3502 via tunnel 18000): batch 0 bare prequential success 0%
  in both arms (uniq_end 20% strict / 30% labeled). Round-0 candidates (3 per arm) passed the static
  check at 16:16–16:17 and are being solved on the 7B now (solver_cache growing).
- TTHE clone pinned at 69a614d in THIRD_PARTY.md; `tthe_official/` ignored.

### 2026-10-12 16:50 EDT — naming pass: official names only (user: "全部按照官方的叫法")
- Correction of my 16:30 sheet: MTP Table 1 lists ONE metric for Circuit Decoding, "Normalized
  circuit-wise accuracy". "Normalized % of trajectories with successful end state" is Word Chaining's
  metric; I had mis-attributed it. Figures 1/5 label every task's number "Success Rate (%)" without
  saying how it is derived for Circuit Decoding.
- New names everywhere (score JSON keys, printed tables, READMEs, experiments.md, evolve.py):
  `circuit_wise_accuracy` / "circuit-wise accuracy" (official, headline) and
  `joint_truth_table_correct` / "joint truth table correct" (whole 24-bit table right; the paper's noun,
  our reading of success). Old keys `exact` / `ckt` migrated in every committed score file.
  uniq@end / eff / zero% / orcl% / rep% / bit stay as our diagnostics, labelled "(ours)".
- Task name in prose: "Circuit Decoding" (paper), identifiers/paths unchanged.
- cd_evolve selection objective switched from uniq@end to circuit-wise accuracy (ties: joint truth
  table correct) for any future run; runs open1/so1 keep their recorded uniq@end-based selections.
- cd_tthe internal keys (bridge.score, solve_worker, proposer render, result.json) NOT renamed yet:
  the two pilots import those files in subprocesses; renaming mid-run would desynchronise the running
  optimize.py. Phase 2 after the pilots finish.

### 2026-10-12 16:56 EDT — strict pilot: first judge decision (batch 0, 2639 s)
- Run `pilot_strict_b10g3r2` (arm scratch = no cd_oracle library; signals strict = raw traces + run
  status only; proposer Claude Opus, judge Claude Opus, solver Qwen2.5-7B on node3502). Batch 0 =
  instances [156, 11, 250, 128, 98, 239, 105, 247, 164, 265].
- Pool on batch 0 (uniq_end / joint truth table correct): bare 0.20/0.00; r0g0 0.30/0.00; r0g1 0.40/0.00;
  **r0g2 1.00/1.00**; r1g0 0.70/0.00; r1g1 0.80/0.00; r1g2 1.00/1.00. The judge saw none of the
  gold numbers, only parsed%/queries/tokens/crashes + traces + source.
- JUDGE -> `cand_pilot_strict_b10g3r2_b0r0g2` (round-0 adversarial-audit candidate). Its reasons:
  enumerates the rule-allowed circuits from the system prompt alone; the model still chooses every
  query (menu of up to 8 informative queries, its own choice sent after bounded re-asks) and the final
  table (numbered candidates); traces decisive, 11.7 queries/episode. Rejected r1g2 (menu of 3, other
  choices refused until the last try = "close to the harness choosing the queries").
- Prequential on batch 1 (10 fresh instances, before any adaptation): joint truth table correct 10/10,
  circuit-wise accuracy 100%, mean 11.1 queries. Trace audit (my script): every sent query appears in
  the model's reply in all 10 episodes; the model wrote the final <guess> itself in 9/10 (in i228 the
  harness assembled the guess from the model's numbered picks).
- Answer to the user's question "proposer 最开始什么也没有，看他能不能长出来": yes — zero hints, the
  enumerator + split menu grew in round 0 and was selected label-free.
- Diagnostic bug noted for phase 2: pool `tokens_out` shows 0 for candidates because the proposer's own
  check runs filled the solver cache for the same (messages, temperature, seq) keys, and cache hits carry
  no usage. Store usage in the cache entries. Does not affect any result.
- Labeled arm: still batch 0, round 1 proposers running.

### 2026-10-12 17:03 EDT — labeled pilot: first judge decision (batch 0, 2987 s); both arms 100% on batch 1
- `pilot_labeled_b10g3r2` (strict signals + per-episode outcome shown to proposer/judge). Pool on batch 0
  (uniq_end / joint truth table correct): bare 0.30/0.00; r0g0 0.60/0.10; r0g1 0.40/0.00; r0g2 1.00/1.00;
  r1g0 1.00/1.00; r1g1 0.00/0.00 (wall timeout 900 s in 9/10 — a formula checker too slow); r1g2 1.00/1.00.
- JUDGE -> `cand_pilot_labeled_b10g3r2_b0r1g2` (round-1 adversarial audit, child of r0g2): same
  rule-derived enumerator, suggestions ranked by binary entropy over the JOINT space (an answer on A also
  prunes B and C), most-likely-first fallback at the end. Judge's tie-break among the three perfect
  candidates was its own judgement about new instances, not a measured difference; it also flagged the
  tokens_out=0 accounting gap and ignored that column.
- Trace audit (batch 0, loose query form): every query sent by r1g2 / r0g2 / r1g0 appears in a model
  reply (my first tag-only audit under-counted: the harness accepts an untagged `X(a, b, c)` when it is
  the only one in the reply). Final <guess> written by the model and equal to the returned guess in 9/10
  (r1g2), 9/10 (r0g2), 8/10 (r1g0); the rest assembled from the model's numbered picks.
- Prequential batch 1 (10 fresh instances each): strict arm 10/10 joint truth table correct, labeled arm
  10/10. Both arms now proposing on batch 1 (strict: b1r0 sessions started 17:0x).

### 2026-10-12 17:20 EDT — batch 1 decisions: accuracy saturated, judges differ on ties
- Strict (1390 s): all 7 harnesses (incumbent r0g2 + 6 new) 10/10 joint truth table correct on batch 1.
  Judge kept the incumbent ("no candidate clearly better", as instructed). From the traces: incumbent
  11.1 queries/episode; the four candidates that spend leftover queries re-checking their deductions use
  all 18 (the judge counted ~7 extra checks/episode that never contradicted the rule reading).
- Labeled (1001 s): all 7 at 10/10 as well. Judge switched to b1r1g2: fewest queries (9.9 vs 10.5),
  refuses queries that carry much less information than the best available, accepts the model's tag
  with a missing `<`, and its final guess is always a joint-consistent gate split. The switch is a
  judgement call on secondary criteria, not an accuracy difference.
- Batch 2 prequential: both arms 10/10 again (strict with r0g2, labeled with b1r1g2). 30-instance
  streams: both arms 0/10 -> 10/10 -> 10/10 prequential after one batch of adaptation.
- Both judges independently flagged tokens_out=0 as an accounting gap and ignored it (correct: cache
  hits from the proposers' own check runs carry no usage; traces of cached candidates also show
  tok_in = 0 and duration 0). Fix in phase 2.

### 2026-10-12 17:51–18:10 EDT — both pilots finished; label-leak audit of every session
- Final (30 instances, 3 batches): strict prequential circuit-wise accuracy 70.0% / joint truth table
  correct 66.7% (0/10 -> 10/10 -> 10/10), transductive 100%; labeled 68.9% / 66.7%, transductive 100%.
  Batch-2 judges both switched on secondary criteria (strict -> b2r0g2 after an offline simulation of
  NOT-heavy inventories: over-budget runs 1/4 vs incumbent 12/28; labeled -> b2r0g0: 0 re-asks, 11.5
  queries vs 12.1; it also found two round-1 candidates byte-identical to b2r0g0 except docstrings, and
  a candidate with an undefined attribute in a fallback path). Strict's final harness spends leftover
  queries re-checking deductions (18.0 queries/episode vs 11-13 before).
- Cost: strict 14 proposer + 3 judge sessions, $14.25 + $1.63 subscription-priced, 99 min wall;
  labeled $13.17 + $0.86, 95 min. Token columns for cached candidates read 0 (fixed in phase 2).
- Phase 2 applied after the runs ended: cd_tthe keys renamed to the official names, cache entries now
  store usage, pilot logs migrated (circuit_wise_accuracy recomputed from the gold records).
- AUDIT (events.jsonl of all 42 sessions, looking for gold VALUES or instance truths in tool results):
  strict b0 round-0 proposers (all three) and the b0 judge: CLEAN. From b0 round 1 on, proposers read
  `branch_lineage.jsonl` (aggregate `exact` of earlier candidates) and in b1/b2 four proposers read trace
  JSON heads containing per-episode gold records of the incumbent; the b2 judge saw aggregate exact in
  the lineage (said it ignored it); b1 judge clean. No session saw an instance truth table. Labeled arm:
  labels are shown by design; no instance truths seen either.
  => Only the batch-0 step of the strict arm (bare -> r0g2, judged label-free, 10/10 prequential on
  batch 1) is a clean zero-hint result. The stream as a whole is not. Leak channels: gold inside trace
  JSON, checks/res_*.json and solve_jobs results; aggregate exact in branch_lineage/opt_log.
- Next: gold moved out of the run directory (controller-only file), strict lineage/opt_log without
  gold, automatic per-session audit that discards a candidate whose session touched gold, then rerun.

### 2026-10-12 18:20 EDT — label isolation for cd_tthe (phase 3), before the rerun
- Worker `reveal` levels: none (proposer self-checks in strict), outcome (labeled arm: joint truth table
  correct / circuit-wise accuracy of the episode, nothing oracle-derived), controller (everything).
- Controller-private store `~/.ar_harness_controller/<run>/` (outside the repo): full per-episode records
  (gold + oracle signals), opt_log_full / branch_lineage_full. The run directory gets only public
  records: traces without gold and with public signals only (health, crashed, parsed, n_queries,
  repeat); opt_log / lineage with public aggregates (strict: n, health, tokens; labeled: + outcome).
  When the run ends the full logs replace the public ones and the private records are copied to
  <run>/controller/ (git-ignored, goes to the release tarball).
- Prompts carry an OFF-LIMITS paragraph (controller dir, mt_puzzles, stage0 unless lib arm, solver_cache,
  other sessions, .claude, anything with truth tables).
- Automatic session audit after every proposer/judge session (requests + returned values; strict also
  forbids gold records; both forbid instance truths and oracle-signal values): a violating proposer's
  candidate is discarded, a violating judge's verdict is discarded (incumbent kept). Results in
  <session>/audit.json and <run>/session_audits.jsonl.
- Verified on a mock dry run (dry_iso) and on synthetic transcripts. Reruns: strict2_b10g3r2 /
  labeled2_b10g3r2, same stream (seed 0, 30 held-out instances), same B=10 G=3 R=2.

### 2026-10-12 18:20–18:40 EDT — rerun strict2/labeled2 stopped: two more channels; strict3/labeled3 launched
- The session audit fired on the first adversarial-audit proposers of both arms: they read
  `~/.claude/projects/-Users-zarreboln-Desktop-ar-harness/memory/` (notes written by pilot sessions from
  batch 2 on). Root cause: `_memory_dir()` built the slug without turning `_` into `-`, so it never
  matched Claude Code's real directory and the isolation was a no-op since the first run. The strict2
  proposer also built its candidate from `agents/cand_pilot_labeled_b10g3r2_b2r1g1.py` (pilot candidates
  were still in the sandbox) and browsed `logs/pilot_*`. Both runs killed at batch 0 (claude -p children too).
- Cleanup: memory notes archived to experiments/08/cross_session_memory_notes and removed; pilot
  candidates moved to experiments/08/candidates (git mv); all previous run dirs moved out of the repo to
  ~/.ar_harness_controller/archive; `agents/` = bare.py (+2 plumbing stubs), `logs/` empty.
- Code: `_memory_dirs()` (correct slug, every project dir under the repo root); audit also rejects paths
  outside meta/cd_tthe, other runs' logs, git commands (Claude Code's tool-results spill files excepted);
  OFF-LIMITS text extended. Unit cases + mock dry run pass. strict3_b10g3r2 / labeled3_b10g3r2 launched
  18:24 EDT (same 30 instances). The slug fix landed after launch; for these two runs the memory channel
  is closed by the audit (any .claude/ read or write discards that session's work), not by the move.

### 2026-10-12 18:30 EDT — which Claude the proposer/judge actually are
- All cd_tthe sessions (pilots, strict2/labeled2, strict3/labeled3) run `claude -p --model opus` through
  Claude Code 2.1.295; the session init and the billing line resolve that alias to canonical model id
  `claude-opus-5-5` (plus a few hundred tokens of `claude-haiku-5-5` per session: Claude Code's own
  auxiliary calls). So proposer = judge = Claude Opus 5.5 by model id; the user's standing instruction
  said "opus5". Flagged to the user; the running strict3/labeled3 keep the alias for consistency with
  the pilots unless told otherwise.
- 18:31 user decision: keep Opus 5.5 (claude-opus-5-5) for proposer and judge; strict3/labeled3 continue.

### 2026-10-12 18:43 EDT — strict3/labeled3 stopped (audit false positive), strict4/labeled4 launched
- The outside-folder rule `ar_harness/(?!meta/cd_tthe/)` matched every session's own prefix
  `cd .../meta/cd_tthe;` (no trailing slash), so the first proposers of both arms were discarded for
  nothing. Fixed to `ar_harness/(?!meta/cd_tthe\b)`; 12 unit cases (own-folder prefixes, repo root,
  sibling folder, experiments, ../.., other runs, git, memory, scratchpad) pass. Both runs killed,
  their files archived, sandbox back to agents/=bare.py and logs/ empty (plumbing stubs removed too).
- Relaunched as strict4_b10g3r2 / labeled4_b10g3r2 (same 30 instances, same settings, Opus 5.5).

### 2026-10-12 19:02 EDT — audit rewritten; strict6/labeled6 launched
- strict4/labeled4 (18:42) stopped at 18:55: three false positives in one session (`ls ../../` from
  inside the trace folder resolves inside the run; `circuit_joint_lookup` appears in our own audit.py
  source; the proposal card mentions cd_oracle in prose). strict5/labeled5 (18:57) stopped at 19:00:
  my test gate did not stop the launch on two failing cases (reading proposer.py returns the pattern
  word `truth_bits`; Claude Code's tool-results spill files live under .claude).
- audit_session rewritten: requests judged on PATHS only (Bash command text, Read/Write/Edit paths,
  Glob/Grep paths), resolved against the command's `cd` target, with a `cd` landing outside flagged;
  absolute tokens must start at a real top-level dir (no more /def, /query fragments); value patterns
  require data form (lower-case JSON booleans, key-colon). 30 unit cases incl. reading every source
  file; regression on the 42 archived pilot transcripts leaves only real escapes (cd .., ../journal.md,
  memory dir, stage0/, repo root). Note: the pilot's b1 judge and several proposers did `ls ../`.
- strict6_b10g3r2 / labeled6_b10g3r2 launched (same 30 instances, Opus 5.5).
- 19:21 strict6 b0 round 0: audit discarded the conservative-repair candidate for a request containing
  `solver_cache` — the session had written its own parallel check runner embedding the solver config
  (cache_dir path) because the given check script runs 10 instances sequentially (~3 min each) and its
  10-minute background cap would cut it short. Not a label channel (the cache holds only the 7B's replies).
  Rule removed from NAME_RULES and the OFF-LIMITS text for future launches; strict6/labeled6 keep running
  with the old module loaded (label isolation unaffected; one round-0 candidate lost). Other 5 round-0
  sessions clean.

### 2026-10-12 19:44 EDT — strict6 batch 0 decided (clean run), 2577 s
- Sessions: 7 (6 proposers + judge); 5 clean, 2 proposer candidates discarded (r0g0: solver_cache name
  rule, my false positive; r1g2: a stray `cat > …/tmp_unused` typo that wrote an empty file to the
  Desktop, which the session itself removed — a real out-of-folder write by the rule). Judge clean.
- Pool (controller gold, cwa / joint): bare .07/.00; r0g1 .57/.10; r0g2 1.00/1.00; r1g0 .43/.00;
  r1g1 1.00/1.00. Public summary to the judge carried only health/tokens.
- JUDGE -> b0r1g1. It built its own label-free checker (enumerate rule-allowed triples, filter by the
  answers, check uniqueness and that the guess equals the unique solution) and tabulated all five.
  r0g2 and r1g1 tied at 10/10; it rejected r0g2 because the model took the harness's top-ranked query
  in 113 of 126 turns and 5 queries were refused ("the query-choosing the interface forbids"), and chose
  r1g1: keeps bare's conversation, appends a board of observed / rule-forced / unknown outputs after each
  answer, never ranks, suggests or rejects queries, stops when nothing is unknown, then has the model copy
  the board circuit by circuit (tables contradicting board or rules are sent back). Transductive 100/100.
  Note: this is a state-only harness (cf. veto_only / answer_grid in cd_evolve) reaching 100% with the
  model choosing every query itself.

### 2026-10-12 21:30 EDT — strict6 / labeled6 ended; the Max subscription hit its session limit at ~19:50
- From ~19:50 EDT every `claude -p` call returned "You've hit your session limit · resets 11pm
  (America/New_York)" immediately. The loop treated those as sessions that produced no candidate
  (branches fell back to their parents) and judges that wrote no choice (incumbent kept), so the runs
  "finished" with default decisions. Also one more audit false positive: strict6 b1r0g1 was discarded
  because the docstring it was writing (inside a Bash heredoc) contained the text `../audit.py`; the path
  scanner reads heredoc/text content of Bash commands.
- strict6 valid parts: batch 0 (judge ran at 19:44, clean) chose b0r1g1 (deduction board, state-only);
  prequential of that harness on batch 1 = 9/10 joint (cwa .97), on batch 2 = 10/10. Batch-1 round 0
  ran (b1r0g0 .80, b1r0g2 1.00); round 1, the b1 judge, and all of batch 2 hit the limit. Stream:
  prequential cwa 67.8 / joint 63.3 (bare 0/10 -> 9/10 -> 10/10), 19/20 on fresh instances with a
  harness grown and selected label-free.
- labeled6: batch-0 proposals ran (r0g2, r1g0, r1g2 at 10/10; all 6 sessions + judge audits clean) but
  the b0 judge hit the limit -> bare kept; batches 1-2 entirely void. The arm must be rerun.
- Lesson: the wrapper must detect the limit message and pause/abort instead of falling back silently.
  Today's ~60 Opus sessions (pilots + 5 launches) consumed the 5-hour window.

### 2026-10-12 21:38 EDT — resume support, limit handling, audit fixes; strict6/labeled6 resumed
- New 7B server job 25690221 (node3204:8122), tunnel 18000 re-pointed; old job 25669627 expires 22:07.
- optimize.py `--resume-from BATCH ROUND`: prunes log rows at/after that point (full + public), restores
  public logs in the run dir, removes the stale controller copy, reloads every complete observation of
  the batch from the controller store and the branch heads from the lineage; ROUND == --rounds means
  judge-only. Verified on a mock run (resume mid-batch and judge-only).
- proposer.py: sessions whose response is the subscription-limit message are renamed *_LIMIT, the loop
  probes every 10 min with a one-word call and reruns the session once the limit lifts (up to 4 h).
- Audit: heredoc bodies are text (a '../x' there counts only in an access-like context such as
  open('../x'); absolute paths always count); the shell's working directory is tracked across a
  session's Bash calls (Claude Code keeps it) — today's `python3 ../audit.py` from agents/ was inside
  the folder; other runs' candidate files (agents/cand_<other>_*.py) are off-limits. Unit set 34 cases
  + the real transcripts; no strict6/labeled6 session had touched the other arm's candidates.
- Resumed: strict6 from batch 1 round 1 (heads b1r0g0 / b0r1g1 / b1r0g2), labeled6 from the batch-0
  judge over its 6 candidates. Subscription window may still be tight before 23:00; the loop will pause.
- 21:37 labeled6 resumed (judge-only at batch 0, then on). 21:37 strict6 launch failed: zsh does not
  word-split an unquoted $COMMON variable, argparse saw one argument; relaunched 21:38:46 with explicit
  arguments: "resuming batch 1 at round 1: reloaded b0r1g1, b1r0g0, b1r0g2", stale controller copy removed.
- 21:57 strict6 batch 1 judged (resumed run, all 4 sessions clean): pool b0r1g1 .90, b1r0g0 .80,
  b1r0g2 1.00, b1r1g0 .90, b1r1g1 1.00, b1r1g2 1.00 (joint, controller gold). Judge REJECTED the three
  10/10 candidates for choosing queries (b1r1g1: harness picks the best-splitting of 5 model-listed
  queries; b1r0g2/b1r1g2: "closest to 50% now" hints followed ~95%), ranked the valid ones with its own
  label-free checker (incumbent 9/10, b1r0g0 8/10 + a crash, b1r1g0 9/10) and committed b1r1g0 (board
  harness; skips queries the board already knows; lists rule-allowed tables at the end; stops the
  early-guess loop). Prequential on batch 2 pending.
- 21:38 labeled6 batch 0 judged (resumed): b0r1g2 (joint-space enumerator + "reject a query that splits
  less evenly than the best"); prequential on batch 1 = 10/10.
- 22:1x user: after today's runs, (a) frozen evaluation of the finished harnesses on the full v2 held-out
  (200 instances, no proposing, 7B only) -> meta/cd_tthe/evaluate.py written and mock-tested (reports
  official metrics with bootstrap CIs on all 200 and on the 170 the stream never used); (b) then add
  --regime open|state_only and run both regimes on the full set (config to be confirmed: 200 instances,
  B=20, strict, sequential, state_only first).
- 22:29 labeled6 finished (resumed run): prequential cwa 70.0 / joint 66.7 (0/10 -> 10/10 -> 10/10),
  transductive 100/100; commits b0r1g2 -> b1r0g2 (batch 1) -> kept (batch 2).

### 2026-10-12 22:33 EDT — strict6 finished (resumed run)
- Batch 2 judge (clean): all six harnesses gave identical, answer-consistent guesses; chose b2r0g1 for
  fewer queries (10 vs 16 on i206, 11 vs 15 on i238, 16 vs 17 on the hardest i266). Final harness
  b2r0g1. Stream prequential: cwa 67.8 / joint 63.3 (0/10 -> 9/10 -> 10/10), uniq@end 80; transductive
  98.9 / 96.7. Sessions: 32, 29 clean; the 3 discards: solver_cache (my over-broad rule, removed),
  stray `cat >` write to ~/Desktop (real), `python3 ../audit.py` from agents/ (cwd not tracked, fixed).
  Every judge session clean. labeled6: 36/36 clean, 0/10 -> 10/10 -> 10/10, cwa 70.0 / joint 66.7.
- 22:35 frozen evaluation launched: 11 harnesses x 200 held-out (48 workers), logs/eval_heldout200.
- 23:47 frozen evaluation done (11 harnesses x 200 held-out, 48 workers): bare cwa 11.8 / joint 0;
  strict final b2r0g1 99.5 / 98.5 (unseen 98.2), b1r1g0 99.3 / 98.0, b0r1g1 97.0 / 92.0; labeled finals
  100 / 100 at 215 tokens/ep; pilot r0g2 99.5 / 99.5; rejected query-steering ones 99.5 / 99.5 and
  99.0 / 97.0. Experiment 09 written.

### 2026-10-13 00:30 EDT — referee candidates for SP (user: test all four on TurtleBench-150)
- Deleted the never-used Llama-3.1-405B-Instruct-FP8 download (454 G) from scratch to make room.
- New scripts engaging/dl_hf_scratch.sbatch (HF cache on scratch) and engaging/calib.sbatch (serve one
  model with vLLM 0.31, run runner/turtle_judge_eval.py, exit). Downloads 25698267-70, calibrations
  25698271/2/3/5 (dependency-chained): openai/gpt-oss-120b (1xH200, max_tokens 2048 for its reasoning
  channel), Qwen/Qwen3-32B (1xH200, thinking off), RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic (1xH200),
  Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 (2xH200, PP=2). Bar: >= 82.0% (405B-INT4) / 82.7% (official).
- 405B job 25698041 resubmitted on mit_preemptable (12 h) after the normal partition's 2-GPU cap.

### 2026-10-13 01:00 EDT — SP ported into the TTHE loop; referee calibrations queued
- meta/cd_tthe now has a task switch (`tasks.py`; `--task cd|sp` in optimize.py / evaluate.py; worker,
  static audit, session audit, prompts, traces, summaries all task-aware). SP pieces: sp_bridge.py (SPEnv
  over AR-Bench data; referee = any served model through the official referee prompt, cached; official
  char-level F1 and key-question recall judged by the referee), sp_harness_base.py, agents_sp/bare.py
  (the official zero-shot loop), sp_signals.py (public signals only: questions used, repeat, yes/no/unknown
  rates, "not a question" rate). Both tasks run end to end on mocks (stub proposer); CD regression ok.
- Downloads: the hf cache uses a content-addressed blobs/xx/ layout, so `du` on the model dir under-reports;
  the snapshots dereference to full sizes (Qwen3-32B 64 G). Calibration jobs released.
