# 已发表方法的 test-split 数字 (2026-10-30)

起止 00:18–00:27 UTC,约 9 分钟,全部 CPU,零 GPU、零花费。

## 结论:不需要重跑,这批数据一直就在 test 上

`bath/run_sharded.sh` 把数据路径写死成 `DATA=$HOME/code/AR-Bench/data/$TASK/test.json`,
而 `make_ar_splits.py` 的定义是 search/selection 从 **train** 里切,test **就是** AR-Bench
自己的 `data/<task>/test.json`,原封不动。所以 2026-10-24..26 在 bath 上跑的那批
`~/results/<task>_<method>.json` 本来就是 test-split 的结果。

`published_baselines.json`(selection)和这批(test)是**两次独立测量**,不是同一批数据换算。
对照:`dc/few_shot_inst` selection 34.0 / test 39.0。

## 判官与指标对齐(与我们的进化系统逐项核对)

| | 基线那侧 | 我们这侧 | |
|---|---|---|---|
| policy | Qwen2.5-7B-Instruct frozen, 0.7/0.7, 25 轮 | 同 | ✓ |
| GN 反馈 | 纯函数,两侧都没有第二个模型 | 同 | ✓ |
| DC 嫌疑人 | 本地同一个 7B @ 0.7/0.7 (`DC_RESPONDER=local`) | `dc/env.py` 同模型 @ 0.7 | ✓ |
| SP 局内 referee | jev shim :8099,`--response_model jev-latest`,0.7/0.7 | `REFEREE_MODEL` 默认 `jev-latest`,同 shim | ✓ |
| SP 终局判官 | `sp_semantic_equivalence.py`,`JEV_MODEL=jev-1.14.0` | `score_val.py --judge jev`(默认)→ 同模块同版本 | ✓ |
| SP 指标 | `(equivalent + 0.5·partial)/n` | 同 | ✓ |
| 成本轴 | input + output / episode | 同 | ✓ |

SP 的 wrapper 保真度在 RUN_LOG 里已经验过:我们的 wrapper 0.665–0.675 mean char-F1 对官方
evaluator 的 0.677–0.703(8 个谜题),在噪声内相等。seed 本身是用 AR-Bench 自己的 evaluator
测的,不存在 rewrap 问题。

## 这次为对齐做的修补

- `runner/dc_backfill_output.py`(新):`dc_backfill_tokens.py` 当初只回填了 input,
  DC 三个 prompt 基线的 output 一直是空的,成本轴因此一半是 input-only。output 就是 `record`
  里每个 assistant turn 的内容,重新 tokenize 即可。mean output:zero_shot 718 / few_shot 749 /
  few_shot_inst 802 —— 相对各自 14–29 万的 input 可忽略,但轴现在是统一的。
- 三个 evaluator 用了三套 token 字段名(`input_token`/`input_token_sum`/`input_token_backfilled`),
  `meta/tools/score_test_grid.py` 一并认。

## 已知缺口(三个,都已记在 json 的 `_gaps` 里)

- **`gn/tot` 跑不了**,不是没跑。`extract_four_digit_numbers` 标注 `-> List[str]` 实际返回 `str`,
  循环在逐字符迭代,`compare_guess` 第一个实例就抛。`gn/tot_fixed` 是修好的那个。
- **`sp/tot` 的成本不可恢复**。回填脚本自己记了原因:record 只留被选中的路径,`expand_node`
  每层生成的兄弟分支和 propose 模板都没存。准确率(4.0)是可靠的。只有重跑能补(约 $0.2 jev + 若干 GPU 分钟)。
- **`sp/uot` 的 episode 文件在已停用的 h100**(百度网盘 `/h100_backup/`),但它的 jev 判决留在
  `sp_equiv_jev.json` 里,所以准确率能按本表口径重算 = **2.5**。
  ⚠ RUN_LOG:49 记的 `sp uot = 5.0` 是 **non-strict** 口径 `(eq+partial)/n`(eq=0, partial=5),
  跟主表的 `(eq+0.5·partial)/n` 不是一个指标。那一行如果直接进表会是指标不一致。

## 全表

见 `published_baselines_test.json`。重跑命令:`~/venv/bin/python ~/code/score_test_grid.py`(bath)。

## 主表口径决定 (2026-10-30)

只保留三个任务都有 accuracy **和** cost 的方法。干净通过的有四个:
`zero_shot` / `few_shot` / `few_shot_inst` / `proactive_cot`。

**ToT 加回来当第五行。** 理由:AR-Bench 的 ToT 本来就是三个任务三份独立实现,但
zero_shot/few_shot 的 prompt 也是每个任务分开写的 —— 横着读一行从来是同一套配方而不是同一段
代码,ToT 按这个标准合格。砍掉它会让 baseline 集合变成「四种提示词写法」,唯一带搜索结构的方法
就没了。取各任务修好的变体:`gn=tot_fixed`、`dc=tot_eval`、`sp=tot_eval`。
唯一瑕疵是 GN 那格的成本只有 input(record 存了 guess_list 不存 conversation),
GN 的 output 约占 input 的 1%,写进脚注。

被刷掉的:`tot`(未修版 GN 跑不了)、`proactive_stop`(GN 没有对应变体,
`dc_proactive_stop.py` 开头解释了为什么 GN 不需要)、`uot`(原始 episode 文件在已停用的 h100)。

主表 `table_main.tex` 和前沿图 `meta/tools/plot_frontier.py` 都已改成全 test 口径;
图直接读 `published_baselines_test.json`,不再手抄数字,表和图不会走偏。
