# The diff against Meta-Harness's loop

`meta_harness.py` is used as published (`meta_harness_official` @ `8123cca`,
`reference_examples/text_classification/meta_harness.py`). Two lines are domain names, not logic,
and this file records them so "we ran their loop" stays checkable with `diff`.

| Line | Published | Ours |
|---|---|---|
| 29 | `BASELINE_FILES = {"__init__.py", "no_memory.py", "fewshot_memory.py", "fewshot_all.py"}` | `{"__init__.py", "baseline_bare.py", "baseline_few_shot_inst.py"}` |
| 203 | `f"from text_classification.agents.{name} import *; print('OK')"` | `f"from arbench_mh.agents.{name} import *; print('OK')"` |

Nothing else changes: iteration accounting, `--resume`, `--fresh`, candidate import-checking, the
Pareto frontier, `finalize_run`, the proposer session and its allowed tools are all theirs.

`claude_wrapper.py` is copied verbatim (MIT) as their reference examples each do, with its
provenance in the first line.

Verify with:

```bash
diff meta_harness_official/reference_examples/text_classification/meta_harness.py \
     meta/arbench_mh/meta_harness.py
```
