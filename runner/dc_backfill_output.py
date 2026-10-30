"""Recover per-episode OUTPUT tokens for the three DC runs whose evaluator recorded neither.

dc_backfill_tokens.py already recovered the input side for dc_{zero_shot,few_shot,few_shot_inst}
by re-tokenizing each policy call's prompt prefix. The output side is simpler: every assistant
turn in `record` is exactly one completion, so tokenizing its content and summing reproduces what
the server generated. Same tokenizer, same model, so this is a count and not an estimate.

Writing both sides means the Pareto cost axis is input+output on all three tasks, which is what
RUN_LOG states it is. Without this, DC's three prompt baselines sit on an input-only axis while
every other cell is input+output.
"""
import json
import os

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
TARGETS = ["dc_zero_shot", "dc_few_shot", "dc_few_shot_inst"]

for name in TARGETS:
    p = os.path.expanduser("~/results/%s.json" % name)
    rows = json.load(open(p))
    n_done = 0
    for r in rows:
        if r.get("output_token_backfilled") is not None:
            n_done += 1
            continue
        rec = r.get("record")
        if not isinstance(rec, list):
            r["output_token_backfilled"] = 0
            continue
        total = 0
        for m in rec:
            if m.get("role") == "assistant":
                total += len(tok(m.get("content") or "", add_special_tokens=False)["input_ids"])
        r["output_token_backfilled"] = total
    out = [r["output_token_backfilled"] for r in rows]
    json.dump(rows, open(p, "w"))
    print("%-18s n=%3d  already-had=%d  mean output=%9.0f  (input mean=%9.0f)"
          % (name, len(rows), n_done, sum(out) / len(out),
             sum(r.get("input_token_backfilled") or 0 for r in rows) / len(rows)))
