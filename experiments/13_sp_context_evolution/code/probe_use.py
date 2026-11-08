"""Asking vs. using: where is information lost - in asking or in using it?
Reuses the saved TEST transcripts (seed 0) of a0, traj and r1; NO new questions are asked, only the final story is
re-written from the same transcript under different conditions:
  orig       the 7B's original final story (already saved)
  qwen235    Qwen3-235B writes the story from the same transcript                      (stronger writer)
  opus       Opus 5.5 writes the story from the same transcript                        (strongest writer)
  facts7b    the 7B first lists confirmed / ruled-out facts, then writes the story      (prompt-only aid)
  yesno7b    the 7B writes from the transcript with Unknown turns removed              (less clutter)
Each story is scored with AR-Bench's guess_eval prompt (key points matched, 0..6) by Haiku and by Gemini, plus char-F1.
Usage: python -I code/probe_use.py [n]   -> runs/probe_use/*.jsonl and report.md
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import evaluate  # noqa: E402
from llm import chat, spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "probe_use"
PUZ = {p["_id"]: p for p in sp.load("test")}
ARMS = ("a0", "traj", "r1")
CONDS = ("orig", "qwen235", "opus", "facts7b", "yesno7b")
WRITERS = {"qwen235": sp.REFEREE, "opus": "anthropic/claude-opus-5.5"}
GEM = "google/gemini-3.7-flash"
FACTS = ("Before answering, list (a) every fact the referee confirmed with Yes and (b) every possibility the referee "
         "ruled out with No. One short line each. Do not write the story yet.")
STORY_AFTER_FACTS = " Your story must be consistent with every confirmed fact and should explain as many of them as possible."


def msgs_for(p, pb, hist):
    m = [{"role": "system", "content": sp.system_message(p["surface"], pb)}]
    for t, (q, a) in enumerate(hist, 1):
        m += [{"role": "user", "content": f"Turn {t}: please propose your next question."},
              {"role": "assistant", "content": q}, {"role": "user", "content": a}]
    return m


def story(cond, p, pb, hist, seed):
    tag = p["_id"]
    if cond in WRITERS:
        m = msgs_for(p, pb, hist) + [{"role": "user", "content": sp.FINAL_PROMPT}]
        opus = cond == "opus"   # Opus 5.5 on OpenRouter requires reasoning; its budget is shared with the answer
        return chat(WRITERS[cond], m, role="writer", temperature=0.7, max_tokens=6000 if opus else 512, seed=seed,
                    reasoning=opus, tag=tag)
    if cond == "yesno7b":
        kept = [(q, a) for q, a in hist if not a.lower().startswith("unknown")]
        m = msgs_for(p, pb, kept) + [{"role": "user", "content": sp.FINAL_PROMPT}]
        return chat(sp.ACTOR, m, role="actor", temperature=0.7, max_tokens=512, seed=seed, extra={"top_p": 0.7}, tag=tag)
    if cond == "facts7b":
        m = msgs_for(p, pb, hist) + [{"role": "user", "content": FACTS}]
        facts = chat(sp.ACTOR, m, role="actor", temperature=0.7, max_tokens=400, seed=seed, extra={"top_p": 0.7}, tag=tag)
        m += [{"role": "assistant", "content": facts}, {"role": "user", "content": sp.FINAL_PROMPT + STORY_AFTER_FACTS}]
        return chat(sp.ACTOR, m, role="actor", temperature=0.7, max_tokens=512, seed=seed + 1, extra={"top_p": 0.7},
                    tag=tag)
    raise ValueError(cond)


def gem_match(p, pred):
    prompt = sp.GUESS_EVAL.format(question=p["surface"], answer=p["bottom"],
                                  keypoints=sp.place_keypoints(sp.keypoints_of(p)), pred=pred)
    txt = chat(GEM, [{"role": "user", "content": prompt}], role="robustjudge", temperature=0.0, max_tokens=2000,
               tag=p["_id"])
    m = re.search(r"Match count:\s*(\d+)", txt or "")
    return min(int(m.group(1)), len(sp.keypoints_of(p))) if m else None


def main(n):
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        rows = [json.loads(x) for x in (ROOT / "runs" / "test" / f"{arm}_s0.jsonl").read_text(encoding="utf-8")
                .splitlines()][:n]
        pb = evaluate.playbook_for(arm, 0)
        for cond in CONDS:
            if cond == "opus" and arm != "a0":    # cost control: the strongest writer only on the empty-playbook arm
                continue

            def one(r):
                p = PUZ[r["episode"]["id"]]
                pred = r["episode"]["pred"] if cond == "orig" else story(cond, p, pb, r["episode"]["history"], 7000)
                return {"arm": arm, "cond": cond, "id": p["_id"], "pred": pred,
                        "haiku": sp.answer_match(p, pred), "gemini": gem_match(p, pred),
                        "f1_char": sp.f1(pred, p["bottom"])}
            with ThreadPoolExecutor(12) as ex:
                res = list(ex.map(one, rows))
            (OUT / f"{arm}_{cond}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in res) + "\n",
                                                     encoding="utf-8")
            g = [x["gemini"] for x in res if x["gemini"] is not None]
            print(f"{arm} {cond}: haiku {mean(x['haiku'] for x in res):.2f} gemini {mean(g):.2f} "
                  f"f1 {mean(x['f1_char'] for x in res):.3f} spent ${spent_usd():.2f}", flush=True)
    report()


def report():
    lines = ["Key points matched by the final story (0-6), same transcripts, seed 0; paired diff vs the 7B's original "
             "story (95% bootstrap CI over puzzles).", "",
             "| arm | condition | Haiku | Gemini | char-F1 | Haiku diff vs orig | Gemini diff vs orig |",
             "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        base = {}
        for cond in CONDS:
            f = OUT / f"{arm}_{cond}.jsonl"
            if not f.exists():
                continue
            res = {x["id"]: x for x in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
            if cond == "orig":
                base = res
            cells = []
            for k in ("haiku", "gemini"):
                common = [i for i in res if i in base and res[i][k] is not None and base[i][k] is not None]
                if cond == "orig" or not common:
                    cells.append("")
                    continue
                m, lo, hi = evaluate.boot([res[i][k] - base[i][k] for i in common])
                cells.append(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            g = [x["gemini"] for x in res.values() if x["gemini"] is not None]
            lines.append(f"| {arm} | {cond} | {mean(x['haiku'] for x in res.values()):.2f} | {mean(g):.2f} | "
                         f"{mean(x['f1_char'] for x in res.values()):.3f} | {cells[0]} | {cells[1]} |")
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
