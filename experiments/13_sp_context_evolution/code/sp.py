"""Situation Puzzles (AR-Bench @9971322) environment, official zero-shot loop + an optional playbook slot.

Prompts are extracted verbatim from the downloaded AR-Bench prompt.py with `ast` (string constants only;
the untrusted file is never imported or executed). The loop mirrors arbench/reasoner/sp/sp_evaluator.py:
system = propose_template(turn, surface); each turn user "Turn {t}: please propose your next question.";
the raw actor reply is sent to the referee (system_prompt_with_2shots with surface+bottom); the referee reply,
stripped of whitespace and a trailing ".", is appended as the next user message; finally get_answer_prompt.
Defaults follow the official script: actor temperature 0.7 / top_p 0.7, referee temperature 0.7 / top_p 0.7.

The only intervention point is PLAYBOOK text appended to the system message (same header as the team's
ctx_harness.py), so context arms differ from the bare loop in exactly one string.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from llm import chat

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "arbench_9971322"   # AR-Bench repo at commit 9971322
PROMPT_FILE = DATA / "src" / "arbench_reasoner_sp_prompt.py"
BUDGET = 25

ACTOR = "qwen/qwen-2.5-7b-instruct"
# Referee chosen 2026-11-07 by TurtleBench-150 (seed 7) calibration, runs/referee_calibration/summary.json:
# qwen3-235b-a22b-2507 acc 0.760, macro recall 0.735 (Unknown 0.63); llama-3.3-70b 0.753 / 0.596 (Unknown 0.11);
# claude-haiku-5.5 (reasoning off) 0.533; sonnet-5.5 needs mandatory reasoning; qwen3-32b ignores reasoning-off.
REFEREE = "qwen/qwen3-235b-a22b-2507"
ROBUST_REFEREE = "meta-llama/llama-3.3-70b-instruct"   # robustness re-run on a test subset
# Key-point judge switched 2026-11-07 after comparison with an Opus-5.5 reference on 100 items:
# haiku-5.5 + reasoning 80/100 hit/no-hit agreement (cheapest strong option); qwen3-235b 71/100 (lenient: called 50
# hits where Opus called 25).
KEYJUDGE = "anthropic/claude-haiku-5.5"
JUDGE_REASONING = {"anthropic/claude-haiku-5.5", "anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5",
                   "google/gemini-3.7-flash"}   # gemini: same 3000-token budget as in the judge comparison
ROBUST_JUDGE = "google/gemini-3.7-flash"      # cross-family re-check: 91/98 agreement with the Opus reference


def _load_prompts() -> dict[str, str]:
    tree = ast.parse(PROMPT_FILE.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                out[node.targets[0].id] = node.value.value
    return out


P = _load_prompts()
PROPOSE = P["propose_template"]
REFEREE_SYS = P["system_prompt_with_2shots"]
FINAL_PROMPT = P["get_answer_prompt"]
KEYPOINT_PROMPT = P["keypoint_hits_prompt"]
GUESS_EVAL = P["guess_eval_prompt"]

PLAYBOOK_HEADER = ("\n\n# Playbook\n"
                   "Notes collected from earlier episodes of this task. They are guidance, not facts about\n"
                   "this instance. Follow them unless this instance contradicts them.\n")


def keypoints_of(p: dict) -> list[str]:
    kq = p["key_question"]
    if isinstance(kq, str):
        try:
            kq = json.loads(kq)
        except Exception:
            kq = ast.literal_eval(kq)
    return list(kq)


def load(split: str) -> list[dict]:
    rows = json.loads((DATA / "sp" / f"{split}.json").read_text(encoding="utf-8"))
    for i, r in enumerate(rows):
        r["_split"], r["_i"] = split, i
        r["_id"] = f"sp-{split}-{i:03d}"
    return rows


def split_train(rows: list[dict], val_n: int = 100) -> tuple[list[dict], list[dict]]:
    """Deterministic hash split of the official train file into TRAIN / VAL (test.json stays untouched)."""
    order = sorted(rows, key=lambda r: hashlib.sha256(r["_id"].encode()).hexdigest())
    return order[val_n:], order[:val_n]


def f1(pred, truth) -> float:
    common = Counter(pred) & Counter(truth)
    same = sum(common.values())
    if same == 0:
        return 0.0
    pr, rc = same / len(pred), same / len(truth)
    return 2 * pr * rc / (pr + rc)


def system_message(surface: str, playbook: str = "") -> str:
    base = PROPOSE.format(turn=BUDGET, question=surface)
    pb = (playbook or "").strip()
    return base if not pb else base + PLAYBOOK_HEADER + pb + "\n"


def referee(p: dict, question: str, model: str = REFEREE) -> str:
    msgs = [{"role": "system", "content": REFEREE_SYS.format(question=p["surface"], answer=p["bottom"])},
            {"role": "user", "content": question}]
    a = chat(model, msgs, role="referee", temperature=0.7, max_tokens=16, extra={"top_p": 0.7},
             tag=p["_id"])
    return a.strip().strip(".")


def actor_next(msgs: list[dict], seed: int, model: str = ACTOR, tag: str = "", temperature: float = 0.7) -> str:
    return chat(model, msgs, role="actor", temperature=temperature, max_tokens=256, seed=seed,
                extra={"top_p": 0.7}, tag=tag)


def run_episode(p: dict, playbook: str = "", seed: int = 0, actor_model: str = ACTOR,
                referee_model: str = REFEREE, prefix: list[tuple[str, str]] | None = None,
                choose=None) -> dict:
    """Play one episode. `prefix` replays fixed (question, answer) turns first. `choose(msgs, t)` may
    return a question to use instead of sampling (used by oracle/teacher ladders); default samples once."""
    msgs = [{"role": "system", "content": system_message(p["surface"], playbook)}]
    hist = []
    for t in range(1, BUDGET + 1):
        msgs.append({"role": "user", "content": f"Turn {t}: please propose your next question."})
        if prefix and t <= len(prefix):
            q, a = prefix[t - 1]
        else:
            q = choose(msgs, t, hist) if choose else None
            if q is None:
                q = actor_next(msgs, seed=seed * 1000 + t, model=actor_model, tag=p["_id"])
            a = referee(p, q, model=referee_model)
        msgs.append({"role": "assistant", "content": q})
        msgs.append({"role": "user", "content": a})
        hist.append((q, a))
    msgs.append({"role": "user", "content": FINAL_PROMPT})
    pred = chat(actor_model, msgs, role="actor", temperature=0.7, max_tokens=512, seed=seed * 1000 + 99,
                extra={"top_p": 0.7}, tag=p["_id"])
    return {"id": p["_id"], "seed": seed, "playbook_sha": hashlib.sha256(playbook.encode()).hexdigest()[:12],
            "history": hist, "pred": pred, "messages_len": len(msgs)}


_HIT = re.compile(r"Hit point:\s*([\d,\s]+)")


def key_hits(p: dict, question: str, model: str = KEYJUDGE) -> set[int]:
    """Team variant of AR-Bench keypoint judging: one question at a time (temperature 0). Same prompt and
    reasoning setting as the judge comparison (runs/labeling/judgecmp_*), where haiku-5.5+reasoning agreed with
    the Opus-5.5 reference on 80/100 items vs 71/100 for qwen3-235b."""
    prompt = KEYPOINT_PROMPT.format(question=p["surface"], answer=p["bottom"],
                                    keypoints=place_keypoints(keypoints_of(p))) + \
        f"\n\nPlayer's question: {question}\n\nAnswer in the format 'Hit point: x, y' or 'No point is hit'."
    txt = chat(model, [{"role": "user", "content": prompt}], role="keyjudge", temperature=0.0,
               max_tokens=3000 if model in JUDGE_REASONING else 40, reasoning=model in JUDGE_REASONING, tag=p["_id"])
    m = _HIT.search(txt or "")
    n = len(keypoints_of(p))
    return {int(x) for x in re.findall(r"\d+", m.group(1))} & set(range(1, n + 1)) if m else set()


def answer_match(p: dict, pred: str, model: str = KEYJUDGE) -> int:
    """AR-Bench's own guess_eval_prompt: number of keypoints the final answer matches (0..6)."""
    prompt = GUESS_EVAL.format(question=p["surface"], answer=p["bottom"],
                               keypoints=place_keypoints(keypoints_of(p)), pred=pred)
    txt = chat(model, [{"role": "user", "content": prompt}], role="judge", temperature=0.0,
               max_tokens=3000 if model in JUDGE_REASONING else 200, reasoning=model in JUDGE_REASONING, tag=p["_id"])
    m = re.search(r"Match count:\s*(\d+)", txt or "")
    return min(int(m.group(1)), len(keypoints_of(p))) if m else 0


def place_keypoints(kps: list[str]) -> str:
    return "".join(f"{i + 1}. {k}\n" for i, k in enumerate(kps))


def clean_q(q: str) -> str:
    q = (q or "").strip()
    return re.sub(r"^\s*Q\s*:\s*", "", q.splitlines()[0] if q else "").strip()


def score_episode(p: dict, ep: dict, judge_keys: bool = True) -> dict:
    hist = ep["history"]
    answers = [a.lower() for _, a in hist]
    out = {"id": ep["id"], "seed": ep["seed"], "playbook_sha": ep["playbook_sha"],
           "f1_char": f1(ep["pred"], p["bottom"]), "f1_word": f1(ep["pred"].split(), p["bottom"].split()),
           "unknown_rate": sum(a.startswith("unknown") for a in answers) / max(1, len(answers)),
           "yes_rate": sum(a.startswith("yes") for a in answers) / max(1, len(answers)),
           "pred_words": len(ep["pred"].split())}
    if judge_keys:
        hit_by_turn, seen = [], set()
        for q, _ in hist:
            seen |= key_hits(p, clean_q(q))
            hit_by_turn.append(len(seen))
        n = len(keypoints_of(p))
        out["key_recall"] = len(seen) / n
        out["key_recall_curve"] = [h / n for h in hit_by_turn]
        out["answer_match"] = answer_match(p, ep["pred"])
    return out
