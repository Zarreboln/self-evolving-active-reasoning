"""Which of the six factors did the search actually write code for?

Section 7.1's Step 2 decomposes the Bayesian agent for this POMDP into six factors. The report
argues about one of them -- question selection -- from the harnesses that were kept. This counts
all of them, over every candidate the search produced, kept or discarded, so the claim rests on
the population rather than on the frontier.

A harness can touch several factors, so each is a separate yes/no judgement against the same
code, in one call. The wording of each criterion is Section 7.1's own.

What is read: the module docstring, which in these harnesses states the mechanism, plus the names
of its methods, which say where the mechanism sits. Not the full body -- the bodies run to
hundreds of lines of prompt text and parsing, and the question is what the harness is for.

    JEV_API=... python meta/tools/factor_census.py <dir>:<task> ... --out census.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

JEV_API = os.environ.get("JEV_API", "https://api.typesafe.ai/v1/systemone")

FACTORS = {
    "hypothesis_generation":
        "The harness writes down SEVERAL rival answers, explanations or stories at once and "
        "carries that set between turns, so that evidence can eliminate some of them. Carrying a "
        "single running best guess, however often it is revised, is NOT this -- there is no set "
        "to eliminate from. Enumerating a set the task's rules already fix, such as all "
        "four-digit numbers or the five suspects the case names, is NOT this either: that state "
        "space was given, not proposed.",
    "observation_model":
        "The harness predicts what answer a question would receive under a given state, or models "
        "how the environment replies, rather than only reading the reply it got.",
    "belief_update":
        "The harness revises what it believes from the answer it received -- striking out "
        "candidates, scoring suspects, maintaining a ledger of confirmed facts, or keeping a "
        "consistent set outside the model.",
    "question_selection":
        "The harness chooses WHICH question to ask by what the answer would be worth -- comparing "
        "candidate questions, vetoing uninformative ones, or ordering them by how much they would "
        "narrow things down. Asking in a fixed or prompted order is NOT this.",
    "stopping":
        "The harness decides when to stop asking and answer, from whether it has enough evidence, "
        "rather than running to the end of the budget.",
    "answer_construction":
        "The harness changes how the FINAL answer is written or certified -- rephrasing it, "
        "checking it against the record, or requiring confirmation before committing to it. This "
        "is not one of the six factors; it is here to catch the work that went somewhere else.",
}
INSTRUCTIONS = ("You are reading one agent harness for a multi-turn question-asking task. Judge "
                "whether the harness does the thing described, from its stated mechanism and the "
                "names of its parts. Answer Yes only if the harness clearly does it.")


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k.strip()
    return subprocess.run(["security", "find-generic-password", "-s", "typesafe_token", "-w"],
                          capture_output=True, text=True, check=True).stdout.strip()


def describe(path):
    """Docstring plus the names of the harness's own methods."""
    src = open(path).read()
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names += ["%s.%s" % (node.name, f.name) for f in node.body
                      if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return " ".join(doc.split())[:4000], names


def judge(desc, names, key, model, retries=4):
    body = {"model": model,
            "state": {"harness_mechanism": desc, "harness_parts": ", ".join(names)},
            "questions": {k: {"type": "choice", "instructions": INSTRUCTIONS,
                              "criteria": {"Yes": v, "No": "The harness does not do that."}}
                          for k, v in FACTORS.items()}}
    req = urllib.request.Request(JEV_API, method="POST", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    last = None
    for a in range(retries):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=90))
            return ({k: (d["answers"].get(k) or {}).get("choice") == "Yes" for k in FACTORS},
                    (d.get("usage") or {}).get("input_tokens", 0))
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"Jev failed after {retries} attempts: {last}")


SKIP = ("__init__", "baseline_", "template_zero_shot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", metavar="DIR:LABEL")
    ap.add_argument("--model", default="jev-1.14.0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = api_key()
    out, spent = [], 0
    for spec in a.sets:
        d, label = spec.rsplit(":", 1)
        for f in sorted(Path(d).glob("*.py")):
            if any(f.stem.startswith(s) or f.stem == s for s in SKIP):
                continue
            desc, names = describe(f)
            got, tin = judge(desc, names, key, a.model)
            spent += tin
            out.append({"set": label, "harness": f.stem, **got})
            print("%-14s %-26s %s" % (label, f.stem,
                                      " ".join(k for k, v in got.items() if v) or "(none)"),
                  file=sys.stderr)
    json.dump(out, open(a.out, "w"), indent=1)
    print("%d harnesses, $%.4f" % (len(out), spent * 0.042 / 1e6), file=sys.stderr)


if __name__ == "__main__":
    main()
