"""How stable is the equivalence judge? Repeat three diagnostic pairs and count the labels.

A metric whose self-check passes on one run and fails on the next is not measuring the answers.
This repeats each pair N times at the configured temperature and reports the label distribution,
so the instability is a number rather than an impression.
"""
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/tmp_l/code")
from sp_semantic_equivalence import ask_policy, EQUIVALENCE_USER_PROMPT, EQUIV_MAP

BASE = os.environ["POLICY_BASE_URL"]
KEY = os.environ.get("POLICY_API_KEY", "x")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

d = json.load(open("/home/tmp_l/results/sp_equiv_selftest.json"))
gold = d[0]["answer"]
other = d[1]["pred"]
partial = d[2]["pred"]
q = f"Explain the following puzzle: {d[0]['question']}"

PAIRS = [("identical  (must be entailment)", gold, gold),
         ("unrelated  (must not be entailment)", gold, other),
         ("partial    (gold entails? partial entails?)", partial, gold)]

for label, a, b in PAIRS:
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(lambda _: ask_policy(BASE, KEY, MODEL,
                                               EQUIVALENCE_USER_PROMPT.format(question=q, text1=a, text2=b))[0],
                          range(N)))
    c = Counter(out)
    print("%-44s %s" % (label, dict(c)))
