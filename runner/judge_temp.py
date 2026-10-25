"""Does temperature 0 stabilise the equivalence judge, or is the instability intrinsic?"""
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/tmp_l/code")
import sp_semantic_equivalence as S

BASE = os.environ["POLICY_BASE_URL"]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
d = json.load(open("/home/tmp_l/results/sp_equiv_selftest.json"))
gold, other = d[0]["answer"], d[1]["pred"]
q = "Explain the following puzzle: " + d[0]["question"]


def ask(temp, prompt):
    import urllib.request
    body = {"model": MODEL, "temperature": temp, "top_p": 1.0, "max_tokens": 512,
            "messages": [{"role": "system", "content": S.EQUIVALENCE_SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "EvaluationScheme", "schema": S.SCHEMA}}}
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer x", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            p = json.loads(r["choices"][0]["message"]["content"])
            if p.get("evaluation") in S.EQUIV_MAP:
                return p["evaluation"]
        except Exception:
            pass
    return "(failed)"


for temp in (0.0, 0.1):
    for label, a, b in (("identical", gold, gold), ("unrelated", gold, other)):
        prompt = S.EQUIVALENCE_USER_PROMPT.format(question=q, text1=a, text2=b)
        with ThreadPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(lambda _: ask(temp, prompt), range(10)))
        print("  temp=%-4s %-10s %s" % (temp, label, dict(Counter(out))))
