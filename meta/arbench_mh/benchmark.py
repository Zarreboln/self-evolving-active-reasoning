"""Evaluate one candidate harness on one split, for gn or dc, and write what the loop reads.

Shared across tasks because the shape is the same: build the task's env, hand it and a frozen-model
callable to the candidate, let `solve()` play the episode, score with the env's own scorer. Only the
env and the return type differ.

Token accounting intercepts `arbench.utils.inference` rather than counting inside `llm()`, so a
candidate that calls AR-Bench's own helpers is costed the same way as one that only calls
`self.llm`.

The cost axis is INPUT PLUS OUTPUT. Input alone hides where the cost of these methods actually
sits: a harness that has the model write long intermediate state out on every turn can read
modestly and generate enormously, and generation is the slow side. Measured on this benchmark, the
gap between the two accountings reaches a factor of four on a single method.

    POLICY_BASE_URL=... python benchmark.py --task gn --harness template_zero_shot --split selection
"""
from __future__ import annotations

import argparse, importlib, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

# Resolve `arbench_mh` from this file, not from a fixed directory: a second copy of the
# package elsewhere on the path would otherwise be imported instead of the one being run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "sp"
                else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

import arbench.utils.inference as ar_inf  # noqa: E402

MODEL = os.environ.get("POLICY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
POLICY = os.environ["POLICY_BASE_URL"].rstrip("/")
NO_THINK = os.environ.get("DISABLE_THINKING", "0") == "1"


class Counter:
    """One per episode, not one shared.

    The first version took a delta of a single shared counter around each episode. With sixteen
    threads the intervals overlap, so every episode billed part of whatever else was in flight --
    measured at 1517 tokens for an episode that really used 682, about 2.2x. The totals were
    right; the per-episode numbers, which are what the cost axis averages, were not.
    """

    def __init__(self, parent=None):
        self.calls = self.in_tok = self.out_tok = 0
        self.parent = parent

    def add(self, i, o):
        self.calls += 1
        self.in_tok += i
        self.out_tok += o
        if self.parent is not None:
            self.parent.add(i, o)


def post(messages, temperature, top_p, max_tokens, counter, model=None):
    kw = {"model": model or MODEL, "messages": messages,
          "temperature": temperature, "top_p": top_p}
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if NO_THINK:
        kw["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(POLICY + "/chat/completions", data=json.dumps(kw).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer x"}, method="POST")
    last = None
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                o = json.loads(r.read())
            u = o.get("usage") or {}
            counter.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            return o["choices"][0]["message"]["content"] or ""
        except Exception as e:                       # AR-Bench's own loop has no retry and one
            last = e                                 # transient failure killed a 16-shard run
            time.sleep(1.5 * (a + 1))
    raise RuntimeError("endpoint failed after 4 attempts: %s" % last)


def install_interceptor(counter):
    class _M:
        def __init__(self, c): self.content = c

    class _C:
        def __init__(self, c): self.message = _M(c)

    class _U:
        def __init__(self): self.prompt_tokens = self.completion_tokens = 0

    class _R:
        def __init__(self, c): self.choices = [_C(c)]; self.usage = _U()

    def patched(messages, model=None, json_format=False, api_key=None, base_url=None,
                temperature=None, top_p=None):
        return _R(post(messages, 0.7 if temperature is None else temperature,
                       0.7 if top_p is None else top_p, None, counter, model))

    ar_inf.inference = patched
    for name in list(sys.modules):
        m = sys.modules[name]
        if name.startswith("arbench.") and getattr(m, "inference", None) is not None:
            m.inference = patched


def make_env(task, item, counter):
    llm = lambda msgs, temperature=0.7, max_tokens=None: post(   # noqa: E731
        msgs, temperature, 0.7, max_tokens, counter)
    if task == "gn":
        from arbench_mh.gn.env import GNEnv
        return GNEnv(item), llm
    from arbench_mh.dc.env import DCEnv
    return DCEnv(item, llm), llm


def run_one(task, item, cls, total):
    counter = Counter(parent=total)                  # this episode's own tally
    env, llm = make_env(task, item, counter)
    try:
        out = cls(env, llm).solve()
    except Exception as e:                           # a candidate that crashes scores zero and
        out = None                                   # the loop keeps going
        print("    episode failed: %s: %s" % (type(e).__name__, str(e)[:120]), flush=True)
    r = env.score(out)
    r["input_token"] = counter.in_tok
    r["output_token"] = counter.out_tok
    r["total_token"] = counter.in_tok + counter.out_tok
    r["llm_calls"] = counter.calls
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("gn", "dc"))
    ap.add_argument("--harness", required=True)
    ap.add_argument("--split", default="selection")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # The held-out split lives outside the directory the proposer works in. Upstream keeps its
    # test results in a separate tree and relies on the skill saying so; this goes one step
    # further and moves the test DATA too, because a proposer that lists the splits directory
    # will read whatever is in it -- one did, on its eighth command, before this was changed.
    root = "~/sealed" if a.split == "test" else "~/arbench_mh_splits"
    split = json.load(open(os.path.expanduser(
        "%s/%s/%s.json" % (root, a.task, a.split))))
    items = split["instances"][:a.limit] if a.limit else split["instances"]
    cls = getattr(importlib.import_module("arbench_mh.%s.agents.%s" % (a.task, a.harness)),
                  "Harness")

    counter = Counter()
    install_interceptor(counter)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(lambda it: run_one(a.task, it, cls, counter), items))
    acc = 100.0 * sum(1 for r in rows if r.get("correctness")) / len(rows)

    out = a.out or os.path.expanduser(
        "~/results/mh_runs/%s_%s_%s.json" % (a.task, a.harness, a.split))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("%s / %-22s %s  n=%d  acc=%.1f%%  %.0fs  %d calls  %d in-tok  -> %s"
          % (a.task, a.harness, a.split, len(rows), acc, time.time() - t0,
             counter.calls, counter.in_tok, out), flush=True)


if __name__ == "__main__":
    main()
