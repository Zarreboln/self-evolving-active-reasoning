"""Evaluate one harness on one split and write the result Meta-Harness's loop reads.

Token accounting is done by intercepting `arbench.utils.inference.inference` rather than by
counting inside `SPHarness.llm`. A seed that reproduces a published method has to be allowed to
call AR-Bench's own helpers -- some of those helpers are a dozen functions deep and
reimplementing it on top of `self.llm` would be reproducing it in name only -- and those helpers
talk to the endpoint directly. Counting at the interception point catches every path, so the cost
axis means the same thing for a one-line seed and for a candidate that calls the package.

Output, one file per (task, harness, model), in the shape `load_results` expects:

    logs/sp/<harness>/<model>/val.json
        accuracy                 non-strict semantic equivalence, 0-1   <- the selection signal
        memory_context_tokens    mean input tokens per episode          <- the cost axis
        strict, partial_share    sentinels; see the skill
        llm_calls, runtime_seconds, seed

    python benchmark.py --harness baseline_zero_shot --split selection
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
from arbench_mh.sp.env import SPEnv  # noqa: E402

MODEL = os.environ.get("POLICY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
POLICY = os.environ["POLICY_BASE_URL"].rstrip("/")
REFEREE = os.environ.get("RESPONSE_BASE_URL", "http://127.0.0.1:8099/v1").rstrip("/")
NO_THINK = os.environ.get("DISABLE_THINKING", "0") == "1"


class Counter:
    """Every call to the endpoint, whoever makes it. One per episode, chained to a run total.

    A single shared counter with a delta around each episode double-counts under sixteen threads:
    overlapping intervals bill each episode for whatever else was in flight, measured at about
    2.2x. The run totals were right; the per-episode numbers the cost axis averages were not.
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


def _post(base, model, messages, temperature, top_p, max_tokens, counter):
    kw = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p}
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if NO_THINK:
        kw["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(kw).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer x"}, method="POST")
    last = None
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                o = json.loads(r.read())
            u = o.get("usage") or {}
            counter.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            return (o["choices"][0]["message"]["content"] or "")
        except Exception as e:                       # AR-Bench's own loop has no retry; one
            last = e                                 # transient 5xx killed a 16-shard run earlier
            time.sleep(1.5 * (a + 1))
    raise RuntimeError("endpoint failed after 4 attempts: %s" % last)


def install_interceptor(counter):
    """Make AR-Bench's `inference` route here and be counted, keeping its return shape."""
    class _Msg:
        def __init__(self, c): self.content = c

    class _Choice:
        def __init__(self, c): self.message = _Msg(c)

    class _Usage:
        def __init__(self): self.prompt_tokens = self.completion_tokens = 0

    class _Resp:
        def __init__(self, c): self.choices = [_Choice(c)]; self.usage = _Usage()

    def patched(messages, model=None, json_format=False, api_key=None, base_url=None,
                temperature=None, top_p=None):
        base = POLICY if (base_url or POLICY).rstrip("/") != REFEREE else REFEREE
        return _Resp(_post(base, model or MODEL, messages,
                           0.7 if temperature is None else temperature,
                           0.7 if top_p is None else top_p, None, counter))

    ar_inf.inference = patched
    for mod in list(sys.modules):
        m = sys.modules[mod]
        if mod.startswith("arbench.") and getattr(m, "inference", None) is not None:
            m.inference = patched


def run_one(item, harness_cls, total):
    counter = Counter(parent=total)                  # this episode's own tally
    policy = lambda msgs, temperature=0.7, max_tokens=None: _post(
        POLICY, MODEL, msgs, temperature, 0.7, max_tokens, counter)
    referee = lambda msgs, temperature=0.7, max_tokens=None: _post(
        REFEREE, "jev-latest", msgs, temperature, 0.7, max_tokens, counter)
    env = SPEnv(item, referee, method=getattr(harness_cls, "METHOD", "zero_shot"))
    try:
        pred = harness_cls(env, policy).solve()
    except Exception as e:                           # a candidate that crashes scores zero, and
        pred = ""                                    # the loop keeps going
        print("    episode failed: %s: %s" % (type(e).__name__, str(e)[:120]), flush=True)
    r = env.score(pred)
    r["input_token"] = counter.in_tok
    r["output_token"] = counter.out_tok
    r["total_token"] = counter.in_tok + counter.out_tok
    r["llm_calls"] = counter.calls
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--split", default="selection")
    ap.add_argument("--logs-dir", default=os.path.expanduser("~/results/mh_sp/logs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="write the episodes here instead of the derived path; the loop's "
                         "evaluate() passes it so the file lands where score_val.py expects")
    a = ap.parse_args()

    # test data lives outside the proposer's working directory; see benchmark.py
    root = "~/sealed" if a.split == "test" else "~/arbench_mh_splits"
    split = json.load(open(os.path.expanduser("%s/sp/%s.json" % (root, a.split))))
    items = split["instances"][:a.limit] if a.limit else split["instances"]
    mod = importlib.import_module("arbench_mh.sp.agents.%s" % a.harness)
    cls = getattr(mod, "Harness")

    counter = Counter()
    install_interceptor(counter)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(lambda it: run_one(it, cls, counter), items))
    runtime = time.time() - t0

    out = a.out or os.path.join(a.logs_dir, "sp", a.harness, MODEL.split("/")[-1],
                                "episodes.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("%s / %s: %d episodes, %.0fs, %d calls, %d input tokens"
          % (a.harness, a.split, len(rows), runtime, counter.calls, counter.in_tok), flush=True)
    print("  -> %s   (score with score_val.py)" % out)   # the real path, not the derived one


if __name__ == "__main__":
    main()
