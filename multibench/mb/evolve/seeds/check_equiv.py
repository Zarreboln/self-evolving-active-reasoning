"""CPU-only check that the two ported seeds send what the official runners send.

Both sides get the same scripted model replies (a fixed list, indexed by call number, so neither
side's state can steer them). The official side is the very code that produced our baselines
(mb/official/run_balrog.py `episode` -> BALROG's Evaluator/NaiveAgent/OpenAIWrapper;
mb/official/run_swiftsage.py `episode`) with `OpenAI` replaced by a recorder. Our side is the
seed's `run_episode`, driven through the mb adapter by a shim with child.py's Env interface, and a
`chat` recorder that builds the request body as mb/llm.py does. Compared per model call: the
messages and the sampling parameters; per episode: the actions the env executed and the score.

    # BabyAI: the official side needs BALROG's venv, ours mbench's
    $BALROG_PY mb/evolve/seeds/check_equiv.py babyai-official /tmp/bo.json
    python      mb/evolve/seeds/check_equiv.py babyai-seed     /tmp/bs.json
    python      mb/evolve/seeds/check_equiv.py babyai-compare  /tmp/bo.json /tmp/bs.json
    # ScienceWorld: both sides in the mbench venv. The official run's env is recorded and replayed
    # to the seed (ScienceWorld is not reproducible across JVMs, see env_determinism); the seed is
    # also run on the live adapter for comparison.
    python      mb/evolve/seeds/check_equiv.py sciworld
    python      mb/evolve/seeds/check_equiv.py static     # mb.evolve.check.problems on both seeds
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

BABYAI_CASES = [("GoToLocal", "1000000"), ("GoToLocal", "1000001"), ("GoToLocal", "1000002"),
                ("PutNextLocal", "1000000")]
# Raw replies, cycled by call index: valid, padded, punctuated, prose, numbered, digits only (filters
# to "" -> no history entry), prefixed.
BABYAI_REPLIES = ["turn left", "  go forward\n", "go forward.", "I will go forward", "1. turn right", "123",
                  "pick up", "toggle", "Action: turn right", "turn right", "go forward", "drop",
                  "turn left", "go forward", "go forward"]
BABYAI_ACTIONS = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]

SCIWORLD_CASES = [("boil", "test", 0), ("test-conductivity", "test", 0), ("find-living-thing", "test", 1),
                  ("chemistry-mix", "test", 0), ("use-thermometer", "selection", 0),
                  ("measure-melting-point-known-substance", "test", 2), ("grow-plant", "test", 0)]


def load_module(name, path, harness_lib=None):
    if harness_lib is not None:
        sys.modules["harness_lib"] = harness_lib
    else:
        sys.modules.pop("harness_lib", None)
        sys.path.insert(0, os.path.join(ROOT, "mb", "evolve", "ws_template"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def body_like_mb_llm(messages, temperature=0.7, max_tokens=None, top_p=None, stop=None):
    """The request body mb/llm.py's make_chat sends, minus model."""
    body = {"messages": copy.deepcopy(messages)}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = max_tokens
    if top_p is not None:
        body["top_p"] = top_p
    if stop:
        body["stop"] = stop
    return body


def apply_stop(text, stop):
    for s in stop or []:
        if s in text:
            text = text[:text.index(s)]
    return text


def norm_content(c):
    if isinstance(c, list):        # BALROG's [{"type": "text", "text": ...}]
        assert all(p["type"] == "text" for p in c), c
        return "\n".join(p["text"] for p in c)
    return c


def make_fake_openai(log, script):
    class FakeOpenAI:
        def __init__(self, *a, **k):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            log.append(copy.deepcopy(kw))
            text = apply_stop(script(len(log) - 1), kw.get("stop"))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                                            finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    return FakeOpenAI


class Shim:
    """child.py's Env over an mb adapter, in-process (JSON round trip included)."""

    def __init__(self, te, expose_info=False):
        self.te, self.expose_info = te, expose_info
        self.instructions = te.instructions
        self.observation = te.observation
        self.max_steps = te.max_steps
        self.reads_full_reply = te.reads_full_reply
        self.steps, self.done = 0, te.done
        self.adm = json.loads(json.dumps(te.admissible()))
        self.sync_info()

    def sync_info(self):
        if self.expose_info:       # the adapter change proposed for SwiftSage's findValidActionNew
            self.info = {"look": self.te._info["look"], "inv": self.te._info["inv"]}

    def admissible(self):
        return list(self.adm) if self.adm is not None else None

    def act(self, action):
        if self.done:
            raise RuntimeError("act after done")
        obs = self.te.act(str(action))
        self.observation, self.done, self.steps = obs, self.te.done, self.te.steps
        self.adm = json.loads(json.dumps(self.te.admissible()))
        self.sync_info()
        return obs


# ---------------------------------------------------------------- BabyAI

def babyai_official(out):
    import balrog.client as BC
    from balrog.environments.env_wrapper import EnvWrapper
    from mb.official.run_balrog import episode
    # run_balrog's main() imports mb.benches (instances); that import registers the minigrid envs in
    # the parent its forked workers inherit, so do the same here
    from mb.benches import get
    get("babyai")
    res = []
    for k, (level, seed) in enumerate(BABYAI_CASES):
        log, executed = [], []
        BC.OpenAI = make_fake_openai(log, lambda i: BABYAI_REPLIES[i % len(BABYAI_REPLIES)])
        orig_step = EnvWrapper.step

        def step(self, action, orig_step=orig_step, executed=executed):
            executed.append(action)
            return orig_step(self, action)
        EnvWrapper.step = step
        d = tempfile.mkdtemp()
        score, err = episode((d, "naive", level, seed, "http://fake/v1", "fake-model", k))
        EnvWrapper.step = orig_step
        rec = json.load(open(os.path.join(d, "naive", level, f"{seed}.json")))
        calls = []
        for kw in log:
            msgs = [{"role": m["role"], "content": norm_content(m["content"])} for m in kw["messages"]]
            params = {x: kw[x] for x in kw if x not in ("messages", "model")}
            calls.append({"messages": msgs, "params": params,
                          "content_is_parts": all(isinstance(m["content"], list) for m in kw["messages"])})
        res.append({"level": level, "seed": seed, "calls": calls, "executed": executed,
                    "score": score, "error": err, "steps": rec.get("steps")})
        print(f"official {level}/{seed}: {len(calls)} calls, score {score}, error {err}", flush=True)
    json.dump(res, open(out, "w"))


def babyai_seed(out):
    from mb.benches.babyai import make
    seed_mod = load_module("balrog_naive", os.path.join(HERE, "balrog_naive.py"))
    res = []
    for level, seed in BABYAI_CASES:
        calls = []

        def chat(messages, **kw):
            body = body_like_mb_llm(json.loads(json.dumps(messages)), **kw)
            calls.append({"messages": body.pop("messages"), "params": body})
            return BABYAI_REPLIES[(len(calls) - 1) % len(BABYAI_REPLIES)]
        te = make(level, seed)
        seed_mod.run_episode(Shim(te), chat)
        executed = [a if a in BABYAI_ACTIONS else te._env.default_action for a, _ in te.transcript]
        res.append({"level": level, "seed": seed, "calls": calls, "executed": executed,
                    "score": te.score(), "steps": te.steps})
        print(f"seed {level}/{seed}: {len(calls)} calls, score {te.score()}", flush=True)
    json.dump(res, open(out, "w"))


def babyai_compare(a, b):
    off, ours = json.load(open(a)), json.load(open(b))
    ok = True
    for o, s in zip(off, ours):
        tag = f"{o['level']}/{o['seed']}"
        n = min(len(o["calls"]), len(s["calls"]))
        first = next((i for i in range(n) if o["calls"][i]["messages"] != s["calls"][i]["messages"]), None)
        pdiff = sorted({json.dumps([c["params"] for c in (o["calls"][i], s["calls"][i])])
                        for i in range(n) if o["calls"][i]["params"] != s["calls"][i]["params"]})
        same = (first is None and len(o["calls"]) == len(s["calls"]) and not pdiff
                and o["executed"] == s["executed"] and o["score"] == s["score"])
        ok &= same
        maxlen = max(len(c["messages"]) for c in o["calls"])
        print(f"{tag}: calls {len(o['calls'])}/{len(s['calls'])}, max messages/call {maxlen}, "
              f"first differing call {first}, params differ {pdiff or 'no'}, "
              f"executed equal {o['executed'] == s['executed']} ({len(o['executed'])} steps), "
              f"score {o['score']}/{s['score']}, official content sent as text parts: "
              f"{all(c['content_is_parts'] for c in o['calls'])} -> {'IDENTICAL' if same else 'DIFFERENT'}")
        if first is not None:
            om, sm = o["calls"][first]["messages"], s["calls"][first]["messages"]
            for j in range(max(len(om), len(sm))):
                if j >= len(om) or j >= len(sm) or om[j] != sm[j]:
                    print("  message", j, "\n  official:", json.dumps(om[j] if j < len(om) else None)[:600],
                          "\n  seed:    ", json.dumps(sm[j] if j < len(sm) else None)[:600])
                    break
    print("BABYAI", "ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    return ok


# ---------------------------------------------------------------- ScienceWorld

def sciworld_script(gold):
    """Replies built from the gold path with variations; index = call number."""
    out = []
    for j, a in enumerate(gold):
        v = j % 7
        if v == 0:
            out.append(a)
        elif v == 1:
            out += ["think: next I should " + a, a]
        elif v == 2:
            out.append(a + " now please")            # not valid -> token-overlap match
        elif v == 3:
            out.append(a + "\nthis line is cut by stop")
        elif v == 4:
            out.append("   " + a + "  ")
        elif v == 5:
            out.append(a.replace("greenhouse", "green house"))
        else:
            out += [a, "look around", "look around"]   # repeats exercise the last-3 exclusion
    out += ["focus on air", "wait1", "wait1", "wait", "wait1", "wait"] * 40
    return out


def moves_probe():
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", envStepLimit=100)
    env.load("boil", 0, "easy")
    env.reset()
    rows = []
    for a in ["look around", "inventory", "task", "look at agent", "wait1", "open door to kitchen",
              "go to kitchen", "xyzzy", "focus on agent"]:
        env.step(a)
        rows.append((a, env.get_num_moves()))
    env.close()
    return rows


class Mismatch(Exception):
    pass


def slim(info):
    return {"valid": list(info["valid"]), "look": info["look"], "inv": info["inv"], "score": info["score"]}


def recording_env_class(rec):
    """ScienceWorldEnv that records what the official loop's env returned, for replay to the seed."""
    import scienceworld
    base = scienceworld.ScienceWorldEnv

    class RecordingEnv(base):
        def reset(self):
            rec.clear()
            obs, info = super().reset()
            rec.update(init={"desc": self.get_task_description(), "templates": self.get_possible_actions(),
                             "obs": obs, "info": slim(info)},
                       steps=[], live={}, live_valid_same_list=0, live_valid_same_set=0, live_valid_calls=0)
            return obs, info

        def step(self, a):
            o, r, d, info = super().step(a)
            if "init" in rec:
                rec["steps"].append({"action": a, "obs": o, "done": d, "info": slim(info)})
            return o, r, d, info

        def getValidActionObjectCombinations(self):
            live = super().getValidActionObjectCombinations()
            rec["live"][len(rec["steps"])] = list(live)     # what the official loop matched against
            last = (rec["steps"][-1]["info"] if rec.get("steps") else rec["init"]["info"])["valid"]
            rec["live_valid_calls"] += 1
            rec["live_valid_same_list"] += live == last
            rec["live_valid_same_set"] += set(live) == set(last)
            return live
    return base, RecordingEnv


class ReplayShim:
    """child.py's Env serving what the official run's env returned, in the adapter's format
    (mb/benches/sciworld.py: instructions layout, admissible = info['valid'], action.strip(), done at
    max_steps). An action other than the one the official loop sent ends the replay."""

    def __init__(self, rec, max_steps, expose_info):
        from mb.benches.sciworld import SciWorldEnv  # noqa: F401  (layout below is SciWorldEnv.__init__'s)
        init = rec["init"]
        self.instructions = (init["desc"] + "\n\nValid actions (replace OBJ with the name of an object you can see): "
                             + ", ".join(init["templates"]) + "\n\n" + init["obs"])
        self.observation, self.max_steps, self.reads_full_reply = "", max_steps, False
        self.steps, self.done = 0, False
        self.queue, self.executed, self.expose_info = list(rec["steps"]), [], expose_info
        self.live = rec["live"]
        self.set_info(init["info"])

    def set_info(self, info):
        self.cur, self.score = info, max(info["score"], 0) / 100
        if self.expose_info:
            self.info = {"look": info["look"], "inv": info["inv"]}

    def admissible(self):
        # ScienceWorld's valid combinations are not stable between two calls in the same state
        # (ambiguous referents are named differently), so serve the very list the official loop
        # matched against, where it made that call
        return list(self.live.get(self.steps, self.cur["valid"]))

    def act(self, action):
        a = str(action).strip()
        if self.done or not self.queue or self.queue[0]["action"].strip() != a:
            raise Mismatch(f"seed sent {a!r} at env step {self.steps}, official sent "
                           f"{self.queue[0]['action'] if self.queue else None!r}")
        r = self.queue.pop(0)
        self.steps += 1
        self.executed.append(a)
        self.done = r["done"] or self.steps >= self.max_steps
        self.observation = r["obs"]
        self.set_info(r["info"])
        return r["obs"]


def env_determinism(task, var, gold):
    """Two adapter envs of the same instance, stepped along the gold path: where do they differ?"""
    from mb.benches.sciworld import make
    a, b = make(task, var), make(task, var)
    out = {"instructions_equal": a.instructions == b.instructions}
    first_obs = first_list = first_set = None
    for k, act in enumerate([None] + gold[:60]):
        if act is not None:
            if a.done or b.done:
                break
            oa, ob = a.act(act), b.act(act)
            if first_obs is None and oa != ob:
                first_obs = k
        va, vb = a.admissible(), b.admissible()
        if first_list is None and va != vb:
            first_list = k
        if first_set is None and set(va) != set(vb):
            first_set = k
    a.close(), b.close()
    out.update(first_obs_diff_step=first_obs, first_valid_order_diff_step=first_list, first_valid_set_diff_step=first_set)
    return out


def official_vs_adapter_start(task, var):
    """Is the first observation the same when the env is built the official loop's way?"""
    from scienceworld import ScienceWorldEnv
    from mb.benches.sciworld import make
    te = make(task, var)
    obs_adapter = te.instructions.rsplit("\n\n", 1)[1]
    te.close()
    out = []
    for k in range(2):
        env = ScienceWorldEnv("", envStepLimit=100)
        env.getTaskNames()
        env.load(task, int(var), "easy", generateGoldPath=True)
        env.taskdescription()
        obs, _ = env.reset()
        out.append(obs == obs_adapter)
        env.close()
    return out


def sciworld():
    import tiktoken
    import openai
    import scienceworld
    import mb.official.run_swiftsage as RS
    from mb.benches.sciworld import MAX_STEPS_OVERRIDE, MAX_STEPS, instances, make
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    exact_lib = types.ModuleType("harness_lib")
    exact_lib.count_tokens = lambda s: len(enc.encode(s))
    exact = load_module("swiftsage_exact", os.path.join(HERE, "swiftsage_react.py"), exact_lib)
    approx = load_module("swiftsage_approx", os.path.join(HERE, "swiftsage_react.py"),
                         types.ModuleType("harness_lib"))
    assert exact.count_tokens is not None and approx.count_tokens is None

    # how often the focus filter (needs look/inventory) changes the chosen action, on the exact replay
    focus_stats = {"calls": 0, "action_differs": 0}
    orig_fva = exact.find_valid_action

    def fva(pred, combos, look_inv, recent, k=5):
        a = orig_fva(pred, combos, look_inv, recent, k)
        b = orig_fva(pred, combos, None, recent, k)
        focus_stats["calls"] += 1
        focus_stats["action_differs"] += a != b
        return a

    ok = True
    tok = []
    print("moves counted by ScienceWorld after each action:", moves_probe(), flush=True)
    for task, split, idx in SCIWORLD_CASES:
        var = instances(task, split)[idx]
        g = make(task, var)
        gold = g._oracle_plan()
        g.close()
        script = sciworld_script(gold)
        reply = lambda i: script[i] if i < len(script) else "wait1"   # noqa: E731

        # official: run_swiftsage.episode, OpenAI replaced by a recorder, its env recorded
        log, rec = [], {}
        openai.OpenAI = make_fake_openai(log, reply)
        base, recording = recording_env_class(rec)
        scienceworld.ScienceWorldEnv = recording
        d = tempfile.mkdtemp()
        try:
            score, err = RS.episode((d, "react", task, var, "http://fake/v1", "fake-model"))
        finally:
            scienceworld.ScienceWorldEnv = base
        assert err is None, err
        off_prompts = [kw["messages"] for kw in log]
        off_params = {x: v for x, v in log[0].items() if x not in ("messages", "model")}
        off_env_actions = [s["action"] for s in rec["steps"]]
        for m in off_prompts:
            t = m[0]["content"]
            tok.append((t, len(enc.encode(t))))

        def play(mod, shim):
            calls = []

            def chat(messages, **kw):
                body = body_like_mb_llm(json.loads(json.dumps(messages)), **kw)
                calls.append(body)
                return apply_stop(reply(len(calls) - 1), kw.get("stop"))
            mismatch = None
            try:
                mod.run_episode(shim, chat)
            except Mismatch as e:
                mismatch = str(e)
            return calls, mismatch

        max_steps = MAX_STEPS_OVERRIDE.get(task, MAX_STEPS)
        print(f"{task}/{var} (official: {len(off_prompts)} calls, {len(off_env_actions)} env actions, score {score}; "
              f"official-style build gives the adapter's first obs: {official_vs_adapter_start(task, var)}; "
              f"live getValidActionObjectCombinations == last info['valid'] as list "
              f"{rec['live_valid_same_list']}/{rec['live_valid_calls']}, as set {rec['live_valid_same_set']}/{rec['live_valid_calls']})",
              flush=True)
        for name, mod, expose in [("replay exact", exact, True), ("replay no-look", exact, False),
                                  ("replay approx-tok", approx, True), ("replay as-shipped", approx, False)]:
            if name == "replay exact":
                exact.find_valid_action = fva
            shim = ReplayShim(rec, max_steps, expose)
            calls, mismatch = play(mod, shim)
            exact.find_valid_action = orig_fva
            prompts = [c["messages"] for c in calls]
            params = {x: v for x, v in calls[0].items() if x != "messages"}
            same = prompts == off_prompts and shim.executed == [a.strip() for a in off_env_actions] and not mismatch
            if name == "replay exact":
                ok &= same and abs(shim.score - score) < 1e-9
                print(f"   official-only params {({k: v for k, v in off_params.items() if params.get(k) != v})}, "
                      f"seed params {params}")
            first = next((i for i in range(min(len(prompts), len(off_prompts))) if prompts[i] != off_prompts[i]), None)
            print(f"   {name}: calls {len(prompts)}, prompts identical {prompts == off_prompts} (first differing call "
                  f"{first}), env actions identical {shim.executed == [a.strip() for a in off_env_actions]}, "
                  f"score {shim.score} vs {score}{', ' + mismatch if mismatch else ''} -> {'IDENTICAL' if same else 'DIFFERENT'}",
                  flush=True)
        # the same seed on the live adapter (a second ScienceWorld JVM)
        te = make(task, var)
        calls, _ = play(exact, Shim(te, expose_info=True))
        te.close()
        prompts = [c["messages"] for c in calls]
        first = next((i for i in range(min(len(prompts), len(off_prompts))) if prompts[i] != off_prompts[i]), None)
        print(f"   live adapter, exact: prompts identical {prompts == off_prompts} (first differing call {first}), "
              f"env actions identical {[a for a, _ in te.transcript] == [a.strip() for a in off_env_actions]}, score {te.score()}; "
              f"two adapter envs of this instance: {env_determinism(task, var, gold)}", flush=True)
    print("focus-filter effect on the exact replay:", focus_stats)
    for lr in (7, 8, 9, 10, 11, 12):
        approx.LONG_RUN = lr
        rel = [(approx.approx_tokens(t) - n) / n for t, n in tok]
        over = sum(1 for t, n in tok if (approx.approx_tokens(t) > 4036) != (n > 4036))
        print(f"approx_tokens (LONG_RUN={lr}) vs tiktoken over {len(rel)} official prompts: mean rel err "
              f"{sum(rel) / len(rel):+.4f}, min {min(rel):+.4f}, max {max(rel):+.4f}; truncation decision differs on {over}")
    print("SCIWORLD", "ALL IDENTICAL (replay, exact variant)" if ok else "DIFFERENCES FOUND")
    return ok


def static_check():
    from mb.evolve.check import problems
    for f in ("balrog_naive", "swiftsage_react"):
        src = open(os.path.join(HERE, f + ".py")).read()
        print(f"problems({f}.py) = {problems(src)}  ({len(src.encode())} bytes)")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "babyai-official":
        babyai_official(sys.argv[2])
    elif mode == "babyai-seed":
        babyai_seed(sys.argv[2])
    elif mode == "babyai-compare":
        sys.exit(0 if babyai_compare(sys.argv[2], sys.argv[3]) else 1)
    elif mode == "sciworld":
        sys.exit(0 if sciworld() else 1)
    elif mode == "static":
        static_check()
