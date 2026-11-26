"""Fakes for CPU-only tests: a word-guessing environment played by the real generated harness, a fake
policy model, and fake Reflector/Curator replies. Nothing here touches a network or a credential.

FakeBackend.play runs the generated harness file in-process against FakeEnv (as mb/evolve/child.py would
run it against the broker), checks it with mb/evolve/check.py first, and renders the episode with the
broker's own trace renderer. A lost game names its secret in the last observation, as Wordle does, and
the instructions name the game, so a careless Curator would copy both into the playbook.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

from mb.evolve.check import problems
from mb.evolve.broker import _render

LIB = Path(__file__).resolve().parents[2] / "evolve" / "ws_template"


class Preempted(Exception):
    pass


class FakeEnv:
    def __init__(self, secret, retry):
        self.secret, self.retry = secret, retry
        self.instructions = ("You are playing Wordle. Guess the secret 5-letter word in [brackets]. "
                             "You have 3 guesses.")
        self.observation = ""
        self.max_steps, self.steps, self.done = 3, 0, False
        self.reads_full_reply = True
        h = int(hashlib.sha256(secret.encode()).hexdigest(), 16)
        self.win_at = 2 if (h % 3 == 0 or retry) else None

    def admissible(self):
        return None

    def act(self, action):
        if self.done:
            raise RuntimeError("over")
        self.steps += 1
        if self.win_at == self.steps:
            self.done, self.success = True, True
            self.observation = "Congratulations! You guessed the word."
        elif self.steps >= self.max_steps:
            self.done, self.success = True, False
            self.observation = f"You ran out of guesses. The secret word was {self.secret}."
        else:
            self.observation = f"Feedback on {action[:40]!r}: X X G Y X"
        return self.observation


class FakeBackend:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def play(self, key, harness_path, cell, split, instance_ids, visible_only):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise Preempted(f"simulated preemption before {key}")
        src = Path(harness_path).read_text()
        bad = problems(src)
        assert not bad, bad
        self.calls.append({"key": key, "cell": cell, "split": split, "instances": list(instance_ids),
                           "visible_only": visible_only})
        out = []
        for inst in instance_ids:
            e = self._episode(harness_path, inst, retry="**Reflection**" in src)
            trace = _render(e)
            if visible_only:                      # as mb/evolve/worker.py publishes test traces
                lines = trace.splitlines()
                trace = "\n".join(l for k, l in enumerate(lines) if not (k < 4 and l.startswith("score "))) + "\n"
            out.append({"instance": inst, "trace": trace,
                        "score": None if visible_only else e["score"],
                        "success": None if visible_only else e["success"]})
        return out

    def _episode(self, harness_path, inst, retry):
        if str(LIB) not in sys.path:
            sys.path.insert(0, str(LIB))
        spec = importlib.util.spec_from_file_location("fake_candidate", harness_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env, calls, steps = FakeEnv(inst, retry), [], []

        def chat(messages, temperature=0.7, max_tokens=8192, top_p=None, stop=None):
            ids = re.findall(r"\[([a-z]{3,}-\d{5})\]", messages[0]["content"])
            reply = (f"Following [{ids[0]}] I guess. " if ids else "I guess. ") + "[crane]"
            calls.append({"step": env.steps, "messages": messages, "reply": reply})
            return reply

        real_act = env.act

        def act(a):
            o = real_act(a)
            steps.append({"action": a, "observation": o})
            return o
        env.act = act
        mod.run_episode(env, chat)
        s = float(getattr(env, "success", False))
        return {"instance_id": inst, "score": s, "success": bool(s), "steps": env.steps, "end": "end",
                "error": None, "calls": calls, "steps_log": steps}


class FakeLLM:
    """Reflector: blames the guess, repeats the revealed secret, tags every bullet it was shown helpful.
    Curator: proposes a generic bullet, one naming the game, one holding the revealed secret, and (on
    every 7th call) a malformed reply."""

    def __init__(self):
        self.calls = []

    def __call__(self, role, call_id, prompt):
        self.calls.append({"role": role, "call_id": call_id, "prompt": prompt})
        if role == "reflector":
            secret = re.findall(r"The secret word was (\w+)", prompt)
            shown = prompt.split("**Part of Playbook that's used by the generator to answer the question:**")[1]
            ids = sorted(set(re.findall(r"\[([a-z]{3,}-\d{5})\]", shown)))
            return json.dumps({"reasoning": "r", "error_identification": "guessed blindly",
                               "root_cause_analysis": "ignored feedback",
                               "correct_approach": "use the letter feedback",
                               "key_insight": "the answer was " + (secret[-1] if secret else "unknown"),
                               "bullet_tags": [{"id": i, "tag": "helpful"} for i in ids]})
        n = sum(c["role"] == "curator" for c in self.calls)
        if n % 7 == 0:
            return "I cannot produce JSON right now."
        refl = prompt.split("**Recent Reflection:**")[1].split("**Current Playbook:**")[0]
        secret = re.findall(r"the answer was (\w+)", refl)
        ops = [{"type": "ADD", "section": "common_mistakes_to_avoid",
                "content": f"Use every piece of letter feedback before guessing again (note {n})."},
               {"type": "ADD", "section": "strategies_and_insights",
                "content": "In Wordle and Mastermind, eliminate letters greedily."}]
        if secret and secret[0] != "unknown":
            ops.append({"type": "ADD", "section": "others", "content": f"Remember the word {secret[0]}."})
        return json.dumps({"reasoning": "add", "operations": ops})
