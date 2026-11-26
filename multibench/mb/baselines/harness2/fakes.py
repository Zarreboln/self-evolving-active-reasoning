"""CPU-only stand-ins for tests: a small word-guessing environment, a deterministic "policy model",
a play() that runs candidates through the real child sandbox (mb/evolve/child.py after
mb/evolve/check.py, as the broker does), and a deterministic agent for every Harness² session role.
No network, no GPU, no paid call.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from mb.evolve.check import problems as static_problems

EVOLVE = Path(__file__).resolve().parents[2] / "evolve"
CHILD, LIB = EVOLVE / "child.py", EVOLVE / "ws_template"
GAME = "NumberQuest"                              # the fake game's name; the names check must keep it out
WORDS = ["apple", "brick", "cloud", "dream", "eagle", "flame", "grape", "house", "ivory", "joker", "knife", "lemon"]


class Interrupted(BaseException):
    """A simulated kill (preemption, ctrl-C); BaseException so no `except Exception` swallows it."""


class FakeEnv:
    """Guess a hidden word from a listed set; feedback marks letters in the right place."""

    def __init__(self, instance):
        self.secret, self.steps, self.done, self.max_steps, self.success = instance, 0, False, 6, False
        self.instructions = (f"Welcome to {GAME}! A hidden word is one of: {', '.join(WORDS)}. Guess it by replying "
                             f"with one word in square brackets, e.g. [{WORDS[0]}]. After each guess you are told "
                             f"which letters are in the right place. You have {self.max_steps} guesses.")
        self.observation = ""
        self.history = self.instructions
        self.reads_full_reply = True

    def admissible(self):
        return None

    def act(self, action):
        self.steps += 1
        m = re.search(r"\[([a-z]+)\]", action.lower())
        if not m:
            obs = "Invalid move: reply with one word in square brackets."
        else:
            g = m.group(1)
            if g == self.secret:
                obs, self.done, self.success = "Congratulations! You found the hidden word.", True, True
            else:
                mark = " ".join(c if i < len(self.secret) and self.secret[i] == c else "_" for i, c in enumerate(g))
                obs = f"Feedback for [{g}]: {mark}"
        if self.steps >= self.max_steps and not self.done:
            self.done, obs = True, obs + "\nOut of guesses. The game is over."
        self.history += "\n" + obs
        self.observation = self.history          # TextArena-style full-history observation
        return self.observation


def fake_policy(messages):
    """A deterministic stand-in for the frozen model, sensitive to what the harness tells it."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("# skill "):
        system += "\n" + messages[-1]["content"]
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    if "TOOL: skill" in system and not any(m["role"] == "assistant" for m in messages):
        skill = re.search(r"- skill `([a-z0-9_\-]+)`", system)
        if skill:
            return f"Let me check the procedure.\nTOOL: skill {skill.group(1)}"
    guessed = re.findall(r"Feedback for \[([a-z]+)\]: ([a-z_ ]+)", user)
    cands = [w for w in WORDS if w not in {g for g, _ in guessed}]
    if "consistent" in system.lower():
        for g, mark in guessed:
            pat = mark.split(" ")
            cands = [w for w in cands if all(p == "_" or (i < len(w) and w[i] == p) for i, p in enumerate(pat))]
    w = cands[0] if cands else WORDS[0]
    return f"[{w}]" if "bracket" in system.lower() else f"My guess is {w}."


class FakePlay:
    """play() through the real child sandbox, with FakeEnv and fake_policy in place of the broker's
    environment and vLLM server. Rejects a harness that fails mb/evolve/check.py, as the broker does."""

    def __init__(self, crash_after: int | None = None):
        self.calls, self.log, self.crash_after, self.killed = 0, [], crash_after, 0
        self.lock = threading.Lock()

    def __call__(self, harness_path, cell, split, instance_ids, *, key):
        src = Path(harness_path).read_text()
        bad = static_problems(src)
        if bad:
            raise AssertionError(f"an invalid harness reached play(): {bad}")
        out = []
        for inst in instance_ids:
            with self.lock:
                self.calls += 1
                self.log.append((key, cell, inst))
                if self.crash_after is not None and self.calls > self.crash_after:
                    self.killed += 1
                    self.log.pop()
                    raise Interrupted(f"simulated kill before episode {self.calls}")
            out.append(self.episode(harness_path, cell, inst))
        return out

    def episode(self, harness_path, cell, inst):
        env = FakeEnv(inst)
        work = tempfile.mkdtemp(prefix="h2child_")
        child = subprocess.Popen([sys.executable, "-I", str(CHILD), str(harness_path), str(LIB)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                 cwd=work, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})

        def send(m):
            child.stdin.write(json.dumps(m) + "\n")
            child.stdin.flush()

        calls, steps, end, error = [], [], None, None
        send({"instructions": env.instructions, "observation": env.observation, "max_steps": env.max_steps,
              "reads_full_reply": True, "admissible": None})
        while True:
            line = child.stdout.readline()
            if not line:
                end, error = "crash", f"harness process exited ({child.wait()})"
                break
            m = json.loads(line)
            if m["op"] == "chat":
                if len(calls) >= 8 * env.max_steps:
                    send({"error": "model-call budget used up"})
                    continue
                reply = fake_policy(m["messages"])
                calls.append({"step": env.steps, "messages": m["messages"], "reply": reply})
                send({"reply": reply})
            elif m["op"] == "act":
                if env.done:
                    send({"error": "the episode is over"})
                    continue
                obs = env.act(m["action"])
                steps.append({"action": m["action"], "observation": obs, "source": "model"})
                send({"obs": obs, "done": env.done, "steps": env.steps, "admissible": None})
            else:
                end, error = m["op"], m.get("error")
                break
        child.kill()
        err_tail = child.stderr.read()[-2000:]
        return {"bench": cell.split(":")[0], "subtask": cell.split(":")[1], "instance": inst,
                "instance_id": f"{cell}/{inst}", "steps": env.steps, "done": env.done,
                "score": float(env.success), "success": env.success, "steps_log": steps, "calls": calls,
                "end": end, "error": error, "harness_stderr": err_tail}


INV_GUARD = "Always reply with exactly one guess in square brackets, like [word]."
DOM_LINE = ("Only guess words consistent with all feedback so far; when several remain, load the "
            "`narrow_candidates` skill.")
SKILL = ("---\nname: narrow_candidates\ndescription: several candidate words remain\n---\n"
         "List the candidates, drop every one that contradicts a feedback line, guess the first left.\n")
PLUGIN = ('"""Wrap a bare word in brackets."""\nimport re\n\n\ndef on_action(action, ctx):\n'
          '    if re.search(r"\\[[a-z]+\\]", action.lower()):\n        return None\n'
          '    words = re.findall(r"[a-z]+", action.lower())\n    return "[" + words[-1] + "]" if words else None\n')
SCRIPT = ('"""Letters fixed in place so far, from the feedback lines."""\nimport re\n\n\ndef main(text):\n'
          '    marks = re.findall(r"Feedback for \\[[a-z]+\\]: ([a-z_ ]+)", text)\n    return "; ".join(marks) or "none"\n')


class FakeAgent:
    """Raw session runner (role, label, ws, mission) -> (text, usage, error), deterministic.

    It edits `ws/context/harness/` like an editor would, merges the pool like a composer, picks the
    first id like the selector, and writes an experience entry like the reflector. Two scripted
    misbehaviours exercise the shared-store rule: the first DOM attempt of the first dev problem writes
    the game's name and answers into memory (must be rejected and retried), and the reflector of the
    second dev problem names the game in its lesson (must be kept out of the library).
    """

    def __init__(self, crash_after: int | None = None, answers_to_leak=()):
        self.calls, self.crash_after, self.log, self.killed = 0, crash_after, [], 0
        self.answers_to_leak = list(answers_to_leak)
        self.lock = threading.Lock()

    def __call__(self, role, label, ws: Path, mission: str):
        with self.lock:
            self.calls += 1
            self.log.append(label)
            if self.crash_after is not None and self.calls > self.crash_after:
                self.killed += 1
                raise Interrupted(f"simulated kill in session {label}")
        usage = {"sessions": 1, "input_tokens": len(mission) // 4, "output_tokens": 150}
        h = ws / "context" / "harness"
        if role == "editor":
            if "/inv" in label:
                (h / "guardrails.md").write_text(((h / "guardrails.md").read_text().rstrip() + "\n" + INV_GUARD).strip())
                (h / "plugin").mkdir(exist_ok=True)
                (h / "plugin" / "brackets.py").write_text(PLUGIN)
                (h / "scripts").mkdir(exist_ok=True)
                (h / "scripts" / "fixed_letters.py").write_text(SCRIPT)
                return "<notes>\n- rollout base1 action 1: the reply had no brackets -> guardrail + plugin\n</notes>", usage, ""
            if label.startswith("dev-0001/dom1.a1"):
                (h / "LongTermMEMORY.md").write_text(f"In {GAME} the answers seen so far: "
                                                     + ", ".join(self.answers_to_leak))
                return "<notes>\n- memorised the answers\n</notes>", usage, ""
            (h / "systemprompt.md").write_text((h / "systemprompt.md").read_text().rstrip() + "\n" + DOM_LINE)
            (h / "skills" / "narrow_candidates").mkdir(parents=True, exist_ok=True)
            (h / "skills" / "narrow_candidates" / "SKILL.md").write_text(SKILL)
            if "REJECTED" in mission:
                (h / "LongTermMEMORY.md").write_text("")
            return "<notes>\n- base1 guessed a word ruled out by feedback -> consistency rule + skill\n</notes>", usage, ""
        if role == "composer":
            for cand in sorted((ws / "context" / "pool").glob("*/harness")):
                for p in sorted(cand.rglob("*")):
                    if not p.is_file():
                        continue
                    t = h / p.relative_to(cand)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    if t.suffix == ".md" and t.exists():
                        lines = t.read_text().splitlines()
                        lines += [l for l in p.read_text().splitlines() if l not in lines]
                        t.write_text("\n".join(lines))
                    else:
                        t.write_text(p.read_text())
            return "<notes>\nkept -> inv guardrail -> base1 action 1\nkept -> dom skill -> base1 action 2\n</notes>", usage, ""
        if role == "selector":
            ids = re.search(r"Choose exactly one of: ([A-Z, ]+)\.", mission).group(1).split(", ")
            return f"<why>\n{ids[0]} reached the goal in fewer turns.\n</why>\n\n```choice\n{ids[0]}\n```", usage, ""
        if role == "reflector":
            lesson = ("when feedback marks fixed letters, add a skill that filters candidates (skills)"
                      if "dev-0002" not in label else f"in {GAME}, filter candidates (skills)")
            return ("<edits>\ncomponent: guardrails.md\ncandidates: inv1\nchange: bracket rule\n"
                    "observed: base1 invalid moves vs inv1 valid\nverdict: helped\n</edits>\n\n"
                    "<experience>\ncue: the reply must follow a fixed syntax and the model drops it\n"
                    f"lesson: {lesson}\nevidence: guardrails.md row, base1 vs inv1\nconfidence: medium\n"
                    "</experience>"), usage, ""
        raise ValueError(role)
