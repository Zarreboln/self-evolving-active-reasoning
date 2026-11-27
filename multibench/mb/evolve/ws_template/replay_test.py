"""Crash test of a harness on this game's real observations: replays recorded episodes (evidence/*/episodes/
*/trace.md) to the harness. Whatever the harness does, the game answers with the recorded observations in
order and ends when they run out; the "model" answers with the recorded replies in order (then repeats the
last one). It checks that the harness runs on real inputs without crashing, not how well it plays.

    python replay_test.py harness.py                    # every trace under evidence/ (at most 5)
    python replay_test.py harness.py path/to/trace.md   # one trace
Exit status 0 if the harness ran on every trace, 1 otherwise (the traceback is printed).
"""
import glob
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class EpisodeOver(Exception):
    pass


def parse(path):
    """(first user message, observations after each action, model replies) from a trace.md."""
    first_user, observations, replies = None, [], []
    lines = open(path, encoding="utf-8").read().splitlines()
    i, label = 0, None
    while i < len(lines):
        line = lines[i]
        if line.startswith("**") and line.endswith(":**"):
            label = line.strip("*:").strip()
        elif line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            block = "\n".join(lines[i + 1:j])
            if label == "user" and first_user is None:
                first_user = block
            elif label == "observation":
                observations.append(block)
            elif label == "reply":
                replies.append(block)
            label, i = None, j
        i += 1
    return first_user or "", observations, replies


class ReplayEnv:
    def __init__(self, first, observations):
        self.instructions = first
        self.observation = ""
        self.queue = list(observations)
        self.max_steps = max(len(observations), 1)
        self.reads_full_reply = True
        self.steps = 0
        self.done = not self.queue

    def admissible(self):
        return None

    def act(self, action):
        if self.done:
            raise EpisodeOver("act() called after the episode ended")
        self.steps += 1
        self.observation = self.queue.pop(0)
        self.done = not self.queue
        return self.observation


def make_chat(replies):
    state = {"k": 0}

    def chat(messages, temperature=None, max_tokens=None, top_p=None, stop=None):
        assert isinstance(messages, list) and all(isinstance(m, dict) and "role" in m for m in messages)
        r = replies[min(state["k"], len(replies) - 1)] if replies else "Action: [noop]"
        state["k"] += 1
        return r
    return chat


def run(harness, trace):
    spec = importlib.util.spec_from_file_location("candidate_harness", harness)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    first, obs, replies = parse(trace)
    env = ReplayEnv(first, obs)
    mod.run_episode(env, make_chat(replies))
    return env.steps


if __name__ == "__main__":
    harness = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "harness.py")
    traces = sys.argv[2:] or sorted(glob.glob(os.path.join(HERE, "evidence", "*", "episodes", "*", "trace.md")))[:5]
    if not traces:
        print("no trace to replay")
        sys.exit(0)
    bad = 0
    for t in traces:
        try:
            n = run(harness, t)
            print(f"ok   {os.path.relpath(t, HERE)}: {n} actions")
        except Exception:
            bad += 1
            print(f"CRASH {os.path.relpath(t, HERE)}:")
            traceback.print_exc(limit=8)
    sys.exit(1 if bad else 0)
