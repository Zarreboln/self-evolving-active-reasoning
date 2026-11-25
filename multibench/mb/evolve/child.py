"""Runs one candidate harness in a child process that sees the episode only as text.

Started by mb/evolve/broker.py as `python -I child.py HARNESS.py LIB_DIR` in an empty directory with
an empty environment. The broker holds the real environment and the model endpoint; this process
gets the instructions, and then exchanges JSON lines with the broker: `chat` returns a model reply,
`act` returns the next observation. Nothing else of the environment exists here, so a harness
cannot read the secret it holds, and mb/evolve/check.py has already rejected imports and builtins
that could reach files, the network or other objects.

The harness's own prints go to stderr (kept, truncated, in its trace); stdout is the protocol.
"""
import importlib.util
import json
import os
import sys
import traceback

PROTO = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = sys.stderr
REQUESTS = sys.stdin


class EpisodeOver(Exception):
    pass


class ChatError(Exception):
    pass


def call(msg):
    PROTO.write(json.dumps(msg) + "\n")
    line = REQUESTS.readline()
    if not line:
        sys.exit(3)
    return json.loads(line)


class Env:
    """What a harness may use: instructions, the latest observation, admissible actions, act()."""

    def __init__(self, init):
        self.instructions = init["instructions"]
        self.observation = init.get("observation", "")     # the env's first prompt, where it has one
        self.max_steps = init["max_steps"]
        self.reads_full_reply = init["reads_full_reply"]
        self.steps = 0
        self.done = False
        self.adm = init["admissible"]

    def admissible(self):
        return list(self.adm) if self.adm is not None else None

    def act(self, action):
        if self.done:
            raise EpisodeOver()
        r = call({"op": "act", "action": str(action)})
        if "error" in r:
            raise EpisodeOver(r["error"])
        self.observation, self.done, self.steps, self.adm = r["obs"], r["done"], r["steps"], r["admissible"]
        return self.observation


def chat(messages, temperature=0.7, max_tokens=8192, top_p=None, stop=None):
    r = call({"op": "chat", "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
              "top_p": top_p, "stop": stop})
    if "error" in r:
        raise ChatError(r["error"])
    return r["reply"]


def main():
    harness, lib = sys.argv[1], sys.argv[2]
    sys.path.insert(0, lib)
    init = json.loads(REQUESTS.readline())
    try:
        spec = importlib.util.spec_from_file_location("candidate_harness", harness)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_episode(Env(init), chat)
        PROTO.write(json.dumps({"op": "end"}) + "\n")
    except BaseException:
        PROTO.write(json.dumps({"op": "crash", "error": traceback.format_exc()[-3000:]}) + "\n")


if __name__ == "__main__":
    main()
