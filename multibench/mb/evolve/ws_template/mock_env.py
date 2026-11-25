"""Crash test for harness/harness.py against a fake game and a fake model. It checks that the harness
runs end to end, not how well it plays: the game accepts anything and ends after a few actions, and
the "model" always gives the same reply.

    python mock_env.py                  # runs harness/harness.py
    python check_harness.py harness/harness.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class EpisodeOver(Exception):
    pass


class ChatError(Exception):
    pass


class MockEnv:
    def __init__(self, steps=4):
        self.instructions = ("[MOCK] This is a stand-in game. Submit actions in square brackets, "
                             "e.g. [word]. You have %d turns." % steps)
        self.observation = ""
        self.max_steps = steps
        self.reads_full_reply = True
        self.steps = 0
        self.done = False

    def admissible(self):
        return None

    def act(self, action):
        if self.done:
            raise EpisodeOver()
        self.steps += 1
        self.done = self.steps >= self.max_steps
        self.observation = "[MOCK] You submitted %r. Feedback: nothing to report." % str(action)[:80]
        return self.observation


def mock_chat(messages, temperature=0.7, max_tokens=8192):
    assert isinstance(messages, list) and all(isinstance(m, dict) and "role" in m for m in messages)
    return "<reasoning>mock reasoning</reasoning>\nAction: [mock]"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "harness", "harness.py")
    spec = importlib.util.spec_from_file_location("candidate_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = MockEnv()
    mod.run_episode(env, mock_chat)
    print("ok: %d actions, done=%s" % (env.steps, env.done))
