"""A fake game and a fake Target for interface checks (stdlib only: also copied into the Builder's
workspace). They check that a harness runs end to end and that every tool it lists can be called, not
how well it plays: the game accepts anything and ends after a few actions."""
import json
import re


class MockEnv:
    def __init__(self, steps=4, listed=False, full_reply=True):
        self.instructions = ("[MOCK] This is a stand-in game. Submit actions in square brackets, e.g. [word]. "
                             "You have %d turns." % steps)
        self.observation = "[MOCK] Make your first move."
        self.max_steps, self.reads_full_reply = steps, full_reply
        self.listed = listed
        self.steps, self.done = 0, False

    def admissible(self):
        return ["look", "go north", "take key"] if self.listed and not self.done else None

    def act(self, action):
        if self.done:
            raise RuntimeError("the episode is over")
        self.steps += 1
        self.done = self.steps >= self.max_steps
        self.observation = "[MOCK] You submitted %r. Feedback: nothing to report." % str(action)[:80]
        return self.observation

    def result(self):
        return {"instance_id": "mock", "score": 0.0, "success": False, "steps": self.steps}


def dummy(schema):
    t = (schema or {}).get("type")
    if "enum" in (schema or {}):
        return schema["enum"][0]
    if t == "object":
        props = schema.get("properties", {})
        return {k: dummy(v) for k, v in props.items() if k in schema.get("required", [])}
    return {"string": "x", "integer": 1, "number": 1.0, "boolean": True, "array": []}.get(t)


class MockChat:
    """Calls every listed harness tool once (with dummy arguments of its schema), then acts."""

    def __init__(self):
        self.called = set()

    def __call__(self, messages, temperature=None, max_tokens=None, top_p=None, stop=None):
        assert isinstance(messages, list) and all(isinstance(m, dict) and "role" in m for m in messages)
        system = messages[0]["content"]
        for m in re.finditer(r"^- ([A-Za-z][A-Za-z0-9_]*) (\{.*\}): ", system, re.M):
            name = m.group(1)
            if name not in self.called:
                self.called.add(name)
                return "Let me use a tool.\nTool: %s %s" % (name, json.dumps(dummy(json.loads(m.group(2)))))
        return "<reasoning>mock</reasoning>\nAction: [mock]"


SCENARIOS = [dict(steps=4, listed=False, full_reply=True), dict(steps=3, listed=True, full_reply=False)]
