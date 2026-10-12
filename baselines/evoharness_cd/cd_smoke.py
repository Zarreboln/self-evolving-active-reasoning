"""Scripted-policy smoke test of the CircuitDecoding adapter (no GPU)."""
import re, sys
sys.path.insert(0, ".")
from evoharness.envs import get_domain
from evoharness.rollout import run_episode
from evoharness.workspace import HarnessConfig
from evoharness.domain import Task
dom = get_domain("circuitdecoding")
def q_str(i):
    lb = "ABC"[i // 8]; r = i % 8
    return f"query {lb}({(r>>2)&1}, {(r>>1)&1}, {r&1})"
class Scripted:
    def __init__(self, mode): self.mode, self.n, self.tracked = mode, 0, False
    def __call__(self, messages):
        last = messages[-1]["content"]
        if "Please provide the truth tables" in last or "guess <24 bits" in last:
            if self.mode == "inline" and not self.tracked:
                self.tracked = True; return "<think>c</think><action>track [remaining]</action>"
            m = re.findall(r"\b([01]{24})\b", last); return f"<think>a</think><action>guess {m[0] if m else '0'*24}</action>"
        i = self.n; self.n += 1; return f"<think>s</think><action>{q_str(i % 24)}</action>"
for mode in ("env_only", "always_on", "inline"):
    rec = run_episode(dom.make_env(), Task(id="cd001", data=1), domain=dom, policy=Scripted(mode),
                      harness=HarnessConfig(mode=mode, charge_meta_actions=False, max_meta_actions=18), bank=None, max_steps=19)
    print(mode, "won", rec["won"], "turns", rec["turns"], "meta", rec["meta_actions"]["executed"]["track"])
