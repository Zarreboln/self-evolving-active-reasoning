"""The method's LLM roles, behind one call: session(role, workspace, log) -> record.

Roles: "build" (the Builder writes harness/components.py), "build_repair" (interface repair, paper B.2),
"reflect" (the Builder updates the bank: decision.json), "reflect_repair" (one retry of an invalid
decision; ours). Every role is the same Builder model, as in the paper (B.2: GPT-5.6-Sol builds and
updates); here Claude Opus 5.5.

ClaudeCodeSession   a headless Claude Code session in the Docker sandbox, through
                    sandbox/run_proposer.py, exactly as the Meta-Harness proposer: the container mounts
                    only the workspace, reads TASK.md, and uses the user's Claude subscription
                    (CLAUDE_CODE_OAUTH_TOKEN from a 600 file that run_proposer passes as a container
                    variable; nothing here reads it). No API key, no direct API call.
FakeSession         a Python function plays the role (tests).

The record holds rc, seconds and the token usage of the session's stream-json `result` line.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mb.baselines.metaskill.config import BUILDER_MODEL, SANDBOX

TIMEOUTS = {"build": 45 * 60, "build_repair": 30 * 60, "reflect": 45 * 60, "reflect_repair": 20 * 60}


def usage_of(log: Path) -> dict:
    """Tokens of one session from Claude Code's stream-json result line (as orchestrate.proposer_cost)."""
    out = {}
    if not log.exists():
        return out
    for line in log.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            u = e.get("usage") or {}
            out = {"input": u.get("input_tokens", 0), "cache_write": u.get("cache_creation_input_tokens", 0),
                   "cache_read": u.get("cache_read_input_tokens", 0), "output": u.get("output_tokens", 0),
                   "turns": e.get("num_turns"), "list_price_usd": e.get("total_cost_usd")}
    return out


class ClaudeCodeSession:
    def __init__(self, model: str = BUILDER_MODEL):
        self.model = model

    def __call__(self, role: str, ws: Path, log: Path) -> dict:
        if str(SANDBOX) not in sys.path:
            sys.path.insert(0, str(SANDBOX))
        import run_proposer
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = run_proposer.run(ws, log, model=self.model, timeout=TIMEOUTS[role])
        return {"role": role, "rc": rc, "seconds": round(time.time() - t0), "usage": usage_of(log)}


class FakeSession:
    """fn(role, ws) does what the session would do to the workspace; usage is recorded as zero."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, role: str, ws: Path, log: Path) -> dict:
        self.calls.append((role, str(ws)))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"type": "result", "fake": True}) + "\n")
        self.fn(role, Path(ws))
        return {"role": role, "rc": 0, "seconds": 0, "usage": {}}
