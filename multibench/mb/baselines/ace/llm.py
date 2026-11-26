"""ACE's own LLM (Reflector and Curator): one function, `llm(role, call_id, prompt) -> text`.

The real one is a headless Claude Code session on the user's subscription (Opus 5.5), run in the
proposer's Docker sandbox (ar_harness/sandbox/run_proposer.py: read-only root, fresh HOME, egress only
to the Anthropic API, credential passed as a container environment variable and never written to
disk). A Reflector or Curator call is a pure prompt -> text call, so the session has no tools and one
turn; the prompt goes in on stdin (an 80K-character prompt is too long for one argv entry).

ACE sends each role prompt as a single user message with temperature 0 and max_tokens 4096 to a
non-thinking model. Here: the same single message; max output 4096 tokens
(CLAUDE_CODE_MAX_OUTPUT_TOKENS); thinking off (MAX_THINKING_TOKENS=0); Claude Code cannot set the
temperature, and it needs a system prompt, for which we pass a one-line neutral one.

Every call's usage is appended to a ledger (JSONL) for the cost report.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from mb.baselines.ace.official import ROLE_MAX_TOKENS

SANDBOX = Path(__file__).resolve().parents[4] / "sandbox"
MODEL = "claude-opus-5-5"
SYSTEM = "You are a helpful assistant."


class LLMError(Exception):
    pass


class ClaudeCodeLLM:
    def __init__(self, ledger: Path, model=MODEL, timeout=900, retries=6):
        self.ledger, self.model, self.timeout, self.retries = Path(ledger), model, timeout, retries
        sys.path.insert(0, str(SANDBOX))
        import run_proposer                       # constants and the credential loader of the sandbox
        self.rp = run_proposer
        self.lock = threading.Lock()

    def _cmd(self, var):
        rp = self.rp
        return [rp.DOCKER, "run", "--rm", "-i",
                "--read-only", "--tmpfs", "/home/proposer:rw,exec,uid=1001,gid=1001,size=512m",
                "--tmpfs", "/tmp:rw,exec,size=256m", "--tmpfs", "/work:rw,uid=1001,gid=1001,size=16m",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--memory", "2g", "--pids-limit", "256", "-w", "/work",
                "-e", var, "-e", "HOME=/home/proposer", "-e", "DISABLE_AUTOUPDATER=1",
                "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
                "-e", f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={ROLE_MAX_TOKENS}", "-e", "MAX_THINKING_TOKENS=0",
                "--network", rp.NETWORK, "-e", f"HTTPS_PROXY={rp.PROXY}", "-e", f"HTTP_PROXY={rp.PROXY}",
                rp.IMAGE, "claude", "-p", "--safe-mode", "--output-format", "stream-json", "--verbose",
                "--model", self.model, "--tools", "", "--strict-mcp-config",
                "--system-prompt", SYSTEM]

    def __call__(self, role: str, call_id: str, prompt: str) -> str:
        var, secret = self.rp._credential()
        if var != "CLAUDE_CODE_OAUTH_TOKEN":
            raise LLMError("ACE's roles must run on the subscription (CLAUDE_CODE_OAUTH_TOKEN), "
                           f"but the sandbox credential is {var}")
        env = os.environ | {var: secret, "PATH": os.path.dirname(self.rp.DOCKER) + ":" + os.environ["PATH"]}
        last = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                p = subprocess.run(self._cmd(var), input=prompt, env=env, capture_output=True, text=True,
                                   timeout=self.timeout)
                text, usage = _parse(p.stdout)
                if p.returncode == 0 and text:
                    self._log(role, call_id, prompt, text, usage, time.time() - t0, attempt)
                    return text
                last = f"exit {p.returncode}: {(p.stderr or p.stdout)[-400:]}"
            except subprocess.TimeoutExpired:
                last = f"timeout after {self.timeout}s"
            time.sleep(min(600, 15 * 2 ** attempt))        # ACE's llm.py also backs off and retries
        raise LLMError(f"{role} {call_id}: {last}")         # infra: the run stops and resumes later

    def _log(self, role, call_id, prompt, text, usage, secs, attempt):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.ledger.open("a") as f:
            f.write(json.dumps({"time": time.time(), "role": role, "call_id": call_id, "seconds": round(secs, 1),
                                "attempt": attempt, "prompt_chars": len(prompt), "reply_chars": len(text),
                                **usage}) + "\n")


def _parse(stream: str):
    """The `result` event of a stream-json session: its text and token usage."""
    text, usage = "", {}
    for line in stream.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            text = "" if e.get("is_error") else (e.get("result") or "")
            u = e.get("usage") or {}
            usage = {k: u.get(k, 0) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens")}
            usage["total_cost_usd_list_price"] = e.get("total_cost_usd")
    return text, usage
