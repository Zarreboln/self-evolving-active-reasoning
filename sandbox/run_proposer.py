"""Run one proposer session (Claude Code) inside the ar-proposer container.

The isolation, compared with MAS.S62 HW2 (fresh workspace + --safe-mode, but Bash could still read
any host path):
  * the container sees exactly one host directory, the workspace, at /work; nothing else of the
    host is mounted, so earlier runs, notes, RUN_LOG, test splits and memory are not reachable;
  * HOME is a fresh tmpfs every run: no CLAUDE.md, auto-memory, skills, plugins or old sessions;
  * root filesystem read-only, all capabilities dropped, no privilege escalation, pid/memory caps;
  * --safe-mode on top, and no web tools (WebFetch, WebSearch);
  * outbound traffic goes through an allowlist proxy that passes only the Anthropic API (see
    egress/), so Bash cannot fetch benchmark data from the internet either.
The orchestrator builds the workspace (what the proposer may read) and reads back what it wrote.

Credentials come from a 600 file outside every workspace and are passed as an environment variable
of the container; they are never written into /work.

    python run_proposer.py WORKSPACE LOG [--model claude-opus-5-5] [--timeout 2700]
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

IMAGE = "ar-proposer:2.1.326"
DOCKER = os.path.expanduser("~/.docker/bin/docker")
NETWORK = "ar-proposer-net"          # internal network: the proxy is the only way out
PROXY = "http://ar-egress:8888"
TOOLS = "Read,Glob,Grep,Agent,Write,Edit,Bash"
CRED_FILES = {"CLAUDE_CODE_OAUTH_TOKEN": "~/.claude_proposer_oauth_token",   # subscription (claude setup-token)
              "ANTHROPIC_API_KEY": "~/.anthropic_proposer_api_key"}          # pay-as-you-go


def _credential():
    found = [(var, Path(p).expanduser()) for var, p in CRED_FILES.items() if Path(p).expanduser().exists()]
    if len(found) != 1:
        raise RuntimeError(f"need exactly one of {list(CRED_FILES.values())}, found {[str(p) for _, p in found]}")
    var, path = found[0]
    return var, path.read_text().strip()


def run(ws: Path, log: Path, model="claude-opus-5-5", timeout=45 * 60, network=True):
    ws = ws.resolve()
    task = "Follow the instructions in TASK.md (they are repeated here).\n\n" + (ws / "TASK.md").read_text()
    var, secret = _credential()
    cmd = [DOCKER, "run", "--rm", "-i",
           "--read-only", "--tmpfs", "/home/proposer:rw,exec,uid=1001,gid=1001,size=512m",
           "--tmpfs", "/tmp:rw,exec,size=512m",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--memory", "6g", "--pids-limit", "512",
           "-v", f"{ws}:/work:rw", "-w", "/work",
           "-e", var, "-e", "HOME=/home/proposer", "-e", "DISABLE_AUTOUPDATER=1",
           "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"]
    if network:
        cmd += ["--network", NETWORK, "-e", f"HTTPS_PROXY={PROXY}", "-e", f"HTTP_PROXY={PROXY}"]
    else:
        cmd += ["--network", "none"]
    cmd += [IMAGE, "claude", "-p", task, "--safe-mode", "--permission-mode", "acceptEdits",
            "--output-format", "stream-json", "--verbose", "--model", model,
            "--allowedTools", TOOLS, "--disallowedTools", "WebFetch,WebSearch"]
    env = os.environ | {var: secret, "PATH": os.path.dirname(DOCKER) + ":" + os.environ["PATH"]}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        try:
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            return r.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[run_proposer] timed out after {timeout}s\n")
            return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", type=Path)
    ap.add_argument("log", type=Path)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--timeout", type=int, default=45 * 60)
    a = ap.parse_args()
    print(run(a.workspace, a.log, a.model, a.timeout))
