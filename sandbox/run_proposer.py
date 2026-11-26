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
DOCKER = "/Applications/Docker.app/Contents/Resources/bin/docker"   # ~/.docker/bin links break when Docker restarts
NETWORK = "ar-proposer-net"          # internal network: the proxy is the only way out
PROXY = "http://ar-egress:8888"
TOOLS = "Read,Glob,Grep,Agent,Write,Edit,Bash"
CRED_FILES = {"CLAUDE_CODE_OAUTH_TOKEN": "~/.claude_proposer_oauth_token",   # subscription (claude setup-token)
              "ANTHROPIC_API_KEY": "~/.anthropic_proposer_api_key"}          # pay-as-you-go
# DeepSeek serves an Anthropic-compatible endpoint, so Claude Code drives it unchanged (as in MAS.S62 HW2).
DEEPSEEK_KEY_FILE = "~/.deepseek_proposer_api_key"
DEEPSEEK_URL = "https://api.deepseek.com/anthropic"
# Prime Intellect inference also accepts Anthropic-format requests (checked 2026-11-26); model ids
# carry a provider prefix, e.g. qwen/qwen3.5-9b.
PRIME_KEY_FILE = "~/.prime_proposer_api_key"
PRIME_URL = "https://api.pinference.ai/api"
# Prime-hosted models that its Anthropic-format API does not serve go through the LiteLLM bridge
# (sandbox/bridge/), a container on the proposer network: model "bridge:<name in bridge/config.yaml>".
BRIDGE_URL = "http://ar-bridge:4000"
BRIDGE_KEY_FILE = "~/.bridge_proposer_key"
# USD per 1M tokens (input cache miss, input cache hit, output), off-peak; peak hours
# (01:00-04:00 and 06:00-10:00 UTC, Mon-Fri) are double. Taken from MAS.S62 HW2 (hw2/model.py, 2026-11).
PRICES = {"deepseek-v4-pro": (0.66, 0.022, 1.98), "deepseek-flash": (0.15, 0.003, 0.60)}
# Prime Intellect catalog (api.pinference.ai/api/v1/models, read 2026-11-26); no peak pricing.
PRIME_PRICES = {"qwen/qwen3.5-9b": (0.17, 0.153, 0.25), "qwen/qwen3-235b-a22b-2507": (0.25, 0.25, 1.0),
                "anthropic/claude-opus-5.5": (4.0, 4.0, 20.0),
                "qwen3.5-9b": (0.17, 0.153, 0.25), "qwen3.6-35b-a3b": (0.19, 0.17, 1.7),
                "qwen3.8-27b": (0.45, 0.05, 3.2)}


def _credential():
    found = [(var, Path(p).expanduser()) for var, p in CRED_FILES.items() if Path(p).expanduser().exists()]
    if len(found) != 1:
        raise RuntimeError(f"need exactly one of {list(CRED_FILES.values())}, found {[str(p) for _, p in found]}")
    var, path = found[0]
    return var, path.read_text().strip()


def _deepseek_env(model, key_file=DEEPSEEK_KEY_FILE, url=DEEPSEEK_URL):
    secret = Path(key_file).expanduser().read_text().strip()
    names = {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_MODEL": model,
             "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
             "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "CLAUDE_CODE_SUBAGENT_MODEL": model}
    return "ANTHROPIC_AUTH_TOKEN", secret, names


def deepseek_cost(log: Path):
    """Metered USD of a DeepSeek session from its own token counts (Claude Code's total_cost_usd uses
    Anthropic list prices, about nine times too high here; HW2). Off-peak price, x2 if any part of the
    session fell in DeepSeek's peak hours; returns (usd, peak)."""
    import json, time
    cost, peak = 0.0, False
    for line in log.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            cost = 0.0
            for m, u in (e.get("modelUsage") or {}).items():
                miss, hit, out = PRICES.get(m, PRICES["deepseek-v4-pro"])
                cost += ((u.get("inputTokens", 0) + u.get("cacheCreationInputTokens", 0)) * miss
                         + u.get("cacheReadInputTokens", 0) * hit + u.get("outputTokens", 0) * out) / 1e6
    t = time.gmtime(log.stat().st_mtime)
    peak = t.tm_wday < 5 and (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)
    return round(cost * (2 if peak else 1), 4), peak


def prime_cost(log: Path):
    """Metered USD of a Prime Intellect session from its own token counts and the catalog price."""
    import json
    cost = 0.0
    for line in log.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            cost = 0.0
            for m, u in (e.get("modelUsage") or {}).items():
                miss, hit, out = PRIME_PRICES[m]
                cost += ((u.get("inputTokens", 0) + u.get("cacheCreationInputTokens", 0)) * miss
                         + u.get("cacheReadInputTokens", 0) * hit + u.get("outputTokens", 0) * out) / 1e6
    return round(cost, 4)


def run(ws: Path, log: Path, model="claude-opus-5-5", timeout=45 * 60, network=True):
    ws = ws.resolve()
    task = "Follow the instructions in TASK.md (they are repeated here).\n\n" + (ws / "TASK.md").read_text()
    extra = {}
    if model.startswith("deepseek"):
        var, secret, extra = _deepseek_env(model)
    elif model.startswith("bridge:"):
        model = model.split(":", 1)[1]
        var, secret, extra = _deepseek_env(model, BRIDGE_KEY_FILE, BRIDGE_URL)
        extra["NO_PROXY"] = extra["no_proxy"] = "ar-bridge"
    elif "/" in model:                                  # provider/model -> Prime Intellect
        var, secret, extra = _deepseek_env(model, PRIME_KEY_FILE, PRIME_URL)
    else:
        var, secret = _credential()
    cmd = [DOCKER, "run", "--rm", "-i",
           "--read-only", "--tmpfs", "/home/proposer:rw,exec,uid=1001,gid=1001,size=512m",
           "--tmpfs", "/tmp:rw,exec,size=512m",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--memory", "6g", "--pids-limit", "512",
           "-v", f"{ws}:/work:rw", "-w", "/work",
           "-e", var, "-e", "HOME=/home/proposer", "-e", "DISABLE_AUTOUPDATER=1",
           "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"]
    for k, v in extra.items():
        cmd += ["-e", f"{k}={v}"]
    if network:
        cmd += ["--network", NETWORK, "-e", f"HTTPS_PROXY={PROXY}", "-e", f"HTTP_PROXY={PROXY}"]
    else:
        cmd += ["--network", "none"]
    cmd += [IMAGE, "claude", "-p", task, "--safe-mode", "--permission-mode", "acceptEdits",
            "--output-format", "stream-json", "--verbose", "--model", model,
            "--tools", TOOLS, "--allowedTools", TOOLS, "--disallowedTools", "WebFetch,WebSearch"]
    env = os.environ | {var: secret, "PATH": os.path.dirname(DOCKER) + ":" + os.environ["PATH"]}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        try:
            r = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout)
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
