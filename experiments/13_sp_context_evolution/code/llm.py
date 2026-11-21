"""OpenRouter chat client for the context-evolution experiments.

- Standard library only (urllib); no installs.
- The key is read from secrets/openrouter.key or env OPENROUTER_API_KEY and is never printed or logged.
- Every call is appended to a JSONL ledger with role, model, token usage and the provider-reported cost,
  so spending can be audited per role (actor / referee / judge / reflector).
- A hard budget guard refuses new calls once the ledger total reaches the cap.
- A disk cache keyed by the full request makes reruns free and the referee deterministic within a run.
- MOCK mode (env MAS_MOCK=1) returns canned replies so plumbing can be tested with zero spend.
  Mock output is engineering evidence only, never a result.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = ROOT / "secrets" / "openrouter.key"
# One ledger file per process: concurrent appends from several processes to one file interleave on Windows
# (observed 2026-11-07). runs/ledger.jsonl holds the earlier single-file records; ledger_files() reads all.
LEDGER_DIR = ROOT / "runs" / "ledger"
LEDGER = Path(os.environ.get("MAS_LEDGER", str(LEDGER_DIR / f"p{os.getpid()}_{int(time.time())}.jsonl")))
CACHE_DIR = ROOT / "runs" / "cache"
URL = "https://openrouter.ai/api/v1/chat/completions"
BUDGET_USD = float(os.environ.get("MAS_BUDGET_USD", "300"))

_lock = threading.Lock()
_spent = None


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding="ascii", errors="ignore").strip()
    if not k:
        raise RuntimeError("No OpenRouter key: see notes/OPENROUTER_SETUP.md")
    return k


def ledger_files() -> list[Path]:
    files = sorted(LEDGER_DIR.glob("*.jsonl")) if LEDGER_DIR.exists() else []
    legacy = ROOT / "runs" / "ledger.jsonl"
    return ([legacy] if legacy.exists() else []) + files


def ledger_rows() -> list[dict]:
    """All parseable ledger records from every process (malformed lines skipped)."""
    out = []
    for f in ledger_files():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def spent_usd() -> float:
    """Spend recorded by all processes' ledgers at first call, plus this process's own calls since.
    The authoritative hard cap is the OpenRouter key limit (see key_status)."""
    global _spent
    with _lock:
        if _spent is None:
            _spent = _stream_total() if "MAS_LEDGER" not in os.environ else 0.0
        return _spent


def _stream_total() -> float:
    """Sum cost_usd line by line without keeping the records: the ledger reached ~224 MB on 2026-11-08 and loading
    all rows in several concurrent processes caused MemoryError."""
    tot = 0.0
    for f in ledger_files():
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"cost_usd"' not in line:
                    continue
                try:
                    tot += float(json.loads(line).get("cost_usd") or 0)
                except Exception:
                    pass
    return tot


def _log(rec: dict):
    global _spent
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _spent = (_spent or 0.0) + float(rec.get("cost_usd") or 0)


def _mock_reply(messages, role):
    last = messages[-1]["content"] if messages else ""
    if role == "referee":
        return random.choice(["Yes", "No", "Unknown"])
    if role == "keyjudge":
        return random.choice(["Hit point: 1", "No point is hit", "Hit point: 2, 3"])
    if role == "judge":
        return '{"verdict": "neutral"}'
    if role == "reflector":
        return ('{"analysis": "mock", "ops": [{"op": "ADD", "when": "three answers in a row were No or Unknown", '
                '"do": "ask one broad question about a different aspect of the story"}]}')
    if "give your answer" in last.lower():
        return "A: The man faked his own death."
    return f"Q: Was something hidden from the family (mock {random.randint(0, 999)})?"


# Models that reason by default on OpenRouter; their hidden reasoning eats short max_tokens budgets
# (observed 2026-11-07: empty content with finish=length at max_tokens=8).
REASONING_BY_DEFAULT = ("anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5",
                        "qwen/qwen3-32b")


def chat(model: str, messages: list[dict], *, role: str, temperature: float = 0.7,
         max_tokens: int = 256, seed: int | None = None, use_cache: bool = True,
         tag: str = "", extra: dict | None = None, reasoning: bool = False) -> str:
    """One chat completion. Returns the text (empty string on a final failure, which is logged).
    reasoning=False switches hidden reasoning off for models that reason by default."""
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "usage": {"include": True}}
    if seed is not None:
        body["seed"] = seed
    if not reasoning and model in REASONING_BY_DEFAULT:
        body["reasoning"] = {"enabled": False}
    if extra:
        body.update(extra)
    h = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cpath = CACHE_DIR / h[:2] / f"{h}.json"
    if use_cache and cpath.exists():
        try:   # another process may be mid-write; an unreadable entry is treated as a miss
            return json.loads(cpath.read_text(encoding="utf-8"))["text"]
        except Exception:
            pass

    if os.environ.get("MAS_MOCK") == "1":
        text = _mock_reply(messages, role)
        _log({"t": time.time(), "role": role, "model": model, "mock": True, "cost_usd": 0, "tag": tag})
        return text

    if spent_usd() >= BUDGET_USD:
        raise RuntimeError(f"Budget guard: ledger total ${spent_usd():.2f} >= cap ${BUDGET_USD:.2f}")

    data = json.dumps(body).encode()
    last_err = None
    for attempt in range(8):
        req = urllib.request.Request(URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
            "X-Title": "mas-context-evolution"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read().decode("utf-8"))
            if "error" in out:
                raise RuntimeError(str(out["error"])[:300])
            text = (out["choices"][0]["message"].get("content") or "").strip()
            usage = out.get("usage") or {}
            rec = {"t": t0, "dt": round(time.time() - t0, 3), "role": role, "model": model,
                   "provider": out.get("provider"), "finish": out["choices"][0].get("finish_reason"),
                   "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
                   "cost_usd": usage.get("cost"), "tag": tag}
            _log(rec)
            if use_cache and text:   # never cache an empty reply; write atomically (temp file + rename)
                cpath.parent.mkdir(parents=True, exist_ok=True)
                tmp = cpath.with_name(f"{cpath.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps({"text": text, "rec": rec}, ensure_ascii=False), encoding="utf-8")
                try:
                    os.replace(tmp, cpath)
                except OSError:   # another writer won the race with identical content
                    tmp.unlink(missing_ok=True)
            return text
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (400, 401, 402, 403):
                try:
                    last_err += " " + e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                break
            if e.code == 429:
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    time.sleep(min(60, float(ra)) if ra else min(60, 5 * (attempt + 1) + random.random() * 5))
                except ValueError:
                    time.sleep(10)
                continue
        except Exception as e:  # network / parse
            last_err = type(e).__name__ + ": " + str(e)[:200]
        time.sleep(min(30, 2 ** attempt + random.random()))
    _log({"t": time.time(), "role": role, "model": model, "error": last_err, "cost_usd": 0, "tag": tag})
    return ""


def key_status() -> dict:
    """Remaining credit for the key (OpenRouter /key endpoint). Never returns the key itself."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode("utf-8")).get("data", {})
    return {k: d.get(k) for k in ("limit", "limit_remaining", "usage", "is_free_tier")}
