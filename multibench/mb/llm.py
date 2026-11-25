"""Chat callables for the policy and for LLM-played environment roles.

Both talk to an OpenAI-compatible endpoint. The model and endpoint are fixed per run from the
environment, never chosen by a harness, and every call is counted. Thinking is switched off on the
server (`--default-chat-template-kwargs`), so nothing here sends a template flag.

    POLICY_BASE_URL / POLICY_MODEL         the frozen model a harness wraps
    RESPONDER_BASE_URL / RESPONDER_MODEL   who plays 20Q's gamemaster, DC's suspects, SP's referee
                                           (defaults to the policy's endpoint and model)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request


class Counter:
    def __init__(self):
        self.calls = self.in_tok = self.out_tok = 0

    def as_dict(self):
        return {"calls": self.calls, "in_tok": self.in_tok, "out_tok": self.out_tok}


def make_chat(base_url: str, model: str, counter: Counter | None = None, retries: int = 4):
    url = base_url.rstrip("/") + "/chat/completions"

    def chat(messages, temperature=0.7, max_tokens=None, top_p=None, stop=None):
        body = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop:
            body["stop"] = stop
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for k in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    o = json.loads(r.read())
                break
            except Exception:
                if k == retries - 1:
                    raise
                time.sleep(2 ** k)
        if counter is not None:
            u = o.get("usage") or {}
            counter.calls += 1
            counter.in_tok += u.get("prompt_tokens", 0)
            counter.out_tok += u.get("completion_tokens", 0)
        return o["choices"][0]["message"]["content"] or ""

    return chat


def policy_chat(counter=None):
    return make_chat(os.environ["POLICY_BASE_URL"], os.environ["POLICY_MODEL"], counter)


def responder_chat(counter=None):
    return make_chat(os.environ.get("RESPONDER_BASE_URL", os.environ["POLICY_BASE_URL"]),
                     os.environ.get("RESPONDER_MODEL", os.environ["POLICY_MODEL"]), counter)
