"""An OpenAI-compatible endpoint that answers AR-Bench's referee turns with Jev.

AR-Bench's official evaluators separate the two model roles by environment variable: `POLICY_*` is
the reasoner under test, `RESPONSE_*` is the referee that answers the player. Both are called
through `openai.OpenAI(base_url=...).chat.completions.create`. This server speaks that API and
nothing else, so the official baselines run unmodified against Jev:

    RESPONSE_BASE_URL=http://127.0.0.1:8099/v1 RESPONSE_API_KEY=x bash script/evaluation/sp_evaluation.sh

Why Jev for this role. The referee is a three-way typed decision -- Yes / No / Unknown from a
hidden story -- which is Jev's native shape, and it needs no GPU, so the reasoner can have the whole
card. Calibration on TurtleBench-150: Jev 78.7 against the official Llama-3.1-405B's 83.3, with
better Unknown recall (36.8 against 26.3) and the only usable confidence among eleven candidates
(experiments/02). It is a substitution, not a reproduction: AR-Bench's referee is the 405B, so
numbers produced here are comparable across methods run under this same referee, not to the paper's.

The server rewrites the referee turn into one `choice` question whose criteria are AR-Bench's own
three rules, reads back Jev's label, and returns exactly "Yes", "No" or "Unknown" -- the three
strings the evaluators parse. Any request that is not a referee turn is refused rather than
answered by something else, so a silent fallback can never pollute a run.

    python runner/jev_referee_shim.py --port 8099          # key from the keychain
    python runner/jev_referee_shim.py --port 8099 --log runs/referee.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

JEV_API = "https://api.typesafe.ai/v1/systemone"

# AR-Bench's referee rules (arbench/reasoner/sp/prompt.py), as the criteria of one typed question.
CRITERIA = {
    "Yes": "The player's guess is correct, or the answer to it is affirmative.",
    "No": "The player's guess is incorrect, or the answer to it is negative.",
    "Unknown": "The guess cannot be answered from the <Surface> and the <Bottom>, and cannot be "
               "concluded from them by reasoning.",
}
INSTRUCTIONS = (
    "You are the referee of a game. Players are shown the <Surface>; you are given the <Bottom>. "
    "Understand the whole story from both, then judge the player's question. Players can only see "
    "the <Surface>, so they ask about the <Surface>. Answer Unknown for anything the story does not "
    "settle and that cannot be deduced from it."
)
SURFACE = re.compile(r"<Surface>\s*(.*?)\s*(?:<Bottom>|$)", re.S | re.I)
BOTTOM = re.compile(r"<Bottom>\s*(.*?)\s*(?:##|$)", re.S | re.I)


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k.strip()
    try:
        return subprocess.run(["security", "find-generic-password", "-s", "typesafe_token", "-w"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sys.exit("no API key: set TYPESAFE_API_KEY or store it as keychain service 'typesafe_token'")


def referee_turn(messages):
    """Pull (surface, bottom, question) out of a referee call, or None if this is not one."""
    system = " ".join(m.get("content") or "" for m in messages if m.get("role") == "system")
    joined = "\n".join((m.get("content") or "") for m in messages)
    s, b = SURFACE.search(joined), BOTTOM.search(joined)
    if not (s and b):
        return None
    if "referee" not in system.lower() and "referee" not in joined.lower():
        return None
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    question = (users[-1] if users else "").strip()
    # the last user turn may still carry the rules; keep only its final line, which is the question
    if len(question.splitlines()) > 1:
        question = question.splitlines()[-1].strip()
    return (s.group(1).strip(), b.group(1).strip(), question) if question else None


def ask_jev(surface, bottom, question, key, model, retries=4):
    body = {"model": model,
            "state": {"Surface": surface, "Bottom": bottom, "player_question": question},
            "questions": {"verdict": {"type": "choice", "instructions": INSTRUCTIONS,
                                      "criteria": CRITERIA}}}
    req = urllib.request.Request(JEV_API, method="POST", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    last = None
    for a in range(retries):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=60))
            ans = d["answers"]["verdict"]
            return ans.get("choice"), ans.get("confidence"), ans.get("probabilities"), d.get("usage")
        except Exception as e:                      # 429 / 5xx / transport
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"Jev failed after {retries} attempts: {last}")


class Handler(BaseHTTPRequestHandler):
    key = model = None
    log_path = None

    def _send(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):                      # keep stdout for our own lines
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": self.model, "object": "model"}]})
        self._send(404, {"error": {"message": "only /v1/chat/completions and /v1/models"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "only /v1/chat/completions"}})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": {"message": f"bad JSON: {e}"}})
        turn = referee_turn(req.get("messages") or [])
        if turn is None:
            # Refuse rather than answer with something else: a silent fallback would put a
            # different model's words into the transcript and nobody would see it.
            return self._send(400, {"error": {"message":
                "this endpoint only answers AR-Bench referee turns (a <Surface>, a <Bottom> and a "
                "player question); point the reasoner at a real chat model instead"}})
        surface, bottom, question = turn
        try:
            label, conf, probs, usage = ask_jev(surface, bottom, question, self.key, self.model)
        except Exception as e:
            return self._send(502, {"error": {"message": str(e)}})
        if label not in CRITERIA:
            return self._send(502, {"error": {"message": f"Jev returned {label!r}"}})
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"question": question, "label": label, "confidence": conf,
                                    "probabilities": probs, "usage": usage}) + "\n")
        self._send(200, {"id": "jev-referee", "object": "chat.completion", "model": self.model,
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": label}}],
                         "usage": {"prompt_tokens": (usage or {}).get("input_tokens", 0),
                                   "completion_tokens": 0,
                                   "total_tokens": (usage or {}).get("input_tokens", 0)}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--log", default=None, help="append every verdict, with Jev's confidence")
    a = p.parse_args()
    Handler.key, Handler.model = api_key(), a.model
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Handler.log_path = a.log
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Jev referee on http://127.0.0.1:{a.port}/v1  (model {a.model})", flush=True)
    print("point AR-Bench at it:  RESPONSE_BASE_URL=http://127.0.0.1:%d/v1 RESPONSE_API_KEY=x" % a.port,
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
