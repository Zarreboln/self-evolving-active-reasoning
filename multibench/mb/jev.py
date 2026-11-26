"""Jev (TypeSafe's model via JevHub, pinned jev-1.14.0) as the environment's answerer wherever the role is a typed
choice from hidden information: SP's referee, 20Q's gamemaster, Guess Who's gamemaster. DC's
suspects answer in free-text role play, which is not Jev's shape, so they stay on the chat model.

  * SP reuses the referee of ~/code/jev_referee_shim.py unchanged (its instance extraction, the
    AR-Bench rules as criteria, the Yes/No/Unknown labels), so SP scores here are on the same
    referee as the Meta-Harness run1 and the q35 baseline plan.
  * 20Q and Guess Who answer with TextArena's own option strings ("Yes", "No", "I don't know"),
    since TextArena matches the gamemaster's reply against exactly those.

Jev bills input tokens only (USD 0.042 per million); every call's usage is counted on the
responder so a run can report its spend.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

# The user's access is a JevHub (jevgo.cn, Jev's official reseller) key: same protocol, same model
# names, different base URL. The direct TypeSafe organisation was deactivated on 2026-11-25.
API = os.environ.get("JEV_API", "https://api.jevgo.cn/v1/systemone")
MODEL = "jev-1.14.0"
RATE_IN = 0.042 / 1_000_000

sys.path.insert(0, os.path.expanduser("~/code"))


def _key():
    k = os.environ.get("JEVHUB_API_KEY")
    return k.strip() if k else open(os.path.expanduser("~/.jevhub_token")).read().strip()


class Jev:
    def __init__(self, model=MODEL):
        self.model = model
        self.key = _key()
        self.calls = self.in_tok = 0
        self._lock = threading.Lock()

    def choose(self, state, instructions, criteria, retries=8):
        body = {"model": self.model, "state": state,
                "questions": {"answer": {"type": "choice", "instructions": instructions,
                                         "criteria": criteria}}}
        req = urllib.request.Request(API, method="POST", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        last = None
        for a in range(retries):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=120))
                with self._lock:
                    self.calls += 1
                    self.in_tok += (d.get("usage") or {}).get("input_tokens", 0) or 0
                label = d["answers"]["answer"].get("choice")
                if label in criteria:
                    return label
                last = ValueError(f"Jev returned {label!r}")
            except urllib.error.HTTPError as e:     # keep the server's reason and what was asked
                last = RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}; state_chars={len(json.dumps(state))}; "
                                   f"state={json.dumps(state)[:300]}")
            except Exception as e:          # 429 / 5xx / transport
                last = e
            # JevHub intermittently answers a 400 'Unknown model' under concurrent load (a short request
            # that fails in a burst succeeds alone), so back off long enough to ride it out
            time.sleep(min(60.0, 1.5 * 2 ** a))
        raise RuntimeError(f"Jev failed after {retries} attempts: {last}")

    def usage(self):
        return {"calls": self.calls, "in_tok": self.in_tok, "usd": round(self.in_tok * RATE_IN, 6)}


class SPReferee:
    """callable(messages, temperature, max_tokens) -> "Yes" | "No" | "Unknown", as SPEnv expects."""
    jev_role = "sp"

    def __init__(self, jev):
        import jev_referee_shim as shim
        self.shim, self.jev = shim, jev

    def __call__(self, messages, temperature=0.7, max_tokens=None):
        turn = self.shim.referee_turn(messages)
        if turn is None:
            raise ValueError("not an AR-Bench referee turn")
        surface, bottom, question = turn
        return self.jev.choose({"Surface": surface, "Bottom": bottom, "player_question": question},
                               self.shim.INSTRUCTIONS, self.shim.CRITERIA)


_GM_CRITERIA = {
    "Yes": "The truthful answer to the player's question about the secret is yes.",
    "No": "The truthful answer to the player's question about the secret is no.",
    "I don't know": "The question cannot be answered yes or no about the secret, or the secret's "
                    "description does not settle it.",
}


class Gamemaster:
    """Answers one yes/no question about a known secret with TextArena's option strings."""
    jev_role = "gamemaster"

    def __init__(self, jev):
        self.jev = jev

    def answer(self, game, secret, question):
        instructions = (f"You are the gamemaster of {game}. You know the secret given in the state; "
                        "the player does not. Answer the player's question truthfully about the "
                        "secret, choosing exactly one option. A question that is not a yes-or-no "
                        "question (e.g. 'what color is it?') gets \"I don't know\".")
        return self.jev.choose({"secret": secret, "player_question": question},
                               instructions, _GM_CRITERIA)


def responder_for(bench, jev):
    """The Jev responder for a bench whose role it can play, else None."""
    if bench == "sp":
        return SPReferee(jev)
    if bench in ("twentyq", "guesswho"):
        return Gamemaster(jev)
    return None
