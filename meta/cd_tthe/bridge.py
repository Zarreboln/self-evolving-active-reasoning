"""Data, environment, frozen solver and trace bridge for the Circuit Decoding TTHE loop.

The environment holds the truth; a harness only ever sees answer bits through env.query. The
solver is a frozen chat model behind an OpenAI-compatible endpoint (or a scripted mock for CPU
tests); every call is cached on disk keyed by (messages, temperature, seq) so re-observing an
unchanged harness replays for free, as in TTHE's solver cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "stage0"))
sys.path.insert(0, str(ROOT / "runner"))
from cd_oracle import CDGame, load_instances  # noqa: E402   (public hypothesis space)

BUDGET, C, K = 18, 3, 3
LABELS = "ABC"
INSTRUCTION = (ROOT / "mt_puzzles/cd_instruction.txt").read_text()
LAST_TURN = (ROOT / "mt_puzzles/cd_last_turn_prompt.txt").read_text()
INSTANCES = load_instances(ROOT / "mt_puzzles/cd_test.jsonl")
_GAMES = {}


def game_for(inventory):
    key = tuple(sorted(inventory.items()))
    if key not in _GAMES:
        _GAMES[key] = CDGame(dict(key))
    return _GAMES[key]


def fill_instruction(inventory, budget=BUDGET):
    labels = list(LABELS[:C])
    comp = ", ".join(f"{v} {g} gate{'s' if v != 1 else ''}" for g, v in sorted(inventory.items()) if v)
    order = ", ".join(f"{lb}({','.join(str((r >> (K - 1 - i)) & 1) for i in range(K))})"
                      for lb in labels for r in range(1 << K))
    return (INSTRUCTION.replace("{components}", comp).replace("{num_circuits}", str(C))
            .replace("{circuit_labels}", ", ".join(labels)).replace("{circuit_labels_str}", ", ".join(labels))
            .replace("{num_inputs}", str(K)).replace("{num_questions_per_episode}", str(budget))
            .replace("{truth_table_description}", order))


class BudgetExhausted(Exception):
    pass


class CDEnv:
    """One episode's environment. Truth is private; a harness uses query/remaining/history."""

    def __init__(self, index, budget=BUDGET):
        inst = INSTANCES[index]
        self.index = index
        self.budget = int(budget)
        self.inventory = dict(inst["components"])
        self._truth = inst["truth"]                   # never read by a harness (audit forbids it)
        self.history = []
        self.system_prompt = fill_instruction(self.inventory, self.budget)
        self.last_turn_prompt = LAST_TURN
        self.events = []                              # [(kind, payload)] for the trace

    @property
    def remaining(self):
        return self.budget - len(self.history)

    @staticmethod
    def format_answer(c, a, b, x, bit):
        return f"{c}({a}, {b}, {x}) = {bit}"

    def query(self, c, a, b, x):
        c = str(c).upper().strip()
        if c not in LABELS or any(int(v) not in (0, 1) for v in (a, b, x)):
            raise ValueError(f"bad query {c}({a}, {b}, {x})")
        if self.remaining <= 0:
            raise BudgetExhausted(f"{self.budget} queries already used")
        q = LABELS.index(c) * (1 << K) + (int(a) << 2) + (int(b) << 1) + int(x)
        bit = int(self._truth[q])
        self.history.append(((c, int(a), int(b), int(x)), bit))
        self.events.append(("query", {"c": c, "a": int(a), "b": int(b), "x": int(x), "q": q, "bit": bit}))
        return bit

    # ---- controller-side scoring (never called by a harness) ----
    def score(self, guess_text):
        bits = re.findall(r"[01]{24}", re.sub(r"\s", "", guess_text or ""))
        guess = np.array([int(ch) for ch in bits[-1]], dtype=np.uint8) if bits else None
        # the benchmark's own metric (circuit-wise accuracy, MTP Table 1) and the whole joint truth table
        joint = guess is not None and bool((guess == self._truth).all())
        circuit_wise = (sum(int((guess[i * 8:(i + 1) * 8] == self._truth[i * 8:(i + 1) * 8]).all()) for i in range(C)) / C
                        if guess is not None else 0.0)
        return {"parsed_guess": guess is not None, "joint_truth_table_correct": joint, "circuit_wise_accuracy": circuit_wise,
                "guess": "".join(map(str, guess)) if guess is not None else None}


# --------------------------------------------------------------------------- solver

class Solver:
    """callable(messages, temperature, max_tokens) -> str, cached on disk."""

    def __init__(self, base_url=None, model=None, cache_dir=None, mock=None, mock_game=None, seed=0):
        self.base_url, self.model = base_url, model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mock, self.mock_game, self.seed = mock, mock_game, seed
        self._client = None
        self._seq = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens_in = self.tokens_out = 0

    def _key(self, messages, temperature, max_tokens, seq):
        h = hashlib.sha1(json.dumps([messages, temperature, max_tokens, seq, self.model, self.seed],
                                    sort_keys=True).encode()).hexdigest()
        return h

    def __call__(self, messages, temperature=0.7, max_tokens=1200):
        with self._lock:
            sig = json.dumps([messages, temperature, max_tokens], sort_keys=True)
            seq = self._seq.get(sig, 0)
            self._seq[sig] = seq + 1
            self.calls += 1
        key = self._key(messages, temperature, max_tokens, seq)
        if self.cache_dir:
            p = self.cache_dir / f"{key}.json"
            if p.exists():
                d = json.loads(p.read_text())
                # a replayed call still costs what it cost: account the stored usage (entries written
                # before 2026-10-12 carry none, which is why the pilots' candidate token columns read 0)
                self.tokens_in += int(d.get("in") or 0)
                self.tokens_out += int(d.get("out") or 0)
                return d["text"]
        u_in = u_out = 0
        if self.mock:
            from cd_mock import MockClient
            import asyncio
            if not hasattr(self, "_mock_client"):
                self._mock_client = MockClient(self.mock, self.mock_game, C, K, seed=self.seed)
            r = asyncio.run(self._mock_client.chat.completions.create(model="mock", messages=messages))
            text = r.choices[0].message.content
        else:
            if self._client is None:
                import httpx
                from openai import OpenAI
                # trust_env=False: the mac's system proxy must not intercept the SSH tunnel on 127.0.0.1
                self._client = OpenAI(base_url=self.base_url, api_key="EMPTY", max_retries=3, timeout=600.0,
                                      http_client=httpx.Client(trust_env=False, timeout=600.0))
            # Llama-3 checkpoints whose generation_config lacks <|eot_id|> run on past the end of turn
            extra = {"stop_token_ids": [128001, 128008, 128009]} if "llama" in (self.model or "").lower() else {}
            r = self._client.chat.completions.create(model=self.model, messages=messages,
                                                     temperature=temperature, max_tokens=max_tokens, seed=self.seed, extra_body=extra)
            text = r.choices[0].message.content or ""
            if r.usage:
                u_in, u_out = r.usage.prompt_tokens or 0, r.usage.completion_tokens or 0
                self.tokens_in += u_in
                self.tokens_out += u_out
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps({"text": text, "in": u_in, "out": u_out}))
        return text


# --------------------------------------------------------------------------- stream

def make_stream(split="heldout", split_file=None, seed=0, limit=None):
    sf = Path(split_file) if split_file else ROOT / "splits/cd_split_v2.json"
    keys = list(range(len(INSTANCES))) if split == "all" else json.loads(sf.read_text())[split]
    rng = np.random.default_rng(seed)
    keys = [int(k) for k in rng.permutation(keys)]
    return keys[:limit] if limit else keys
