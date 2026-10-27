"""AR-Bench's Guess Number ToT, with the type bug that makes it crash on the first instance fixed.

As published, `gn/tot` cannot run. `ToTSearch.select` does

    valid_numbers = extract_four_digit_numbers(guess_response)
    for number in valid_numbers:
        candidate_node = SearchNode(number, parent=current_node)
        candidate_node.get_value(self.target)

but `extract_four_digit_numbers` is annotated `-> List[str]` and returns a `str`, so the loop
iterates *characters*: each "candidate guess" is "1" or "," or " ", and `compare_guess` raises
`ValueError: Lists must have the same length` on the first instance. All 16 shards died in six
seconds. The function is also wrong for this caller in a second way -- its filter keeps only
captures containing exactly four digits, so a reply holding the three guesses the prompt asks for
("suggest 3 possible four-digit numbers") is discarded outright.

The fix uses a function already in the same file. `_gn_extract_multiple_guesses` returns up to
three guesses as `List[List[int]]`, which is exactly what a tree search needs and what UoT already
calls. Nothing else changes: the same prompts, the same node scoring, the same selection rule, the
same record shape. Guesses are stored as four-character strings so the saved records match every
other Guess Number run and the existing scorers read them unchanged.

AR-Bench is not modified; this is a separate module, as runner/md_evaluator.py and
runner/{sp,dc}_proactive_stop.py are. The main table's "official implementation, unmodified"
property survives, and this cell is reported as ours with the fix named.

One thing worth reading off this run rather than inferring. ToT scores a candidate by
`compare_guess(target, guess)` -- it asks the oracle how good each candidate is *before* choosing
which to play. With three candidates per depth over 25 depths that is ~75 oracle consultations
against a budget of 25 guesses, and the saved record shows only the 25 that were played. This
module counts them, so the asymmetry is a number in the table instead of a footnote.

    POLICY_BASE_URL=... python gn_tot_fixed.py --data_path data/gn/test.json \
        --output_path results/gn_tot_fixed.json --model Qwen/Qwen2.5-7B-Instruct
"""

import json
import os
import random
import sys
import time
from typing import List, Optional, Tuple

from fire import Fire
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.gn.gn_evaluator import _gn_extract_multiple_guesses  # noqa: E402
from arbench.reasoner.gn.prompt import (  # noqa: E402
    Final_prompt,
    Game_rule,
    Guess_number_prompt,
)
from arbench.utils.inference import inference  # noqa: E402
from arbench.utils.utils_gn import compare_guess  # noqa: E402

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
TARGET_FEEDBACK = [4, 0]
MAX_RETRIES = 3


def as_str(guess) -> str:
    return guess if isinstance(guess, str) else "".join(str(d) for d in guess)


class Node:
    """AR-Bench's SearchNode, with the guess kept as a four-character string."""

    def __init__(self, guess: str, parent: Optional["Node"] = None):
        self.guess = as_str(guess)
        self.parent = parent
        self.value = -1.0
        self.feedback = [-1, -1]
        self.children: List["Node"] = []

    def score_against(self, target: List[int]) -> None:
        correct_pos, different_pos, self.value = compare_guess(target, self.guess)
        self.feedback = [correct_pos, different_pos]

    def history(self, target: List[int]) -> Tuple[str, List[List]]:
        chain, node = [], self
        while node.parent:
            chain.append(node.guess)
            node = node.parent
        chain.reverse()
        text, rows = "", []
        for i, g in enumerate(chain):
            cp, dp, _ = compare_guess(target, g)
            text += (f"The guess number {i + 1} is {g}, and the feedback is: "
                     f"{cp} digits are present in the answer and in the correct positions, "
                     f"{dp} digits are present in the answer but in the different positions.\n")
            rows.append([g, cp, dp])
        return text, rows


def ask(messages, model, temperature, top_p, tries=3):
    for _ in range(tries):
        r = inference(messages=messages, model=model, temperature=temperature, top_p=top_p,
                      api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        try:
            return r.choices[0].message.content
        except AttributeError:
            time.sleep(1.5)
    return None


def search(target: List[int], model: str, max_depth: int, temperature: float, top_p: float):
    node = Node("0000")
    oracle_calls = 0

    for _ in range(max_depth):
        if node.feedback == TARGET_FEEDBACK:
            break
        record, _ = node.history(target)
        content, candidates, tries = None, [], 0
        while not candidates and tries < MAX_RETRIES:
            content = ask([{"role": "user",
                            "content": Game_rule + Guess_number_prompt.format(guess_record=record)}],
                          model, temperature, top_p)
            candidates = _gn_extract_multiple_guesses(content or "")
            tries += 1
        if not candidates:
            break

        best = None
        for cand in candidates:
            child = Node(cand, parent=node)
            child.score_against(target)          # ToT's own scoring: it consults the oracle here
            oracle_calls += 1
            node.children.append(child)
            if best is None or child.value > best.value:
                best = child
        node = best

    record, rows = node.history(target)
    ask([{"role": "user", "content": Game_rule + Final_prompt.format(guess_record=record)}],
        model, temperature, top_p)
    return rows, oracle_calls


def main(data_path: str, output_path: str, model: str, max_turn: int = 25,
         temperature: float = 0.7, top_p: float = 0.7) -> None:
    with open(data_path) as f:
        dataset = json.load(f)
    logs: List[dict] = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            logs = json.load(f)
    print(f"Model: {model}  Max turn: {max_turn}  Method: tot (type bug fixed)")

    for i in tqdm(range(len(logs), len(dataset))):
        target = [int(d) for d in str(dataset[i]).zfill(4)]
        rows, oracle_calls = search(target, model, max_turn, temperature, top_p)
        if not rows:
            rows = [["0000", 0, 0]]
        logs.append({"idx": i,
                     "true_number": dataset[i],
                     "guess_list": rows,
                     "guess_round": len(rows),      # real, not max_turn
                     "oracle_calls": oracle_calls,  # consultations, including those never played
                     "correctness": rows[-1][1] == 4})
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    Fire(main)
