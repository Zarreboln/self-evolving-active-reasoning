"""ToT on Situation Puzzles with the state evaluator Yao et al. specify and AR-Bench leaves out.

Tree of Thoughts is defined by four choices (Yao et al. 2023, §3): how to decompose a thought, how
to generate candidates, **how to heuristically evaluate states**, and what search algorithm to use.
AR-Bench implements the first, second and fourth and drops the third: `SearchTreeNode.__init__`
sets `self.value = -1` and nothing ever assigns it again, so in `select_best_child_node` every
child ties at -1 and the choice falls entirely through to the tie-break

    answer_priority = {"Yes": 2, "No": 1, "Unknown": 0}

i.e. "keep the branch whose question got a Yes". There is no heuristic, so there is also nothing to
threshold, which is why the method can never terminate early -- and why BALAR's statement that
"BALAR, UoT, ToT, Proactive CoT, and MediQ Expert may terminate earlier according to their own
stopping behavior" is not true of this implementation.

This adds the missing evaluator in the form the paper gives for it: "a value prompt reasons about
the state s to generate a scalar value v (e.g. 1-10) or a classification (e.g. sure/likely/
impossible) that could be heuristically turned into a value". The classification form is used, with
the weighting from the authors' released Game-of-24 code, sure=20 / likely=1 / impossible=0.001.
A state judged `sure` ends the episode: that is the stop, and it is the paper's own signal rather
than one invented here.

Two properties worth stating because the GN ToT gets them wrong. The evaluator sees the surface and
the question-answer record **only** -- never the hidden bottom -- so unlike `gn/tot`, which scores
each candidate with `compare_guess(target, guess)` before choosing, nothing here consults the
answer. And the oracle budget is the question budget: one referee call per expanded child, counted.

AR-Bench is not modified. `expand_node` and `select_best_child_node` are imported and used
unchanged; the evaluator runs between them, which is exactly the slot the paper describes.

    POLICY_BASE_URL=... RESPONSE_BASE_URL=... python sp_tot_eval.py \
        --data_path data/sp/test.json --output_path results/sp_tot_eval.json \
        --policy_model ... --response_model jev-latest
"""

import json
import os
import re
import sys
import time
from typing import Dict, List

from fire import Fire
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.sp.sp_evaluator import (  # noqa: E402
    RESPONSE_TEMPLATE,
    SearchTreeNode,
    expand_node,
    get_final_prediction,
    select_best_child_node,
)
from arbench.utils.inference import inference  # noqa: E402
from arbench.utils.utils_sp import f1_score as calculate_f1_score  # noqa: E402

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")

# Yao et al. §3: a value prompt that classifies the state. The weighting is the one in the
# authors' Game-of-24 code; the labels are theirs.
VALUE = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}

VALUE_PROMPT = """You are judging how close a line of questioning is to solving a situation puzzle.

Puzzle (what the player can see):
{surface}

Questions asked so far and the referee's answers:
{record}

Judge the state, not the next question. Evaluate whether the answers so far already determine the
explanation.

- sure: the answers pin the explanation down; another question could not change it.
- likely: real progress has been made, but the explanation is not yet determined.
- impossible: this line is ruled out or has stopped yielding anything.

Reply with one short sentence of reasoning, then a final line of exactly:
VALUE: sure
or
VALUE: likely
or
VALUE: impossible"""

VALUE_RE = re.compile(r"VALUE:\s*(sure|likely|impossible)", re.I)


def ask(messages, model, temperature, top_p, tries=3):
    for _ in range(tries):
        r = inference(messages=messages, model=model, temperature=temperature, top_p=top_p,
                      api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        try:
            return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens
        except AttributeError:
            time.sleep(1.5)
    return None, 0, 0


def value_state(node, surface, policy_model, temperature, top_p):
    """Yao et al.'s V(p_theta, S): classify this state, never consulting the hidden bottom."""
    prompt = VALUE_PROMPT.format(surface=surface, record=node.get_record() or "(no questions yet)")
    text, a, b = ask([{"role": "user", "content": prompt}], policy_model, temperature, top_p)
    m = VALUE_RE.search(text or "")
    label = m.group(1).lower() if m else "likely"      # unreadable judgement is not evidence
    return label, VALUE[label], a, b


def main(data_path: str, output_path: str, policy_model: str, response_model: str,
         max_turn: int = 25, branch: int = 3,
         policy_temperature: float = 0.7, policy_top_p: float = 0.7,
         response_temperature: float = 0.7, response_top_p: float = 0.7) -> None:
    with open(data_path) as f:
        dataset = json.load(f)
    logs: List[Dict] = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            logs = json.load(f)
    print(f"Policy: {policy_model}  Referee: {response_model}  max_turn={max_turn} branch={branch}"
          f"  state evaluator: sure/likely/impossible (Yao et al. restored)")

    for i in tqdm(range(len(logs), len(dataset))):
        surface, bottom = dataset[i]["surface"], dataset[i]["bottom"]
        log_info = {"policy_model": policy_model, "response_model": response_model}
        root = SearchTreeNode(None)
        root.question = surface
        node = root
        in_tok = out_tok = referee_calls = 0
        committed = False
        turns = 0
        chosen_labels = []      # what the evaluator said about the branch actually taken

        for depth in range(max_turn):
            node.chosen = True
            children = expand_node(node, depth=depth, max_depth=max_turn,
                                   respond_template=RESPONSE_TEMPLATE, surface=surface,
                                   bottom=bottom, log_info=log_info, branch=branch,
                                   policy_temperature=policy_temperature,
                                   policy_top_p=policy_top_p,
                                   response_temperature=response_temperature,
                                   response_top_p=response_top_p)
            if not children:
                break
            referee_calls += len(children)
            for c in children:
                in_tok += getattr(c, "input_token", 0)
                out_tok += getattr(c, "output_token", 0)

            # the slot AR-Bench leaves empty
            labels = {}
            for c in children:
                lab, val, a, b = value_state(c, surface, policy_model,
                                             policy_temperature, policy_top_p)
                c.value = val
                labels[id(c)] = lab
                in_tok, out_tok = in_tok + a, out_tok + b

            best = select_best_child_node(node)     # AR-Bench's own rule, now with real values
            if best is None:
                break
            node = best
            turns += 1
            chosen_labels.append(labels.get(id(best)))
            if labels.get(id(best)) == "sure":
                committed = True
                break

        pred, a, b, _ = get_final_prediction(node, policy_model, policy_temperature, policy_top_p)
        in_tok, out_tok = in_tok + a, out_tok + b
        pred = pred or ""

        logs.append({"idx": i, "question": surface, "answer": bottom, "pred": pred,
                     "f1_score_char": calculate_f1_score(pred, bottom),
                     "f1_score_word": calculate_f1_score(pred.split(), bottom.split()),
                     "round": turns, "stopped_early": committed,
                     "value_labels": chosen_labels,
                     "referee_calls": referee_calls,     # branch per depth, not one
                     "record": node.record,
                     "input_token": in_tok, "output_token": out_tok})
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    Fire(main)
