"""ToT on Detective Cases with a label-free state evaluator in place of the oracle one.

The companion of runner/sp_tot_eval.py, but the defect it addresses is the opposite one. On
Situation Puzzles AR-Bench has no state evaluator at all (`value` stays -1). On Detective Cases it
has one, and the one it has reads the answer key: `expand_node` builds

    keypoint_hits_prompt.format(question=init_info, name=..., answer=item["story"],
                                keypoints=format_keypoints(item["key_question"]))

and scores each candidate question by how many of the suspect's **hidden** key questions it hit --
`current_node.value = len(parse_keypoints(check_results))`. The search is steered by the gold
annotations, so the branch it keeps is the one an oracle liked, not the one the reasoner judged
promising. ToT is a test-time method; this makes it a supervised one.

This arm removes the leak and restores the evaluator the paper actually specifies. AR-Bench's own
`expand_node` is called with the `key_question` field **removed** from every suspect. The agents
that do the gold scoring are built with `if "key_question" in item`, a presence test, so emptying
the list is not enough -- the agent is still created and still spends a response-model call per
child. Dropping the key makes `keypoint_eval_agents` empty, `selected_suspect in keypoint_eval_agents`
false, and the whole block unreachable: no leak, no wasted call (75 per episode), and `value` stays
at -1 for the evaluator below to fill. (Emptying `key_question_dict` alone does neither: the agents
come from `suspects`, and the block then indexes `key_question_dict[suspect]`.)
The value prompt from Yao et al. §3
(a classification "sure/likely/impossible", weighted as in the authors' Game-of-24 code) then fills
it, reading the case background and the interrogation record and nothing else. `sure` ends the
episode, which is also the early termination BALAR's comparison assumes these methods have and
AR-Bench's never performs.

AR-Bench is not modified. `expand_node` and `select_best_child_node` are imported and used
unchanged.

    POLICY_BASE_URL=... RESPONSE_BASE_URL=... python dc_tot_eval.py \
        --data_path data/dc/test.json --output_path results/dc_tot_eval.json \
        --policy_model ... --response_model ...
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

from arbench.reasoner.dc.dc_evaluator import (  # noqa: E402
    ANSWER_CHOICES,
    CHOICE_TO_INDEX,
    SearchTreeNode,
    _format_initial_info,
    expand_node,
    extract_answer_choice,
    select_best_child_node,
)
from arbench.reasoner.dc.prompt import (  # noqa: E402
    propose_template,
    select_murderer_template_searching,
)
from arbench.utils.inference import inference  # noqa: E402

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")

VALUE = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}

VALUE_PROMPT = """You are judging how close an interrogation is to identifying the murderer.

Case background:
{init_info}

Interrogation so far:
{record}

Judge the state, not the next question. Evaluate whether the answers so far already single out one
suspect.

- sure: the answers identify the murderer; another question could not change who you would name.
- likely: real progress has been made, but more than one suspect is still open.
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


def record_text(node) -> str:
    rows = node.get_conversation_record() if node.parent else []
    if node.parent:
        rows = rows + [{"suspect": node.suspect, "question": node.question, "feedback": node.answer}]
    return "".join(f"Question for {e['suspect']}: {e['question']} Feedback: {e['feedback']}\n"
                   for e in rows) or "(no questions yet)"


def value_state(node, init_info, policy_model, temperature, top_p):
    prompt = VALUE_PROMPT.format(init_info=init_info, record=record_text(node))
    text, a, b = ask([{"role": "user", "content": prompt}], policy_model, temperature, top_p)
    m = VALUE_RE.search(text or "")
    label = m.group(1).lower() if m else "likely"
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
    print(f"Policy: {policy_model}  Suspects: {response_model}  max_turn={max_turn} branch={branch}"
          f"  evaluator: label-free sure/likely/impossible (keypoint leak removed)")

    for i in tqdm(range(len(logs), len(dataset))):
        init_info = _format_initial_info(dataset[i]["initial_information"])
        label = dataset[i]["label"]
        choice_str = ", ".join(f"{x}. {it['name']}" for x, it in
                               zip(ANSWER_CHOICES, dataset[i]["initial_information"]["suspect"]))
        suspect_name_str = ", ".join(it["name"] for it in dataset[i]["initial_information"]["suspect"])
        # the suspects as published, with the gold key questions withheld from the search
        suspects_blind = [{k: v for k, v in s_.items() if k != "key_question"}
                          for s_ in dataset[i]["suspects"]]
        node = SearchTreeNode(None)
        in_tok = out_tok = suspect_calls = 0
        committed, turns, chosen_labels = False, 0, []

        for j in range(max_turn):
            expand_node(node=node, branch=branch, current_round=j + 1, max_round=max_turn,
                        init_info=init_info, suspect_name_str=suspect_name_str,
                        suspects=suspects_blind,         # gold key questions withheld
                        key_question_dict={},            # unreachable once the key is gone
                        model=policy_model, response_model=response_model,
                        policy_temperature=policy_temperature, policy_top_p=policy_top_p,
                        response_temperature=response_temperature, response_top_p=response_top_p)
            if not node.children:
                break
            suspect_calls += len(node.children)
            labels = {}
            for c in node.children:
                in_tok += getattr(c, "input_token", 0)
                out_tok += getattr(c, "output_token", 0)
                lab, val, a, b = value_state(c, init_info, policy_model,
                                             policy_temperature, policy_top_p)
                c.value = val
                labels[id(c)] = lab
                in_tok, out_tok = in_tok + a, out_tok + b
            best = select_best_child_node(node)
            if best is None:
                break
            node = best
            turns += 1
            chosen_labels.append(labels.get(id(best)))
            if labels.get(id(best)) == "sure":
                committed = True
                break

        record_str = record_text(node)
        prompt = (propose_template.format(turn=max_turn, background=init_info) +
                  select_murderer_template_searching.format(record=record_str, choice=choice_str))
        raw, a, b = ask([{"role": "user", "content": prompt}], policy_model,
                        policy_temperature, policy_top_p)
        in_tok, out_tok = in_tok + a, out_tok + b
        pred = CHOICE_TO_INDEX.get(extract_answer_choice(raw or "").strip(), "")

        logs.append({"idx": i, "pred": pred, "label": label,
                     "correctness": pred == label,
                     "round": turns, "stopped_early": committed,
                     "value_labels": chosen_labels,
                     "suspect_calls": suspect_calls,
                     "record": node.get_conversation_record() if node.parent else [],
                     "input_token": in_tok, "output_token": out_tok})
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    Fire(main)
