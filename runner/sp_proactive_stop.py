"""Proactive CoT on Situation Puzzles, with the answer-now action its own paper defines.

Deng et al. (2023b) define Proactive CoT by a decision over an action space that contains
`Act: ["Directly Answer", "Ask a question"]` -- the system first decides *whether* to ask, then
what. AR-Bench's adaptation keeps the strategy selection and drops the decision: every one of its
seven Situation Puzzles strategies is a kind of asking (FACT_GATHERING, TIMELINE_EXPLORATION,
CHARACTER_INVESTIGATION, LOCATION_PROBING, CAUSE_EFFECT_ANALYSIS, HYPOTHESIS_TESTING,
CONSTRAINT_CLARIFICATION), and the loop runs to the budget on all 100 instances. BALAR's paper
assumes the dropped behaviour is present -- it writes that "BALAR, UoT, ToT, Proactive CoT, and
MediQ Expert may terminate earlier according to their own stopping behavior" -- so its comparison
credits Proactive CoT with a stop that AR-Bench's code does not have.

This restores it. DIRECT_ANSWER is added to the action list and nothing else about the method
changes: same system prompt, same strategy descriptions, same referee, same final-answer prompt,
same scoring. AR-Bench itself is not touched; this is a separate module, the way md_evaluator.py
is, so the official implementation stays byte-identical for the main table and this is a second
column beside it.

One deliberate choice. When the reasoner commits, the final explanation is still produced by
AR-Bench's own final prompt, with the shorter history. Letting the commit text itself be the
prediction would change the answer's *format* as well as the question count, and the metric that
matters here -- bidirectional semantic equivalence -- is sensitive to answer length (measured:
r = -0.47 pooled). Keeping the final prompt fixed isolates the one variable this arm is about.

    POLICY_BASE_URL=... RESPONSE_BASE_URL=... python sp_proactive_stop.py \
        --data_path data/sp/test.json --output_path results/sp_proactive_stop.json
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

from fire import Fire
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.sp.prompt import (  # noqa: E402
    proactive_cot_question_prompt_template,
    proactive_cot_system_prompt,
    proactive_cot_system_prompt_final_guess,
    system_prompt_with_2shots,
)
from arbench.reasoner.sp.sp_evaluator import _extract_question_from_text, _format_history  # noqa: E402
from arbench.utils.inference import inference  # noqa: E402
from arbench.utils.utils_sp import f1_score as calculate_f1_score  # noqa: E402

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")

# The one addition, in the shape of the strategies already in the list.
DIRECT_ANSWER = """
8. DIRECT_ANSWER: Stop asking and give the solution now, because the answers so far already
   determine it. Choose this only when another question would not change your explanation.
"""
COMMIT_INSTRUCTION = """

If you select DIRECT_ANSWER, do not ask a question. Write exactly this line instead:
DECISION: DIRECT_ANSWER
"""


def stoppable_template() -> str:
    """AR-Bench's own template with DIRECT_ANSWER appended to the action list."""
    t = proactive_cot_question_prompt_template
    anchor = "Based on this analysis, select the most appropriate action strategy for this turn:"
    if anchor not in t:
        raise SystemExit("AR-Bench's proactive_cot template changed shape; re-read it before using this")
    return t.replace(anchor, DIRECT_ANSWER + "\n" + anchor) + COMMIT_INSTRUCTION


def ask(messages, model, temperature, top_p, api_key, base_url, tries=3):
    for _ in range(tries):
        r = inference(messages=messages, model=model, temperature=temperature, top_p=top_p,
                      api_key=api_key, base_url=base_url)
        try:
            return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens
        except AttributeError:
            time.sleep(1.5)
    return None, 0, 0


def main(data_path: str, output_path: str, policy_model: str, response_model: str,
         max_turn: int = 25, policy_temperature: float = 0.7, policy_top_p: float = 0.7,
         response_temperature: float = 0.7, response_top_p: float = 0.7) -> None:
    template = stoppable_template()
    with open(data_path) as f:
        dataset = json.load(f)
    logs: List[Dict] = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            logs = json.load(f)

    print(f"Policy: {policy_model}  Referee: {response_model}  max_turn={max_turn}  "
          f"action space: AR-Bench's seven + DIRECT_ANSWER")

    for i in tqdm(range(len(logs), len(dataset))):
        surface, bottom = dataset[i]["surface"], dataset[i]["bottom"]
        question_record: List[Dict] = []
        in_tok = out_tok = 0
        committed = False

        for turn in range(1, max_turn + 1):
            prompt = template.format(puzzle=surface, history=_format_history(question_record),
                                     turn=turn, max_turn=max_turn)
            content, a, b = ask([{"role": "system", "content": proactive_cot_system_prompt},
                                 {"role": "user", "content": prompt}],
                                policy_model, policy_temperature, policy_top_p,
                                POLICY_API_KEY, POLICY_BASE_URL)
            in_tok, out_tok = in_tok + a, out_tok + b
            if content is None:
                break

            if "DECISION: DIRECT_ANSWER" in content:
                committed = True
                break

            question = _extract_question_from_text(content)
            if not question:
                continue

            answer, a, b = ask([{"role": "system",
                                 "content": system_prompt_with_2shots.format(question=surface, answer=bottom)},
                                {"role": "user", "content": question}],
                               response_model, response_temperature, response_top_p,
                               RESPONSE_API_KEY, RESPONSE_BASE_URL)
            in_tok, out_tok = in_tok + a, out_tok + b
            question_record.append({"question": question, "answer": answer or "Unknown"})

        # AR-Bench's own final prompt, unchanged, on whatever history was gathered.
        final_prompt = (
            f"Puzzle: {surface}\n\n"
            f"Previous Questions and Answers:\n{_format_history(question_record)}\n\n"
            f"Based on all the questions and answers above, what do you think is the solution to "
            f"this puzzle? Please provide a detailed explanation.\n\n"
            f"Your answer:"
        )
        pred, a, b = ask([{"role": "system", "content": proactive_cot_system_prompt_final_guess},
                          {"role": "user", "content": final_prompt}],
                         policy_model, policy_temperature, policy_top_p,
                         POLICY_API_KEY, POLICY_BASE_URL)
        pred = pred or ""
        in_tok, out_tok = in_tok + a, out_tok + b

        logs.append({"idx": i, "question": surface, "answer": bottom, "pred": pred,
                     "f1_score_char": calculate_f1_score(pred, bottom),
                     "f1_score_word": calculate_f1_score(pred.split(), bottom.split()),
                     "round": len(question_record),      # real, not max_turn
                     "stopped_early": committed,
                     "record": question_record,
                     "input_token": in_tok, "output_token": out_tok})
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    Fire(main)
