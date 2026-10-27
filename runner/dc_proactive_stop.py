"""Proactive CoT on Detective Cases, with the answer-now action its own paper defines.

The companion of runner/sp_proactive_stop.py; see that file for the argument. In short: Deng et al.
(2023b) define Proactive CoT by a decision over an action space containing `Act: ["Directly
Answer", "Ask a question"]`, AR-Bench's adaptation keeps the strategy selection and drops the
decision -- all three of its Detective Cases strategies are ways of asking (GATHER INFORMATION,
FOLLOW UP, CROSS-REFERENCE) -- and BALAR's comparison nevertheless credits Proactive CoT with a
stop, writing that it "may terminate earlier according to their own stopping behavior".

Guess Number needs no such variant. There, every turn already *is* an answer attempt, the oracle
says `same_pos == 4` the moment a guess is right, and AR-Bench already breaks on that. A separate
"I am confident, stop now" action could only end the episode on a wrong guess, which is strictly
worse. The ask-versus-answer decision exists only where asking and answering are different acts:
Situation Puzzles and Detective Cases.

AR-Bench is not modified. This copies its loop, adds DIRECT_ANSWER to the strategy list, and
leaves everything else -- suspect selection, the referee, the final-decision prompt and its retry,
the record shape and the scoring -- exactly as published. When the reasoner commits, the final
decision is still produced by AR-Bench's own final prompt on the shorter record, so the arms differ
in how many questions were asked and in nothing else.

    POLICY_BASE_URL=... RESPONSE_BASE_URL=... python dc_proactive_stop.py \
        --data_path data/dc/test.json --output_path results/dc_proactive_stop.json \
        --policy_model ... --response_model ...
"""

import json
import os
import random
import re
import sys
import time
from typing import Dict, List

from fire import Fire
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.dc.dc_evaluator import (  # noqa: E402
    CHOICE_TO_INDEX,
    _extract_choice_letter,
    _extract_selected_suspect,
    _format_initial_info,
    _place_question_records,
)
from arbench.reasoner.dc.prompt import (  # noqa: E402
    proactive_cot_question_template,
    proactive_cot_select_suspect_template,
    proactive_cot_system_prompt,
    proactive_cot_system_prompt_conclusion,
    respond_template,
)
from arbench.utils.inference import inference  # noqa: E402

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")

DIRECT_ANSWER = """4. DIRECT_ANSWER: Stop interrogating and name the murderer now, because the
   answers so far already identify them. Choose this only when another question would not change
   who you would name.
"""
COMMIT_INSTRUCTION = """

If you select DIRECT_ANSWER, do not ask a question. Write exactly this line instead:
DECISION: DIRECT_ANSWER
"""


def stoppable_template() -> str:
    t = proactive_cot_question_template
    anchor = "Based on your analysis of this case, which strategy is the best?"
    if anchor not in t:
        raise SystemExit("AR-Bench's dc proactive_cot template changed shape; re-read it first")
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
    print(f"Policy: {policy_model}  Suspects: {response_model}  max_turn={max_turn}  "
          f"action space: AR-Bench's three + DIRECT_ANSWER")

    for i in tqdm(range(len(logs), len(dataset))):
        init_info_dict = dataset[i]["initial_information"]
        init_info = _format_initial_info(init_info_dict)
        label = dataset[i]["label"]
        response_agents = {
            item["name"]: [{"role": "system",
                            "content": respond_template.format(name=item["name"], task=item["task"],
                                                               story=item["story"])}]
            for item in dataset[i]["suspects"]
        }
        question_record: List[Dict[str, str]] = []
        suspect_list_str = ", ".join(s["name"] for s in init_info_dict["suspect"])
        in_tok = out_tok = 0
        committed = False

        for turn in range(1, max_turn + 1):
            sel_prompt = proactive_cot_select_suspect_template.format(
                init_info=init_info, history=_place_question_records(question_record),
                turn=turn, max_turn=max_turn, suspect_list=suspect_list_str)
            sel, a, b = ask([{"role": "system", "content": proactive_cot_system_prompt},
                             {"role": "user", "content": sel_prompt}],
                            policy_model, policy_temperature, policy_top_p,
                            POLICY_API_KEY, POLICY_BASE_URL)
            in_tok, out_tok = in_tok + a, out_tok + b
            if sel is None:
                break
            suspect = _extract_selected_suspect(sel)
            if not suspect or suspect not in response_agents:
                matched = [n for n in response_agents
                           if suspect and (suspect.lower() in n.lower() or n.lower() in suspect.lower())]
                suspect = matched[0] if matched else random.choice(list(response_agents))

            ask_prompt = template.format(
                init_info=init_info, history=_place_question_records(question_record),
                turn=turn, max_turn=max_turn, selected_suspect=suspect)
            content, a, b = ask([{"role": "system", "content": proactive_cot_system_prompt},
                                 {"role": "user", "content": ask_prompt}],
                                policy_model, policy_temperature, policy_top_p,
                                POLICY_API_KEY, POLICY_BASE_URL)
            in_tok, out_tok = in_tok + a, out_tok + b
            if content is None:
                break
            if "DECISION: DIRECT_ANSWER" in content:
                committed = True
                break

            m = re.search(r"Question:\s*(.+)", content)
            question = m.group(1).strip() if m else content.strip().split("\n")[-1].strip()
            if not question:
                continue

            response_agents[suspect].append({"role": "user", "content": question})
            feedback, a, b = ask(response_agents[suspect], response_model,
                                 response_temperature, response_top_p,
                                 RESPONSE_API_KEY, RESPONSE_BASE_URL)
            in_tok, out_tok = in_tok + a, out_tok + b
            feedback = feedback or ""
            response_agents[suspect].append({"role": "assistant", "content": feedback})
            question_record.append({"suspect": suspect, "question": question, "feedback": feedback})

        # AR-Bench's final decision, unchanged, on whatever record was gathered.
        index_list = ["A", "B", "C", "D", "E"]
        choice_str = ", ".join(f"{x}. {it['name']}" for x, it in zip(index_list, init_info_dict["suspect"]))
        final_prompt = (
            f"Initial case information:\n{init_info}\n\n"
            f"Interrogation record:\n{_place_question_records(question_record)}\n\n"
            f"Based on your investigation, who do you believe is the true murderer from these "
            f"suspects?\n{choice_str}\n\n"
            f"Provide your reasoning and then your final answer in the format:\n"
            f"Reason: [Your detailed analysis]\nAnswer: [A, B, C, D, or E]")
        final_messages = [{"role": "system", "content": proactive_cot_system_prompt_conclusion},
                          {"role": "user", "content": final_prompt}]
        reasoning, a, b = ask(final_messages, policy_model, policy_temperature, policy_top_p,
                              POLICY_API_KEY, POLICY_BASE_URL)
        reasoning = reasoning or ""
        in_tok, out_tok = in_tok + a, out_tok + b

        ans = _extract_choice_letter(reasoning).strip()
        pred, success, attempts = "", False, 3
        while not success and attempts > 0 and ans:
            attempts -= 1
            pred = CHOICE_TO_INDEX.get(ans, "")
            if pred in [0, 1, 2, 3, 4]:
                success = True
                break
            final_messages += [{"role": "assistant", "content": reasoning},
                               {"role": "user", "content": "Based on your analysis above, who is "
                                "the murderer? Please answer only with the letter A, B, C, D, or E."}]
            retry, a, b = ask(final_messages, policy_model, policy_temperature, policy_top_p,
                              POLICY_API_KEY, POLICY_BASE_URL)
            in_tok, out_tok = in_tok + a, out_tok + b
            ans = _extract_choice_letter(retry or "").strip()
            if not ans:
                mm = re.search(r"\b[A-E]\b", retry or "")
                ans = mm.group(0) if mm else ""
        if not success:
            pred = random.randint(0, 4)

        logs.append({"idx": i, "pred": pred, "label": label, "record": question_record,
                     "reasoning": reasoning, "round": len(question_record),
                     "stopped_early": committed, "correctness": pred == label,
                     "input_token": in_tok, "output_token": out_tok})
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    Fire(main)
