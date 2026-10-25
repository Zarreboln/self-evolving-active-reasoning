"""iCraft-MD as a fourth active-reasoning task, in AR-Bench's shape and outside AR-Bench's code.

Why this task. The three AR-Bench tasks each have a measurement problem we have now documented.
Situation Puzzles is scored by character-level F1 against a reference paragraph, which correlates
0.806 with answer length and gives 0.637 to a prediction that names the wrong cause of death. Guess
Number scores `guess_list[-1]` for the few-shot family, which never stops early, so an episode that
found the number at turn 11 and kept guessing is recorded as wrong -- and a 7B reasoner sits at the
floor there anyway (0-2 of 100). Circuit Decoding's circuit-wise accuracy is clean but is one task.

iCraft-MD (BALAR, arXiv 2605.05386, `data/mediQ/all_craft_md.jsonl`, 140 instances) has neither
pathology: the reasoner picks one of four diagnoses and BALAR scores it by case-insensitive exact
match against the gold option, with no judge, no entailment model and no length confound. A four-way
choice also has a 25 % floor, so unlike Guess Number it cannot collapse to zero and leave the
methods indistinguishable. And it moves domain -- medical history-taking rather than number games,
puzzles or circuits -- which is the out-of-family axis the claim needs.

Why this file is ours and not a fourth package under `arbench/`. Everything we report on AR-Bench
runs its official evaluators unmodified; that is the point of the Jev shim, which speaks the
OpenAI API so the benchmark needs no patch. Adding a task inside the package would spend that
property for convenience. This module lives in our tree, imports AR-Bench's `inference` helper, and
mirrors the SP and DC command line and record shape so the existing sharding, merging and audit
tooling applies without a special case.

Two deviations from BALAR, both deliberate and both visible in the numbers:

* The patient is the same served model as the physician, one more role on one endpoint, where
  BALAR's example configuration uses gpt-5-nano for both roles. BALAR's patient prompt is kept
  verbatim, but it answers in plain text rather than the strict JSON schema BALAR parses with
  pydantic, because we are not running BALAR's belief machinery and a per-turn JSON contract on a
  7B only adds a failure mode.
* The question budget defaults to 25 turns, matching our other three tasks, where BALAR uses
  `--max_ask_rounds 10` for this dataset. Pass `--max_turn 10` to match BALAR instead.

    PYTHONPATH=~/code/AR-Bench python md_evaluator.py \
      --method few_shot --data_path data/md/test.json --output_path results/md_few_shot.json \
      --policy_model Qwen/Qwen2.5-7B-Instruct --response_model Qwen/Qwen2.5-7B-Instruct
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from fire import Fire
from tqdm import tqdm

from arbench.utils.inference import inference

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")


# --------------------------------------------------------------------------- #
# The physician. Three system prompts for AR-Bench's three traditional methods:
# nothing, one worked exemplar, and the exemplar plus explicit instructions.
# --------------------------------------------------------------------------- #

_BASE = """You are a physician taking a history from a patient. You may ask {turn} questions.
In each turn, ask exactly one question about the patient's history, symptoms, exposures or
examination findings, and the patient will answer from their medical record. Begin your question
with "Q: " and ask one question only.

After the last turn you will be asked to choose the most likely diagnosis from a fixed list of
options. Use your questions to separate those options from each other."""

_EXEMPLAR = """

Here is one turn of an example consultation:

Q: Has the rash appeared anywhere other than the hands and feet?
A: The rash is limited to the palms and the soles; there is nothing on the trunk.
Q: Was there a sore anywhere before the rash started?
A: There was a single painless ulcer on the genitals about six weeks ago that healed on its own."""

_INSTRUCTIONS = """

Guidance for choosing questions:
- Ask about the finding that would most change which option is correct, not about what you already
  know from the information you were given.
- One question per turn. Do not combine several questions into one sentence.
- A patient answers only what was asked, so ask narrowly.
- Do not state your diagnosis until you are asked for it."""

METHOD_PROMPT = {
    "zero_shot": _BASE,
    "few_shot": _BASE + _EXEMPLAR,
    "few_shot_inst": _BASE + _EXEMPLAR + _INSTRUCTIONS,
}

TURN_PROMPT = "Turn {turn}: please ask your next question."

FINAL_PROMPT = """You have no questions left. Choose the most likely diagnosis.

{options}

Reply with exactly one line, in the form "Answer: X", where X is the letter of your choice."""


# --------------------------------------------------------------------------- #
# The patient. BALAR's user-simulator prompt (prompts/medical.py), kept verbatim,
# with its free-text answer variant rather than its strict JSON schema.
# --------------------------------------------------------------------------- #

PATIENT_SYSTEM = ("You are simulating a patient being examined by a physician. Answer questions "
                  "based ONLY on your medical record and symptoms. Use facts from your medical "
                  "record. Do not give information that was not specifically asked about.")

PATIENT_USER = """<USER_CONTEXT>
{context}
</USER_CONTEXT>

<QUESTION>
{question}
</QUESTION>

Task:
You are a patient being examined by a physician. Answer <QUESTION> in a way that is consistent with
your medical record and symptoms described in <USER_CONTEXT>. Use facts from your medical record. Do
not give information that was not specifically asked about.

Constraints:
Use ONLY information that is supported by <USER_CONTEXT>.
Answer in one or two sentences, as the patient, with no preamble."""


def patient_context(row: Dict) -> str:
    """What the patient knows: the public demographics plus the case's key facts.

    `facts` is the record BALAR's `use_key_facts=True` restricts the simulator to, so the patient
    cannot invent findings the case does not contain.
    """
    p = row.get("patient") or {}
    out = ["The patient information is :",
           f" - Age : {p.get('age', 'unknown')}",
           f" - Gender : {p.get('gender', 'unknown')}"]
    ctx = row.get("context") or []
    if ctx:
        out.append(f"- Chief complaint : {ctx[0]}")
    out.append("")
    out.append("Their medical record is :")
    out.extend(row.get("facts") or [])
    out.append("")
    out.append("If asked any question, ALWAYS respond with facts from your medical record ONLY. "
               "Do NOT USE any other information.")
    return "\n".join(out)


def physician_opening(row: Dict) -> str:
    """What the physician starts with: the question, the options, and the public presentation."""
    p = row.get("patient") or {}
    ctx = row.get("context") or []
    opts = "\n".join(f"{k}. {v}" for k, v in (row.get("options") or {}).items())
    lines = [row.get("question", "Which of the following is the most likely diagnosis?"), "",
             "Options:", opts, "",
             "Patient information:",
             f"- Age : {p.get('age', 'unknown')}",
             f"- Gender : {p.get('gender', 'unknown')}"]
    if ctx:
        lines.append(f"- Chief complaint : {ctx[0]}")
    return "\n".join(lines)


def text_of(response) -> Optional[str]:
    """AR-Bench's `inference` returns the response object, or the string "{}" when the call failed."""
    try:
        return response.choices[0].message.content
    except AttributeError:
        return None


def tokens_of(response) -> Tuple[int, int]:
    try:
        return response.usage.prompt_tokens, response.usage.completion_tokens
    except AttributeError:
        return 0, 0


CHOICE = re.compile(r"answer\s*[:\-]?\s*\(?([A-Za-z])\)?", re.I)


def parse_choice(reply: str, options: Dict[str, str]) -> Optional[str]:
    """Return the chosen option's text, which is what BALAR's exact match compares.

    The letter is read first because that is what the final prompt asks for. If the reply states a
    diagnosis without the letter, fall back to the option whose text appears in it -- the longest
    such option, so that an option name containing another one is not misread.
    """
    m = CHOICE.search(reply or "")
    if m:
        key = m.group(1).upper()
        if key in options:
            return options[key]
    low = (reply or "").lower()
    hits = [v for v in options.values() if v.lower() in low]
    return max(hits, key=len) if hits else None


def run_episode(row: Dict, system_prompt: str, policy_model: str, response_model: str,
                max_turn: int, policy_temperature: float, policy_top_p: float,
                response_temperature: float, response_top_p: float) -> Dict:
    context = patient_context(row)
    options = row.get("options") or {}
    physician = [{"role": "system", "content": system_prompt.format(turn=max_turn)},
                 {"role": "user", "content": physician_opening(row)}]
    record: List[Dict[str, str]] = list(physician)
    in_tok = out_tok = 0
    asked = 0

    for turn in range(1, max_turn + 1):
        physician.append({"role": "user", "content": TURN_PROMPT.format(turn=turn)})
        record.append(physician[-1])
        r = inference(messages=physician, model=policy_model, temperature=policy_temperature,
                      top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        question = text_of(r)
        a, b = tokens_of(r)
        in_tok, out_tok = in_tok + a, out_tok + b
        if question is None:
            physician.pop()
            record.pop()
            break
        physician.append({"role": "assistant", "content": question})
        record.append(physician[-1])
        asked += 1

        r = inference(messages=[{"role": "system", "content": PATIENT_SYSTEM},
                                {"role": "user", "content": PATIENT_USER.format(
                                    context=context, question=question)}],
                      model=response_model, temperature=response_temperature,
                      top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
        answer = text_of(r)
        a, b = tokens_of(r)
        in_tok, out_tok = in_tok + a, out_tok + b
        answer = (answer or "The record does not say.").strip()
        physician.append({"role": "user", "content": f"A: {answer}"})
        record.append(physician[-1])

    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    physician.append({"role": "user", "content": FINAL_PROMPT.format(options=opts)})
    record.append(physician[-1])
    r = inference(messages=physician, model=policy_model, temperature=policy_temperature,
                  top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
    final = text_of(r) or ""
    a, b = tokens_of(r)
    in_tok, out_tok = in_tok + a, out_tok + b
    record.append({"role": "assistant", "content": final})

    pred = parse_choice(final, options)
    gold = row.get("answer", "")
    return {"question": row.get("question"),
            "options": options,
            "record": record,
            "final": final,
            "pred": pred,
            "answer": gold,
            # BALAR's score_answer: case-insensitive exact match on the option text.
            "correctness": bool(pred) and pred.strip().lower() == gold.strip().lower(),
            "parsed": pred is not None,
            "round": asked,
            "input_token": in_tok,
            "output_token": out_tok}


def main(method: str, data_path: str, output_path: str, policy_model: str, response_model: str,
         max_turn: int = 25, policy_temperature: float = 0.7, policy_top_p: float = 0.7,
         response_temperature: float = 0.7, response_top_p: float = 0.7) -> None:
    if method not in METHOD_PROMPT:
        raise SystemExit(f"method must be one of {sorted(METHOD_PROMPT)}; got {method!r}")
    system_prompt = METHOD_PROMPT[method]

    with open(data_path) as f:
        dataset = json.load(f)
    logs: List[Dict] = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            logs = json.load(f)

    print(f"Policy Model: {policy_model}, Patient Model: {response_model}")
    print(f"Max turn: {max_turn}, Method: {method}, instances: {len(dataset)}")

    # Resume from the log, exactly as AR-Bench's evaluators do, so a shard can be re-entered.
    for i in tqdm(range(len(logs), len(dataset))):
        rec = run_episode(dataset[i], system_prompt, policy_model, response_model, max_turn,
                          policy_temperature, policy_top_p,
                          response_temperature, response_top_p)
        rec["idx"] = i
        logs.append(rec)
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    Fire(main)
