"""BALROG's `naive` agent on BabyAI (BALROG config.yaml defaults), as run by mb/official/run_balrog.py.

What the model sees, call for call, is what BALROG's stack sends:
  * HistoryPromptBuilder(max_text_history=16, max_image_history=0): the instruction prompt as a
    *user* message, then the last 32 events (deque maxlen = 2 * max(16, 0)) alternating
    observation (user) and previous action (assistant). Every observation but the last is
    labelled "Observation:\\n<text>", the last "Current Observation:\\n<text>" (BabyAI's
    short-term context is empty, so nothing else is inserted); only the newest 16 observations
    carry their text, an older one left in the 32-event window is just "Observation:".
  * NaiveAgent appends "\\n\\n" + its instruction to the last (user) message.
  * OpenAIWrapper (client vllm): temperature 1.0, max_tokens 8192, nothing else (no top_p, no
    stop); completion = content.strip().
  * NaiveAgent._extract_final_answer keeps only [a-zA-Z\\s:]; that string is the action sent to the
    env AND the action recorded in history (Evaluator sets `action = response.completion` after
    stepping). An empty filtered string is falsy, so `if prev_action:` skips it and the next
    observation follows the previous one directly.
  * Invalid actions (no exact match) are replaced with "go forward" and the feedback text is put
    in front of the next observation -- both done by mb/benches/babyai.py exactly as BALROG's
    EnvWrapper.check_action_validity + Evaluator.run_episode do.

The adapter gives the instruction prompt and the first observation joined as
`prompt + "\\n\\nCurrent Observation:\\n" + obs`; they are split back here (the prompt never contains
that marker). Remaining differences (see check_equiv.py): BALROG sends each message's content as a
one-element list of text parts, here a plain string (vLLM renders both the same for a text-only
part); BALROG retries an *empty* model reply up to 5 times and then errors the episode out, here
the reply is resampled up to 5 times and then treated as empty (-> default action).
Source: BALROG balrog/agents/naive.py, balrog/prompt_builder/history.py, balrog/client.py,
balrog/evaluator.py (run_episode), balrog/config/config.yaml.
"""
import re

NAIVE_INSTRUCTION = ("You always have to output one of the above actions at a time and no other text. "
                     "You always have to output an action until the episode terminates.")
MARK = "\n\nCurrent Observation:\n"
MAX_TEXT_HISTORY = 16
MAX_EVENTS = 2 * MAX_TEXT_HISTORY      # deque(maxlen=max(max_text_history, max_image_history) * 2)
TEMPERATURE = 1.0
MAX_TOKENS = 8192
MAX_RETRIES = 5


def build_messages(system_prompt, events):
    msgs = []
    if system_prompt:
        msgs.append({"role": "user", "content": system_prompt})
    window = events[-MAX_EVENTS:]
    # only the last MAX_TEXT_HISTORY observations keep their text; an older one still in the
    # window (possible when an empty action left no history entry) is sent as its label alone
    n_obs = sum(1 for kind, text in window if kind == "obs")
    seen = 0
    for i, (kind, text) in enumerate(window):
        if kind == "obs":
            seen += 1
            parts = ["Current Observation:" if i == len(window) - 1 else "Observation:"]
            if n_obs - seen < MAX_TEXT_HISTORY:
                parts.append(text)
            msgs.append({"role": "user", "content": "\n".join(parts)})
        else:
            msgs.append({"role": "assistant", "content": text})
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] += "\n\n" + NAIVE_INSTRUCTION
    return msgs


def filter_letters(s):
    return re.sub(r"[^a-zA-Z\s:]", "", s)


def run_episode(env, chat):
    k = env.instructions.find(MARK)
    if k < 0:                           # not the BabyAI adapter's layout: treat it all as the prompt
        system_prompt, first = env.instructions, env.observation
    else:
        system_prompt, first = env.instructions[:k], env.instructions[k + len(MARK):]
    events = [("obs", first)]
    while not env.done:
        msgs = build_messages(system_prompt, events)
        completion = ""
        for attempt in range(MAX_RETRIES):
            completion = (chat(msgs, temperature=TEMPERATURE, max_tokens=MAX_TOKENS) or "").strip()
            if completion:
                break
        action = filter_letters(completion)
        obs = env.act(action)
        if action:
            events.append(("act", action))
        events.append(("obs", obs))
