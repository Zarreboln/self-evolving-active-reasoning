"""AR-Bench's zero_shot agent (GN and DC), no reasoning.

The env plays the user side of AR-Bench's zero-shot dialogue message for message, so the agent is
AR-Bench's: one growing chat, the system prompt first, every env message as a user turn, every reply
as an assistant turn, nothing truncated. Sampling as AR-Bench's `inference`: temperature 0.7,
top_p 0.7, no max_tokens.
Source: AR-Bench arbench/reasoner/{gn,dc}/*_evaluator.py `_run_traditional_evaluation`,
arbench/utils/inference.py.
"""


def run_episode(env, chat):
    messages = [{"role": "system", "content": env.instructions},
                {"role": "user", "content": env.observation}]
    while not env.done:
        reply = chat(messages, temperature=0.7, top_p=0.7, max_tokens=None)
        messages.append({"role": "assistant", "content": reply})
        obs = env.act(reply)
        if not env.done:
            messages.append({"role": "user", "content": obs})
