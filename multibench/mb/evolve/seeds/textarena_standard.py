"""TextArena's standard agent (OpenAIAgent with STANDARD_GAME_PROMPT), used for Wordle, Hangman,
Mastermind, Minesweeper and 20Q.

One stateless call per turn: the system prompt and, as the only user message, the observation string
TextArena's default wrappers give an agent (the full game history so far). Our adapters expose exactly
that string: env.instructions before the first move, env.observation after each move
(mb/benches/{wordle,tagames,twentyq}.py read it from the wrapper's get_observation()). The reply is
stripped and sent whole, as TextArena's loop does. Sampling: OpenAIAgent passes no temperature or
max_tokens, so the server's defaults apply (temperature=None, max_tokens=None here).
Source: textarena 0.7.4, agents/basic_agents.py (STANDARD_GAME_PROMPT, OpenAIAgent._make_request).
"""

STANDARD_GAME_PROMPT = ("You are a competitive game player. Make sure you read the game instructions "
                        "carefully, and always follow the required format.")


def run_episode(env, chat):
    obs = env.instructions
    while not env.done:
        messages = [{"role": "system", "content": STANDARD_GAME_PROMPT}, {"role": "user", "content": obs}]
        reply = chat(messages, temperature=None, max_tokens=None)
        obs = env.act(reply.strip())
