# Improve a game-playing agent's harness

You are improving the harness of an agent that plays a text game. The harness is everything around its model
(`{model}`, a frozen open model with its own thinking mode switched off): the prompts, how observations and history
are shown to the model, the control flow, any extra model calls, and any code that tracks state or helps decide.
In each episode the harness gets the game's instructions, sends actions, and gets one text observation back per
action. When the game ends it gives the episode a score in [0, 1].

## Your workspace

- `harness/harness.py` is the harness you improve, a copy of candidate `{parent}`. Edit it in place.
- `archive/summary.md` lists every candidate so far with its parent and its score.
- `archive/<id>/` holds one candidate. `harness/harness.py` is its code and `proposal.md` says what its proposer
  changed and why. `meta.json` has its parent, and the reason if it was rejected. `scores.json` has its mean score and
  the result of every episode. `episodes/<episode>/trace.md` shows every message the model saw, every reply, every
  action and every observation of one episode, and `result.json` next to it has that episode's result.
- `harness_lib.py` holds helpers you may import. `mock_env.py` runs `harness/harness.py` against a fake game and a fake
  model, to check that it runs end to end. `check_harness.py` applies the static rules below.

## The interface

`harness/harness.py` must define `run_episode(env, chat)`. It is called once per episode and should return when
`env.done` is true.

- `env.instructions` (str): the game's instructions (in a chat, the system prompt), shown before the first action.
- `env.observation` (str): the latest observation; before the first action, the game's first prompt if it has one, else empty.
- `env.act(action: str) -> str`: take one action, return the next observation. Also updates `env.observation`,
  `env.done` and `env.steps`. Raises an exception if the episode is already over.
- `env.max_steps` (int): the episode ends after this many actions at the latest.
- `env.admissible()`: the valid actions right now if the game lists them, else None.
- `env.reads_full_reply` (bool): if true, the game parses the action out of whatever text you pass to `env.act`
  (as the seed does, it is fine to pass the model's whole reply minus any blocks you add).
- `chat(messages, temperature=0.7, max_tokens=8192, top_p=None, stop=None) -> str`: one call to the model; `messages` is an OpenAI-style
  list of `{{"role": ..., "content": ...}}`. The server's context is 64K tokens; a longer prompt raises an exception.
  At most {chat_cap} calls per action on average over an episode; beyond that `chat` raises.

The harness runs in its own process that sees only the above: it cannot read files, the network, or the game's
internals. Allowed imports: `re`, `json`, `math`, `itertools`, `collections`, `random`, `string`, `functools`,
`heapq`, `typing`, `dataclasses`, `harness_lib`. Names and attributes starting with `_`, and builtins such as `open`,
`eval`, `exec`, `getattr`, `type`, are rejected. If the harness raises or returns early, the episode keeps the score
it has at that point.

## How to read the evidence

The score alone will not tell you what to change, and it is noisier than it looks. Each candidate is scored on
{n} episodes, one per game instance, and the model samples at temperature 0.7, so the same harness can win an
instance once and lose it the next time. The standard error of the parent's score is about {noise:.3f}. Two
candidates within about twice that of each other have not been told apart, and a change worth making is one that
could plausibly move several episodes.

Before you decide what to change, look past the mean:

1. In the parent's `scores.json`, tabulate every episode's `score`, `success`, `steps`, `end`, `error` and
   `model_calls`. How episodes end tells you a lot: a win, running out of turns, an invalid move that ends the game,
   a harness error.
2. Look at the distribution of scores and steps, not only their average. A cluster far from the rest usually means
   a distinct failure mode rather than a hard instance.
3. Read the last few steps of failing traces, not only the first ones. Find the step where the model went wrong and
   what it saw at that point.
4. Compare with other candidates in the archive: what did earlier changes fix, and what did they break?

## What to do

1. Read `archive/summary.md`, then work through the evidence above for the parent and for other candidates. Name the
   failure pattern you find and count how many episodes show it.
2. Change the harness to fix one pattern that you saw in several episodes. Make the smallest change that could fix
   it: one mechanism, and as few lines as you can. A large rewrite is both harder to judge from one score and more
   likely to be a change that only suits the instances in the archive. Do not try to play the game yourself; the loop
   evaluates your harness.
   Before you finish, check that it passes `python check_harness.py harness/harness.py` and runs with
   `python mock_env.py`.
3. Write `proposal.md` in the workspace root. State the failure pattern and cite the episodes and steps where you saw
   it, with the numbers from step 1. Then state your change, why you expect it to help, and which of the failing
   episodes you expect to flip. Say what you would conclude if the score does not move.

## Rules

- Work only with the files in this workspace. Edit only `harness/harness.py`, and write `proposal.md`.
- Do not hard-code anything about specific instances, such as answers you saw in the traces. The final harness is
  scored on instances that are not in the archive; a harness that holds the archive's answers is rejected.
- General knowledge about the game is fine.
- Your change must be something a harness could do on an instance it has never seen. If it only works because you
  know what is in the archive, it is the wrong change.
