# Official initial agents of the suite's benchmarks (checked 2026-11-25, read-only)

Decision (user, 2026-11-25): where public code exists, follow it exactly (environment and agent); where it does
not, use one unified setting, which is BALROG's task-agnostic `naive` / `cot` pair. Two versions per task, one
without and one with reasoning. Blicket: environment as CannyLab's public causal_env_v0 defaults (3 objects,
subtask `official`, replacing n5); no public LLM agent, so the unified BALROG pair.

| task | official? | without reasoning | with reasoning | examples | source |
|---|---|---|---|---|---|
| GN, DC (AR-Bench) | yes, six per task | `zero_shot` (README example) | `proactive_cot` | none (few_shot has 1) | AR-Bench `arbench/reasoner/{gn,dc}/prompt.py`, `*_evaluator.py` |
| Wordle, Mastermind, Hangman, 20Q (TextArena 0.7.4) | one generic agent | `STANDARD_GAME_PROMPT` + `LLMObservationWrapper` (full history) + `ActionFormattingWrapper` | `ThoughtAgentWrapper` (separate thinking call) | none | `textarena/agents/basic_agents.py:10`, `agents/wrappers.py:8-62`, `envs/__init__.py` |
| BabyAI (BALROG) | yes | `naive` (paper's main results, history 16) | `cot` | none | BALROG `config/config.yaml`, `agents/naive.py`, `agents/chain_of_thought.py` |
| ALFWorld | no; ReAct is the standard | ReAct `act_put_{1,0}` | ReAct `react_put_{1,0}` (`think:` steps) | 2 hand-written demos | ysymyth/ReAct `alfworld.ipynb`, `prompts/alfworld_3prompts.json` |
| ScienceWorld | no (original baselines are RL) | SwiftSage SayCan prompt (`prompt_no_think.jsonl`) | SwiftSage ReAct baseline (`prompt.jsonl`, key "27" = test-conductivity) | 1 demo from a gold trajectory | SwiftSage branch `science_world`, `baselines/eval_agent_react.py` |
| Blicket | no public interactive agent (Kosoy et al. 2206.08353 is one-shot passive inference; GX-Chen 2505.09614 has no code) | unified: BALROG `naive` | unified: BALROG `cot` | none | env: CannyLab causal_overhypotheses `causal_env_v0.py` |

## Details that matter for porting
- AR-Bench zero_shot never stops early: GN plays 25 turns and is judged on a final guess after them; DC asks 25
  questions (two calls per turn: pick a suspect, then the question), then a verdict with "Reason: ... Answer: X".
  ToT in all three tasks scores branches with the hidden answer (oracle): not usable as a seed.
- TextArena agents are stateless: history lives in the wrapped observation. ThoughtAgentWrapper replaces the
  standard system prompt in both of its calls.
- BALROG naive parser: keep letters, spaces, colons; the action must match exactly, else "go forward" and the next
  observation is prefixed with "Your previous output did not contain a valid action. Defaulted to action: ...".
  GoToLocal is not one of BALROG's five BabyAI tasks; we use BALROG's wrapper and prompt on the official level.
- ReAct ALFWorld: text-davinci-002, T=0, stop "\n", at most 49 steps, think steps answered "OK.", full history.
- SwiftSage ReAct ScienceWorld also re-maps invalid actions to valid ones, drops the last 3 actions, and stops on
  loops; those are extra mechanisms, not the prompt. Seeds take the prompt and demo only.

## Mismatches found in our adapters (to fix)
1. GN system prompt was AR-Bench's ToT `Game_rule`, not zero_shot's `propose_template`.
2. TextArena games built with `-raw`: no `LLMObservationWrapper` / `ActionFormattingWrapper`.
3. BabyAI parser more lenient than BALROG's; invalid-action text not BALROG's.
4. ALFWorld step limit 50 vs ReAct's 49.
5. Blicket n5: no source uses 5 objects (original 3; GX-Chen 3, 4, 8). Resolved: use the public env's default, subtask `official` (3 objects).
