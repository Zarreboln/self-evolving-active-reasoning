"""Shared code base of the soft tree (notes/method_decomposition.md): one harness whose behaviour is set by
CONFIG, a dict the tree fills in per task (root values, overridden along the task's branches).

The code-editing model edits this file (new modules, each behind a CONFIG switch) and the node
configurations; it never writes task names into either. With the default CONFIG below this is exactly
TextArena's standard agent (seeds/textarena_standard.py), the starting point the one-harness-for-all
Meta-Harness also uses.

Modules and their CONFIG keys:
  system_prompt (str)        the system message
  history (int)              earlier (observation, reply) exchanges kept in the chat; 0 = stateless
  show_admissible (bool)     append the admissible actions to the observation when the game lists them
  extract_action (bool)      send only the `Action:` line of the reply instead of the whole reply
  action_format_hint (bool)  ask the model to end with an `Action:` line (useful with extract_action)
  temperature, max_tokens    sampling; None = the server's default
"""
from harness_lib import extract_action, with_admissible

DEFAULT_CONFIG = {
    "system_prompt": ("You are a competitive game player. Make sure you read the game instructions "
                      "carefully, and always follow the required format."),
    "history": 0,
    "show_admissible": False,
    "extract_action": False,
    "action_format_hint": False,
    "temperature": None,
    "max_tokens": None,
}
HINT = "\n\nEnd your reply with a line of the form `Action: <action>`."

CONFIG = {}          # filled in by the tree when a task's harness is assembled


def setting(key):
    return CONFIG.get(key, DEFAULT_CONFIG[key])


def run_episode(env, chat):
    system = setting("system_prompt") + (HINT if setting("action_format_hint") else "")
    obs = env.instructions if not env.observation else env.instructions + "\n\n" + env.observation
    past = []
    while not env.done:
        shown = with_admissible(obs, env) if setting("show_admissible") else obs
        keep = int(setting("history"))
        msgs = [{"role": "system", "content": system}] + (past[-2 * keep:] if keep > 0 else [])
        msgs.append({"role": "user", "content": shown})
        reply = chat(msgs, temperature=setting("temperature"), max_tokens=setting("max_tokens"))
        action = extract_action(reply) if setting("extract_action") else reply.strip()
        obs = env.act(action)
        past += [{"role": "user", "content": shown}, {"role": "assistant", "content": reply}]
