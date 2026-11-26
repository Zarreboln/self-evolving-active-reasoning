"""The fixed part of a Harness² harness in our setting: the agent loop the components plug into.

In Harness² the agent framework (OpenCode or Codex) is the fixed substrate and only the eight file
components around it are edited. Here the substrate is this loop, TextArena's standard agent loop
(the seed `mb/evolve/seeds/textarena_standard.py`), extended with the three on-demand tools a
component tree needs (load a skill, run a script, call a subagent) and two plugin hook points.
With an empty tree (H0) the loop behaves exactly like the seed: one stateless call per turn with
the standard system prompt and the observation, the reply stripped and sent as the action.

`RUNTIME` is pasted verbatim at the end of every generated harness.py (surface.assemble); the editor
reads it as `context/runtime/harness_runtime.py` and cannot change it.
"""

RUNTIME = r'''
# ======================================================================================
# RUNTIME (fixed; generated harnesses always end with this block, the editor cannot change it)
# ======================================================================================
MAX_TOOL_CALLS = 3          # on-demand tool uses per turn (each may cost one extra model call)
TOOL_LINE = re.compile(r"^\s*TOOL:\s*(skill|script|subagent)\s+([a-z0-9][a-z0-9_\-]*)\s*(.*)$", re.I)


def tool_menu():
    """The tools section of the system prompt; empty when the tree has no on-demand component."""
    if not (SKILLS or SCRIPTS or SUBAGENTS):
        return ""
    lines = ["# On-demand tools",
             "Before you answer, you may use a tool by replying with ONE line and nothing else:",
             "  TOOL: skill <name>             -- read that skill's full instructions",
             "  TOOL: script <name> <input>    -- run a deterministic helper on <input> text",
             "  TOOL: subagent <name> <input>  -- ask a helper with its own context about <input>",
             "The result comes back as the next message; then reply with your action as usual. "
             "At most %d tool uses per turn." % MAX_TOOL_CALLS]
    for name in sorted(SKILLS):
        lines.append("- skill `%s`: %s" % (name, SKILLS[name]["description"]))
    for name in sorted(SCRIPTS):
        lines.append("- script `%s`: %s" % (name, SCRIPT_DOCS.get(name, "")))
    for name in sorted(SUBAGENTS):
        lines.append("- subagent `%s`: %s" % (name, SUBAGENTS[name]["description"]))
    return "\n".join(lines)


def system_text():
    parts = [SYSTEM_PROMPT, GUARDRAILS, MEMORY, TOOL_GUIDANCE, tool_menu()]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def call_subagent(chat, name, text):
    sub = SUBAGENTS.get(name)
    if sub is None:
        return "(no subagent named %s)" % name
    return chat([{"role": "system", "content": sub["prompt"]}, {"role": "user", "content": text}],
                temperature=None, max_tokens=None)


def run_tool(chat, kind, name, arg):
    kind = kind.lower()
    try:
        if kind == "skill":
            s = SKILLS.get(name)
            return ("# skill %s\n%s" % (name, s["body"])) if s else "(no skill named %s)" % name
        if kind == "script":
            fn = SCRIPTS.get(name)
            return str(fn(arg)) if fn else "(no script named %s)" % name
        return call_subagent(chat, name, arg)
    except Exception as exc:          # a failing component is visible to the model, not fatal
        return "(%s %s failed: %s)" % (kind, name, str(exc)[:300])


def tool_request(reply):
    lines = [l for l in (reply or "").strip().splitlines() if l.strip()]
    if not lines or not (SKILLS or SCRIPTS or SUBAGENTS):
        return None
    m = TOOL_LINE.match(lines[-1])
    return m.groups() if m else None


def run_hooks(hook, value, ctx):
    for p in PLUGINS:
        fn = p.get(hook)
        if fn is not None:
            try:
                out = fn(value, ctx)
            except Exception:
                out = None
            if out is not None:
                value = str(out)
    return value


def run_episode(env, chat):
    ctx = {"instructions": env.instructions, "turn": 0, "history": [], "state": {},
           "scripts": SCRIPTS, "max_steps": env.max_steps,
           "subagent": lambda name, text: call_subagent(chat, name, text)}
    obs = env.instructions
    while not env.done:
        ctx["admissible"] = env.admissible()
        shown = run_hooks("on_observation", obs, ctx)
        messages = [{"role": "system", "content": system_text()}, {"role": "user", "content": shown}]
        reply = chat(messages, temperature=None, max_tokens=None)
        used = 0
        while used < MAX_TOOL_CALLS:
            req = tool_request(reply)
            if req is None:
                break
            used += 1
            result = run_tool(chat, req[0], req[1], req[2])
            messages = messages + [{"role": "assistant", "content": reply}, {"role": "user", "content": result}]
            reply = chat(messages, temperature=None, max_tokens=None)
        action = run_hooks("on_action", reply.strip(), ctx)
        obs = env.act(action)
        ctx["history"].append((action, obs))
        ctx["turn"] += 1
'''
