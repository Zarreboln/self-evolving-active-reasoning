# ===================== fixed runner (not editable by the Builder) =====================
# Appended below the Builder's components by mb/baselines/metaskill/assemble.py; never imported on
# its own (it reads the component globals INSTRUCTION, MEMORY, CONTEXT, CONTROLLER, VERIFICATION,
# WORKSPACE, TOOLS defined above it).
#
# A port of the released ModularTargetRunner + ModularSession (MetaSkill-AI4AI @72a7b02,
# src/meta_skill_ai4ai/modular/runtime.py and state.py) to the text interface of mb/evolve/child.py.
# Same callbacks, return shapes, event stream, snapshot, memory store, context groups, scratch
# workbench and composed-tool loop. Differences, all forced by our interface (see the package doc):
#   * the native primitive is `act(action)` (one environment step); there is no `finish` tool, the
#     environment ends the episode, so verification gates every action (before_action) instead of
#     the final answer (before_finish);
#   * our chat() has no native tool calling, so tools are listed in the system message and called
#     with a `Tool: name {json}` line; a reply without one is an action;
#   * `act` cannot be hidden by the controller and cannot be called from workspace setup,
#     verification or recovery actions (composed tools may call it, as native tools in the paper);
#   * per action at most RT_LIMITS tool calls / model calls / gate rejections, so a harness cannot
#     exceed the broker's model-call cap (8 per step); the paper's per-task turn and token budgets
#     (Table 5) are replaced by the environment's own step limit and the 64K context;
#   * scratch files live in memory (the child process has no file system access).
# A callback that raises or breaks its contract stops the episode (released: stop_reason
# "module_failure"); the environment keeps the score it has at that point.
import json as rt_json
from harness_lib import extract_action as rt_extract_action

RT_BOUNDARY = "@@BOUNDARY@@"
RT_LIMITS = {"event_tail": 24, "event_chars": 32768, "memory_records": 256, "memory_chars": 131072,
             "workbench_files": 64, "workbench_chars": 524288, "composed_actions": 32,
             "policy_calls": 10000, "state_chars": 65536, "instruction_chars": 8192,
             "observation_chars": 30000, "tool_calls_per_action": 4, "model_calls_per_action": 6,
             "gate_rejections_per_action": 2, "empty_replies": 2, "max_admissible_shown": 60}
RT_SAMPLING = {"temperature": 0.0, "max_tokens": 16384}     # paper §4.1 / Table 5
RT_PRIMITIVES = ["act", "workbench_write", "workbench_read", "workbench_list"]
RT_PRIMITIVE_DOCS = {
    "workbench_write": ("Write a text file in task-local scratch space, not the official deliverable workspace.",
                        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"], "additionalProperties": False}),
    "workbench_read": ("Read a task-local scratch file.",
                       {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                        "additionalProperties": False}),
    "workbench_list": ("List task-local scratch files.",
                       {"type": "object", "properties": {}, "additionalProperties": False}),
}


class RtContractError(Exception):
    pass


class RtStop(Exception):
    pass


def rt_copy(value, limit):
    try:
        text = rt_json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise RtContractError("value is not JSON: " + str(err)[:200])
    if len(text) > limit:
        raise RtContractError("value exceeds " + str(limit) + " characters")
    return rt_json.loads(text)


def rt_shape(value, allowed, required, what):
    if not isinstance(value, dict):
        raise RtContractError(what + " must be an object")
    extra = sorted(set(value) - set(allowed))
    missing = sorted(set(required) - set(value))
    if extra or missing:
        raise RtContractError(what + " has unexpected fields " + str(extra) + " or misses " + str(missing))
    return value


def rt_dump(value, limit):
    text = rt_json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        text = rt_json.dumps({"truncated": True, "original_characters": len(text), "prefix": text[:limit]})
    return text


def rt_groups(transcript):
    """Messages after the system/task pair, one group per assistant reply with what follows it."""
    groups = []
    for msg in transcript[2:]:
        if msg["role"] == "assistant" or not groups:
            groups.append({"id": len(groups), "messages": [msg]})
        else:
            groups[-1]["messages"].append(msg)
    return groups


def rt_describe(groups):
    limit = min(256, 64000 // max(1, len(groups)))
    out = []
    for g in groups:
        text = rt_json.dumps(g["messages"], ensure_ascii=False)
        out.append({"id": g["id"], "roles": "/".join(m["role"] for m in g["messages"]),
                    "characters": len(text), "preview": text[:limit]})
    return out


def rt_parse_tool(reply):
    """(name, arguments) if the last protocol line of the reply is a Tool line, else None."""
    last = None
    for line in (reply or "").splitlines():
        s = line.strip().strip("`").strip()
        low = s.lower()
        if low.startswith("action:"):
            last = None
        elif low.startswith("tool:"):
            body = s[5:].strip()
            name = body.split(" ")[0].split("{")[0].strip()
            rest = body[len(name):].strip()
            last = (name, rest)
    if last is None:
        return None
    name, rest = last
    if not rest:
        return (name, {})
    try:
        args = rt_json.loads(rest)
    except ValueError:
        return (name, None)
    return (name, args if isinstance(args, dict) else None)


def rt_action_text(reply, env):
    """The action of a reply: the text after the last `Action:` line; without one, the whole reply where
    the game parses its own format (env.reads_full_reply), else its last line."""
    for line in reversed((reply or "").splitlines()):
        if line.strip().lower().startswith("action:"):
            return rt_extract_action(reply)
    if env.reads_full_reply:
        return (reply or "").strip()
    return rt_extract_action(reply)


def run_episode(env, chat):
    st = {"records": {}, "state": {}, "phase": "", "instruction": "", "visible": None, "events": [],
          "policy_calls": 0, "files": {}, "model_calls": 0, "tool_calls": 0, "since_action_calls": 0,
          "since_action_tools": 0, "rejections": 0, "empty": 0}
    composed = {}
    for spec in TOOLS:
        composed[spec["name"]] = spec
    tool_names = sorted(set(RT_PRIMITIVES) | set(composed))
    instruction = INSTRUCTION if isinstance(INSTRUCTION, str) else ""
    meta = {"max_actions": env.max_steps, "lists_valid_actions": env.admissible() is not None,
            "game_parses_whole_reply": env.reads_full_reply}
    first = ("Task:\n" + env.instructions + "\n\nInitial observation:\n" + (env.observation or "(none)")
             + "\n\nPublic metadata: " + rt_json.dumps(meta))
    transcript = [{"role": "system", "content": instruction + "\n\n" + RT_BOUNDARY},
                  {"role": "user", "content": first}]

    def remaining():
        return {"actions": env.max_steps - env.steps,
                "model_calls_before_next_action": RT_LIMITS["model_calls_per_action"] - st["since_action_calls"],
                "tool_calls_before_next_action": RT_LIMITS["tool_calls_per_action"] - st["since_action_tools"]}

    def policy(component, name, args):
        st["policy_calls"] += 1
        if st["policy_calls"] > RT_LIMITS["policy_calls"]:
            raise RtContractError("episode policy-call allowance exceeded")
        try:
            return component[name](*args)
        except (RtContractError, RtStop):
            raise
        except Exception as err:
            raise RtContractError(name + " raised " + repr(err)[:300])

    def records():
        return rt_copy(list(st["records"].values()), RT_LIMITS["memory_chars"] * 2)

    def snapshot():
        return {"controller": rt_copy(st["state"], RT_LIMITS["state_chars"]), "phase": st["phase"],
                "memory": records(), "recent_events": rt_copy(st["events"], RT_LIMITS["event_chars"] * 2),
                "remaining": remaining()}

    def memory_apply(decision):
        rt_shape(decision, ["upsert", "delete"], ["upsert", "delete"], "memory.write result")
        if not isinstance(decision["upsert"], list) or not isinstance(decision["delete"], list):
            raise RtContractError("memory operations must be arrays")
        updated = dict(st["records"])
        for ident in decision["delete"]:
            if not isinstance(ident, str) or ident not in updated:
                raise RtContractError("memory deletion must name an existing record")
            updated.pop(ident)
        seen = set()
        for rec in decision["upsert"]:
            if not isinstance(rec, dict):
                raise RtContractError("memory records must be objects")
            ident = rec.get("id")
            if not isinstance(ident, str) or not ident or len(ident) > 128 or ident in seen:
                raise RtContractError("memory upserts need distinct, nonempty string ids")
            seen.add(ident)
            updated[ident] = rt_copy(rec, RT_LIMITS["memory_chars"])
        if len(updated) > RT_LIMITS["memory_records"]:
            raise RtContractError("memory record allowance exceeded; policy must evict")
        rt_copy(list(updated.values()), RT_LIMITS["memory_chars"])
        st["records"] = updated

    def observe(event):
        event = rt_copy(event, RT_LIMITS["event_chars"] * 4)
        st["events"].append(event)
        st["events"] = st["events"][-RT_LIMITS["event_tail"]:]
        while len(st["events"]) > 1 and len(rt_json.dumps(st["events"])) > RT_LIMITS["event_chars"]:
            st["events"].pop(0)
        if MEMORY is not None:
            memory_apply(policy(MEMORY, "write", [event, records()]))
        if CONTROLLER is not None:
            d = rt_shape(policy(CONTROLLER, "on_event", [event, snapshot()]),
                         ["state", "phase", "visible_tools", "instruction"], [], "controller.on_event result")
            if "state" in d:
                if not isinstance(d["state"], dict):
                    raise RtContractError("controller state must be an object")
                st["state"] = rt_copy(d["state"], RT_LIMITS["state_chars"])
            if "phase" in d:
                if not isinstance(d["phase"], str) or len(d["phase"]) > 128:
                    raise RtContractError("controller phase must be a short string")
                st["phase"] = d["phase"]
            if "instruction" in d:
                if not isinstance(d["instruction"], str) or len(d["instruction"]) > RT_LIMITS["instruction_chars"]:
                    raise RtContractError("controller instruction exceeds its limit")
                st["instruction"] = d["instruction"]
            if "visible_tools" in d:
                vis = d["visible_tools"]
                if vis is not None and (not isinstance(vis, list) or len(set(vis)) != len(vis)
                                        or any(x not in tool_names for x in vis)):
                    raise RtContractError("controller selected unknown or duplicate tools")
                st["visible"] = vis

    def scratch_path(path):
        if not isinstance(path, str) or not path or len(path) > 240 or path.startswith("/") \
                or any(p in ("", ".", "..") for p in path.split("/")):
            raise RtContractError("scratch path must be a short relative POSIX path")
        return path

    def run_primitive(name, args):
        if name == "act":
            if env.done:
                return {"ok": False, "observation": {"error": "the episode is over"}, "error_type": "episode_over"}
            try:
                obs = env.act(str(args.get("action", "")))
            except Exception as err:            # the environment refused: the episode is over
                raise RtStop("environment: " + str(err)[:300])
            st["since_action_calls"] = 0
            st["since_action_tools"] = 0
            st["rejections"] = 0
            return {"ok": True, "observation": obs[:RT_LIMITS["observation_chars"]], "error_type": None,
                    "done": env.done}
        if name == "workbench_write":
            path = scratch_path(args.get("path"))
            content = args.get("content")
            if not isinstance(content, str):
                raise RtContractError("scratch file content must be text")
            files = dict(st["files"])
            files[path] = content
            if len(files) > RT_LIMITS["workbench_files"] or sum(len(v) for v in files.values()) > RT_LIMITS["workbench_chars"]:
                raise RtContractError("scratch workspace allowance exceeded")
            st["files"] = files
            return {"ok": True, "observation": {"path": path, "characters": len(content)}, "error_type": None}
        if name == "workbench_read":
            path = scratch_path(args.get("path"))
            if path not in st["files"]:
                raise RtContractError("scratch file does not exist")
            return {"ok": True, "observation": {"path": path, "content": st["files"][path]}, "error_type": None}
        if name == "workbench_list":
            return {"ok": True, "observation": [{"path": k, "characters": len(v)} for k, v in sorted(st["files"].items())],
                    "error_type": None}
        return {"ok": False, "observation": {"error": "unknown primitive tool"}, "error_type": "unknown_tool"}

    def execute_primitive(name, args, origin):
        st["tool_calls"] += 1
        if not isinstance(args, dict):
            result = {"ok": False, "observation": {"error": "arguments must be an object"},
                      "error_type": "invalid_tool_arguments"}
            args = {"rejected_invalid_arguments": True}
        else:
            try:
                result = run_primitive(name, args)
            except RtContractError as err:
                result = {"ok": False, "observation": {"error": str(err)}, "error_type": "workspace_error"}
        observe({"type": "primitive_result", "tool": name, "arguments": args, "origin": origin,
                 "result": result})
        return result

    def run_actions(value, origin):
        if not isinstance(value, list) or len(value) > RT_LIMITS["composed_actions"]:
            raise RtContractError("policy action list exceeds its limit")
        out = []
        for request in value:
            rt_shape(request, ["tool", "arguments"], ["tool", "arguments"], "primitive request")
            if request["tool"] == "act":
                raise RtContractError("act cannot be called from " + origin)
            out.append(execute_primitive(request["tool"], request["arguments"], origin))
        return out

    def invoke_composed(name, args):
        spec = composed[name]
        if not isinstance(args, dict):
            return {"ok": False, "observation": {"error": "arguments must be an object"},
                    "error_type": "invalid_tool_arguments"}
        missing = [k for k in spec.get("parameters", {}).get("required", []) if k not in args]
        if missing:
            return {"ok": False, "observation": {"error": "missing arguments " + str(missing)},
                    "error_type": "invalid_tool_arguments"}
        state, previous, used = {}, None, 0
        for i in range(RT_LIMITS["composed_actions"] + 1):
            d = rt_shape(policy(spec, "next_action", [rt_copy(args, RT_LIMITS["state_chars"]), state, previous]),
                         ["state", "call", "return"], ["state"], "composed tool result")
            if ("call" in d) == ("return" in d):
                raise RtContractError("composed tool must either call a primitive or return")
            state = rt_copy(d["state"], RT_LIMITS["state_chars"])
            if "return" in d:
                if used == 0:
                    st["tool_calls"] += 1
                return {"ok": True, "observation": rt_copy(d["return"], RT_LIMITS["observation_chars"]),
                        "error_type": None}
            if used >= RT_LIMITS["composed_actions"]:
                raise RtContractError("composed tool primitive allowance exceeded")
            call = rt_shape(d["call"], ["tool", "arguments"], ["tool", "arguments"], "composed primitive request")
            if call["tool"] not in RT_PRIMITIVES:
                raise RtContractError("a composed tool may call only primitive tools")
            previous = execute_primitive(call["tool"], call["arguments"], "composed:" + name)
            used += 1
            if env.done:
                return {"ok": True, "observation": {"episode_over": True, "last_result": previous},
                        "error_type": None}
        raise RtContractError("composed tool did not return")

    def verify(action):
        if VERIFICATION is None:
            return True, ""
        d = rt_shape(policy(VERIFICATION, "before_action", [action, snapshot()]),
                     ["allow", "feedback", "actions"], ["allow", "feedback"], "verification.before_action result")
        if not isinstance(d["allow"], bool) or not isinstance(d["feedback"], str):
            raise RtContractError("verification needs boolean allow and text feedback")
        if d["allow"] and d.get("actions"):
            raise RtContractError("verification cannot accept before requested checks execute")
        checks = run_actions(d.get("actions", []), "verification")
        return d["allow"], rt_dump({"verification": d["feedback"], "checks": checks}, RT_LIMITS["observation_chars"])

    def recover(name, result):
        if VERIFICATION is None or "on_failure" not in VERIFICATION or result["ok"]:
            return None
        d = rt_shape(policy(VERIFICATION, "on_failure", [{"tool": name, "result": result}, snapshot()]),
                     ["actions", "feedback"], ["actions", "feedback"], "verification.on_failure result")
        if not isinstance(d["feedback"], str):
            raise RtContractError("recovery feedback must be text")
        done = run_actions(d["actions"], "recovery")
        if d["feedback"] or done:
            return rt_dump({"recovery": {"feedback": d["feedback"], "results": done}}, RT_LIMITS["observation_chars"])
        return None

    def observation_message(obs):
        text = "Observation:\n" + obs
        adm = env.admissible()
        if adm and len(adm) <= RT_LIMITS["max_admissible_shown"] and not env.done:
            text += "\n\nValid actions now: " + ", ".join(adm)
        return text

    def model_input(force):
        selected = [dict(m) for m in transcript]
        retrieved = []
        if MEMORY is not None:
            query = {"task": env.instructions, "phase": st["phase"],
                     "latest_event": st["events"][-1] if st["events"] else None}
            ids = policy(MEMORY, "retrieve", [query, records()])
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) \
                    or len(set(ids)) != len(ids) or any(x not in st["records"] for x in ids):
                raise RtContractError("memory.retrieve returns an array of distinct existing record ids")
            retrieved = [rt_copy(st["records"][x], RT_LIMITS["memory_chars"]) for x in ids]
        if CONTEXT is not None:
            groups = rt_groups(transcript)
            keep = policy(CONTEXT, "select_context", [rt_describe(groups), retrieved])
            if not isinstance(keep, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in keep):
                raise RtContractError("context selection must be an array of integer group ids")
            if keep != sorted(set(keep)) or any(x < 0 or x >= len(groups) for x in keep):
                raise RtContractError("context selection must retain original order without duplicates")
            if groups and len(groups) - 1 not in keep:
                raise RtContractError("context selection must retain the newest complete exchange")
            selected = [dict(m) for m in transcript[:2]]
            for i in keep:
                selected.extend(dict(m) for m in groups[i]["messages"])
        if retrieved or st["instruction"] or st["phase"]:
            selected.append({"role": "user", "content": rt_dump(
                {"harness_memory": retrieved, "harness_phase": st["phase"],
                 "harness_instruction": st["instruction"]},
                RT_LIMITS["memory_chars"] + RT_LIMITS["instruction_chars"])})
        visible = [] if force else (tool_names if st["visible"] is None else list(st["visible"]))
        listing = []
        for name in sorted(set(visible) - {"act"}):
            if name in composed:
                doc, params = composed[name].get("description", ""), composed[name].get("parameters", {})
            else:
                doc, params = RT_PRIMITIVE_DOCS[name]
            listing.append("- " + name + " " + rt_json.dumps(params) + ": " + doc)
        tools_text = ("Harness tools (call with `Tool: <name> <JSON arguments>`):\n" + "\n".join(listing)
                      if listing else "No harness tools are available for this reply; reply with an Action.")
        selected[0]["content"] = selected[0]["content"] + "\n\n" + tools_text
        budget = rt_json.dumps({"remaining_budget_before_this_request": remaining()})
        if force:
            budget += ("\nThe tool and model-call allowance before the next action is used up. Reply now with "
                       "`Action: <action>`.")
        # released: appended to the system message; here at the end of the last message, so the system
        # prefix stays fixed (vLLM prefix caching; trace.md shows each prompt once)
        if selected[-1]["role"] == "user":
            selected[-1]["content"] = selected[-1]["content"] + "\n\n" + budget
        else:
            selected.append({"role": "user", "content": budget})
        return selected, visible

    try:
        observe({"type": "round_start", "task": {"prompt": env.instructions,
                                                 "initial_observation": env.observation,
                                                 "public_metadata": meta}})
        if WORKSPACE is not None:
            setup = rt_shape(policy(WORKSPACE, "setup", [{"prompt": env.instructions,
                                                          "initial_observation": env.observation,
                                                          "public_metadata": meta}]),
                             ["files", "actions"], [], "workspace.setup result")
            files = setup.get("files", [])
            if not isinstance(files, list) or len(files) > RT_LIMITS["workbench_files"]:
                raise RtContractError("invalid workspace setup files")
            for f in files:
                rt_shape(f, ["path", "content"], ["path", "content"], "workspace file")
                execute_primitive("workbench_write", f, "workspace_setup")
            results = run_actions(setup.get("actions", []), "workspace_setup")
            transcript.append({"role": "user", "content": rt_dump(
                {"workbench_files": [{"path": k, "characters": len(v)} for k, v in sorted(st["files"].items())],
                 "setup_results": results}, RT_LIMITS["observation_chars"])})
        while not env.done:
            observe({"type": "before_model", "step": st["model_calls"] + 1, "remaining": remaining()})
            force = st["since_action_calls"] >= RT_LIMITS["model_calls_per_action"] - 1
            messages, visible = model_input(force)
            try:
                reply = chat(messages, temperature=RT_SAMPLING["temperature"], max_tokens=RT_SAMPLING["max_tokens"])
            except Exception as err:
                raise RtStop("model_failure: " + str(err)[:300])
            st["model_calls"] += 1
            st["since_action_calls"] += 1
            if not isinstance(reply, str) or not reply.strip():
                st["empty"] += 1
                observe({"type": "model_output_error", "step": st["model_calls"], "error": "empty_action_and_answer",
                         "consecutive_empty_responses": st["empty"]})
                if st["empty"] > RT_LIMITS["empty_replies"]:
                    raise RtStop("model_failure: empty replies")
                transcript.append({"role": "user", "content": "No tool call or action was returned. Return a "
                                   "valid tool call or an action. This correction uses the remaining budget."})
                continue
            st["empty"] = 0
            transcript.append({"role": "assistant", "content": reply})
            call = None if force else rt_parse_tool(reply)
            if call is not None and call[0] != "act":
                name, args = call
                if name not in visible:
                    result = {"ok": False, "observation": {"error": "tool not exposed in this phase"},
                              "error_type": "tool_not_exposed"}
                    st["tool_calls"] += 1
                elif args is None:
                    result = {"ok": False, "observation": {"error": "invalid tool arguments"},
                              "error_type": "invalid_tool_arguments"}
                    st["tool_calls"] += 1
                elif name in composed:
                    result = invoke_composed(name, args)
                else:
                    result = execute_primitive(name, args, "target")
                st["since_action_tools"] += 1
                if env.done:
                    break
                transcript.append({"role": "user", "content": "Tool result (" + name + "): "
                                   + rt_dump(result, RT_LIMITS["observation_chars"])})
                rec = recover(name, result)
                if rec:
                    transcript.append({"role": "user", "content": rec})
                continue
            if call is not None:
                action = str((call[1] or {}).get("action", ""))
            else:
                action = rt_action_text(reply, env)
            allowed, feedback = verify(action)
            if not allowed and st["rejections"] >= RT_LIMITS["gate_rejections_per_action"]:
                allowed = True
                feedback = feedback + " (rejection allowance used; action sent)"
            if VERIFICATION is not None:
                observe({"type": "action_check_result", "tool": "act", "arguments": {"action": action},
                         "result": {"ok": allowed, "observation": {"feedback": feedback},
                                    "error_type": None if allowed else "verification_rejected"}})
            if not allowed:
                st["rejections"] += 1
                transcript.append({"role": "user", "content": feedback})
                continue
            result = execute_primitive("act", {"action": action}, "target")
            if result["ok"]:
                transcript.append({"role": "user", "content": observation_message(result["observation"])})
            else:
                transcript.append({"role": "user", "content": "Tool result (act): "
                                   + rt_dump(result, RT_LIMITS["observation_chars"])})
    except (RtContractError, RtStop) as err:
        print("runner stop: " + str(err))
        return
