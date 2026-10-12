"""CircuitDecoding prompt text, in the three coordination modes.

The intro carries the task's own rules (Multi-Turn Puzzles ``cd_instruction.txt``)
re-expressed for this harness's I/O format (OBJECTIVE / OBSERVATION / ADMISSIBLE
COMMANDS, ``<think>`` + ``<action>``); the environment's own answer lines are
delivered verbatim in OBSERVATION."""

from ...prompts import PromptSpec

INTRO = """You are playing the Circuit Deduction Challenge.

There are 3 hidden boolean circuits, labeled A, B, C. Each circuit accepts 3 binary inputs (0 or 1) and produces a single binary output. All circuits are built in layers from a shared inventory of AND, OR and NOT gates (given in the OBJECTIVE): at each step either an AND or OR gate combines two available signals into one, or a NOT gate inverts one signal; this continues until a single output remains. All gates are used, distributed without replacement across the 3 circuits. No circuit always outputs 0 or always outputs 1.

Your goal: deduce the truth table of every circuit. You have 18 queries; each query feeds one circuit one input triple and returns its output bit. After the 18th answer you must submit the joint truth table of all circuits as a 24-bit string: circuit A's 8 outputs for inputs 000, 001, 010, 011, 100, 101, 110, 111, then circuit B's 8, then circuit C's 8.

You will receive:
- OBJECTIVE: the inventory of gates and the budget.
- OBSERVATION: the environment's reply to your last action (the answer bit and the remaining budget).
- ADMISSIBLE COMMANDS: the valid environment actions right now."""

ENV_ACTIONS = """## Environment Actions
- `query X(a, b, c)`: ask circuit X (A, B or C) for its output on inputs a, b, c (each 0 or 1). Example: `query A(0, 1, 1)`. One query per turn; each spends one of the 18.
- `guess <24 bits>`: only when the OBSERVATION asks for the final guess. Example: `guess 000101110011100100011111`."""

HARNESS_HEADER = """## Harness Actions (cognitive tools — always available)
These access your external memory and planning systems. They do not spend queries."""

PLAN_TOOL = """### Plan
- `commit [subgoal]`: register ONE subgoal you are pursuing. Example: `commit [pin down circuit A]`, later `commit [pin down circuit B]`."""

BELIEF_TOOL = """### Perception
- `track [what]`: query the harness's exact bookkeeping of what your answers so far imply. `track [remaining]` reports how many circuit configurations are still consistent with every answer, which outputs are already determined (asking them again gains nothing), which outputs are still open, and — once few remain — the full list of remaining joint truth tables. `track [A]` (or B, C) reports that circuit's determined/undetermined outputs."""

EXPERIENCE_TOOL = """### Experience
- `recall [query]`: retrieve experience from earlier episodes (strategies, common mistakes). Example: `recall [how to choose informative queries]`.
- `note [insight]`: record a generalizable discovery for future episodes. Example: `note [query inputs that split the remaining configurations evenly]`."""

WHEN_COMMIT = """### commit — register subgoals
- TASK START: `commit [first subgoal]`; when a circuit is pinned down, `commit [next circuit]`."""

WHEN_TRACK = """### track — check what is already known
Your own notes drift: before spending a query, `track [remaining]` tells you which outputs are already determined by the inventory plus your answers, so you never ask one of those, and lists the remaining configurations once they are few. Use it before the final guess: with one configuration left, its 24 bits are the answer."""

WHEN_RECALL = """### recall — query experience
- TASK START: `recall [how to choose informative queries]`.
- STUCK: `recall [mistakes to avoid]`."""

WHEN_NOTE = """### note — update the experience bank
After the final guess, `note` one generalizable lesson about querying or assembling the answer (not instance-specific)."""

_OUTPUT_FORMAT = """## Output Format (MANDATORY)
<think>Brief reasoning in 1-2 sentences.</think>
<action>your action here</action>"""

FOOTER_INLINE = (
    _OUTPUT_FORMAT
    + """

Rules:
1. For environment actions, choose EXACTLY from the ADMISSIBLE COMMANDS list.
2. Harness actions are free; use them when they save a query.
3. Issue exactly one action per turn.
4. You MUST use <think></think> and <action></action> tags every turn.
5. Do not ask an output you already know."""
)

EXPERIENCE_RULE = """6. After the final guess, use `note` once with a generalizable lesson."""

_ENV_ONLY_RULES = """

Rules:
1. Choose EXACTLY from the ADMISSIBLE COMMANDS list.
2. Issue exactly one action per turn.
3. You MUST use <think></think> and <action></action> tags every turn.
4. Do not ask an output you already know; plan your 18 queries to maximize information."""

FOOTER_ENV_ONLY = _OUTPUT_FORMAT + _ENV_ONLY_RULES

ALWAYS_ON = """## Evolving Memory (provided automatically, every turn)
The harness maintains these sections and refreshes them before every turn:
- STATE: `world:` is exact bookkeeping of what your answers imply — how many circuit configurations remain consistent, which outputs are already determined (asking them gains nothing), which are still open, and the remaining joint truth tables once few remain; `plan:` is the rolling checklist.
- RETRIEVED SKILLS: lessons from earlier episodes, when any exist.
They are evidence computed from your own answers; if the OBSERVATION contradicts them, trust the OBSERVATION."""

FOOTER_ALWAYS_ON = (
    """There are no memory commands. Do not try to query or write memory; emit only environment actions.

"""
    + FOOTER_ENV_ONLY
)

PROMPT = PromptSpec(
    intro=INTRO,
    env_actions=ENV_ACTIONS,
    harness_header=HARNESS_HEADER,
    tools=(("plan", PLAN_TOOL), ("belief", BELIEF_TOOL), ("experience", EXPERIENCE_TOOL)),
    when_header="## When to Use Harness",
    when=(("plan", WHEN_COMMIT), ("belief", WHEN_TRACK), ("experience", WHEN_RECALL), ("experience", WHEN_NOTE)),
    footer_inline=FOOTER_INLINE,
    experience_rule=EXPERIENCE_RULE,
    always_on=ALWAYS_ON,
    footer_always_on=FOOTER_ALWAYS_ON,
    footer_env_only=FOOTER_ENV_ONLY,
)
