"""Learning Meta-Skills (arXiv 2609.38143 v2) as a baseline of experiment B.

Official code: github.com/qiancheng-apodex/MetaSkill-AI4AI, cloned at third_party_b/metaskill, pinned
to 72a7b02 (2026-09-29). It has no license file, so nothing of it is copied here: its prompts, runner
boundary, skill schema and token counter are loaded from the clone at run time (official.py).

Released (used from the code)                      | Not released (implemented from the paper)
---------------------------------------------------|-----------------------------------------------
Builder prompt BUILDER_SYSTEM + build instruction  | the experiment scheduler: dev passes, batches,
interface-repair turn (<= 2 repairs)               |   freeze, test-time construction (§3.1-3.2,
reflection prompt and keep/add/revise validation   |   eq. (2)-(3), §4.1) -> driver.py
  (reflect_skill, SupportSkill)                    | 192-token skill limit (B.1), archive of
runner: callbacks, event stream, snapshot, memory  |   superseded skills (B.1), "cite evidence from
  store, context groups, workbench, composed tools,|   that batch" (§3.1), failures score zero (B.3)
  verification gate (ModularTargetRunner, ported   | BM25 top-2 retrieval (C): not used; the paper's
  to our interface in runner_template.py)          |   better mode, full bank (§4.2), is the one run
neutral bundle H0 (neutral_bundle)                 |

Mapping onto experiment B (notes/exp_b_design.md):
  Builder              Claude Opus 5.5 as a headless Claude Code session in the Docker sandbox
                       (agents.py; subscription, no API key). Paper: GPT-5.6-Sol.
  Target               the frozen Qwen3.5-9B, played on ORCD through the broker (executor.py).
  development tasks    the 7 of the design; one batch = one task's 20 search problems; one round
                       (= one development pass) = 7 batches = 140 episodes; 8 rounds (paper: 2 passes).
  harness              one file of our interface: the Builder's components + the fixed runner (assemble.py),
                       checked by interface_check.py and check.py, run only through broker/child.
  test                 per new task, one harness from the frozen full bank and the task's first public
                       prompt, played once on each problem in the fixed order (driver.py --build-per task);
                       --build-per problem builds one per problem as in the paper's test protocol.

The seven component families as slots of a single-file harness (for reuse by our method):
  family (Table 1)          slot in components.py        callback(s), arity         runner calls it
  instructions              INSTRUCTION: str             -                          system message, first
  memory                    MEMORY                       write(event, records) 2    on every event
                                                         retrieve(query, records) 2 before every model call
  context organization      CONTEXT                      select_context(groups, retrieved) 2   before every model call
  composed tools            TOOLS: [{name, description,  next_action(arguments, state, previous) 3
                            parameters, next_action}]                               when the Target calls it
  execution control         CONTROLLER                   on_event(event, snapshot) 2  on every event; sets state,
                                                                                    phase, visible_tools, instruction
  verification & recovery   VERIFICATION                 before_action(action, snapshot) 2   before every env action
                                                         on_failure(failure, snapshot) 2 (optional)  after a failed tool
  workspace preparation     WORKSPACE                    setup(task) 1              once, at the start
A disabled slot is None (TOOLS = []). Return shapes are the released ones (see components_template.py).

Entry points: driver.py (Mac), prep_cluster.py (cluster, CPU, once), selftest.py (CPU test with fakes).
"""
