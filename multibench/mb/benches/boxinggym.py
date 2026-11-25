"""BoxingGym (Gandhi et al., NeurIPS 2025), its experimental-design ("oed") protocol.

Each subtask is one (environment, goal) pair that ships with a config in boxing-gym/conf/envs; the
environment parameters and the number of evaluation questions are read from that config. Moral
Machines and Emotion are left out: their environments answer with an OpenAI model, not a
simulator. The IRT goals other than `direct` have no config and are left out too.

The protocol is run_experiment.py's, with the text of its agent (agents/agent.py) as the
environment's side of the conversation:
  * 10 experiments (the last checkpoint of conf/exp/oed.yaml). An experiment is a reply holding
    `<observe>...</observe>`; the follow-up prompt carries the result. A reply that does not parse
    gets the agent's own retry message; an input the environment rejects gets the environment's
    error text. Neither counts as an experiment, as in the original loop.
  * Then `num_evals` questions, each prefixed with "The final result is {last observation}." and
    suffixed with the answer-format line, answered in `<answer>...</answer>`.
  * The goal's own `evaluate_predictions` scores the answers.
include_prior is True (conf/config.yaml).

Two things differ by construction and are stated: the original agent drops each evaluation Q/A
from its context before the next question, and here the harness owns its context, so whether it
keeps them is the harness's business; and this adapter runs only the final checkpoint, not the
0/1/3/5/7 ones.

score            = (raw - norm_mu) / norm_sigma, the standardisation of the paper's analysis
                   notebooks, with the goal's own norm factors
raw_score        = the first value `evaluate_predictions` returns
higher_is_better = True only for the two goals whose metric is an accuracy (hyperbolic `direct`,
                   survival `direct`); every other goal returns an error (MSE, |error|, 1 - acc)
success          = None (BoxingGym defines none)
An answer set the goal's scorer cannot evaluate (location/source raises on a wrong shape) leaves
the score NaN and the error in `eval_error`, instead of crashing the run. Likewise an experiment
input that BoxingGym's validator raises on (e.g. "1] (observation: 0)" -> float()) is returned to
the agent as an invalid input, uncounted, as rejected inputs already are.
"""
from __future__ import annotations

import os
import re

import numpy as np
import yaml

from mb.core import TextEnv, seeded

# BoxingGym draws everything (hidden parameters, observation noise, evaluation points) from the
# global numpy/random state. Each call below reseeds from (instance seed, call index) through
# core.seeded, so an instance replays identically whatever ran before it or alongside it.


def _pin_pymc():
    """pymc samplers draw from their own entropy-seeded RNG unless given `random_seed`. Default it
    to a draw from the (just reseeded) numpy state, so the environments that sample through pymc
    (IRT, survival, dugongs, peregrines) replay too. BoxingGym's own code is untouched."""
    import pymc as pm
    if getattr(pm, "_mb_pinned", False):
        return
    for name in ("sample_prior_predictive", "sample_posterior_predictive", "sample", "draw"):
        orig = getattr(pm, name)

        def wrapped(*a, _orig=orig, **k):
            if k.get("random_seed") is None:
                k["random_seed"] = int(np.random.randint(2 ** 31 - 1))
            return _orig(*a, **k)
        setattr(pm, name, wrapped)
    pm._mb_pinned = True


def _seeded(seed, salt, fn, *args):
    return seeded(seed * 1000 + salt, fn, *args)

BENCH = "boxinggym"
_CONF = os.path.expanduser(os.environ.get("BOXING_GYM_ROOT", "~/code/multibench/boxing-gym")) + "/conf/envs"
SUBTASKS = {  # subtask -> (config file, module, env class, goal class)
    "hyperbolic/direct": ("hyperbolic_direct", "hyperbolic_temporal_discount", "TemporalDiscount", "DirectGoal"),
    "hyperbolic/discount": ("hyperbolic_discount", "hyperbolic_temporal_discount", "TemporalDiscount", "DiscountGoal"),
    "location/direct": ("location_finding_direct", "location_finding", "Signal", "DirectGoal"),
    "location/source": ("location_finding_source", "location_finding", "Signal", "SourceGoal"),
    "death/direct": ("death_process_direct", "death_process", "DeathProcess", "DirectDeath"),
    "death/infection": ("death_process_infection", "death_process", "DeathProcess", "InfectionRate"),
    "irt/direct": ("irt_direct", "irt", "IRT", "DirectCorrectness"),
    "survival/direct": ("survival_direct", "survival_analysis", "SurvivalAnalysis", "DirectGoal"),
    "dugongs/direct": ("dugongs_direct", "dugongs", "Dugongs", "DirectGoal"),
    "peregrines/direct": ("peregrines_direct", "peregrines", "Peregrines", "DirectGoal"),
    "lotka_volterra/direct": ("lotka_volterra_direct", "lotka_volterra", "LotkaVolterra", "DirectGoal"),
}
_ACCURACY = {"hyperbolic/direct", "survival/direct"}
DIMENSIONS = {"world": "no", "hypotheses": "open (continuous model)", "observation": "simulator, noisy",
              "cost": "experiments", "stop": "fixed budget"}
RESPONDER = None
N_EXPERIMENTS = 10
_RANGES = {"search": 0, "selection": 100_000, "test": 1_000_000}

# agents/agent.py, verbatim
_FIRST = ("Think about where to observe next. Articulate your strategy for choosing measurements in "
          "<thought>.\nProvide a new measurement point in the format:\n<thought>your thought</thought>\n"
          "<observe> your observation</observe>\nMake an observation now.")
_NEXT = ("Result: {r}\nThink about where to observe next. Articulate your strategy for choosing "
         "measurements in <thought>.\nProvide a new measurement point in the format:\nThought: <thought>\n"
         "<observe> your observation (remember the type of inputs accepted)</observe>")
_RETRY_OBS = ("Please stick to the specified format and respond using <observe> tags. Continue making "
              "observations even if you think you have an accurate estimate. Your previous response was not valid.")
_RETRY_ANS = ("Please stick to the specified format and respond using <answer> tags. Make assumptions and "
              "provide your best guess. Your previous response was not valid.")
_ANSWER_FMT = "\nAnswer in the following format:\n<answer>your answer</answer>."


def _parse(reply, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", reply or "", re.S)
    if not m:
        return None
    r = m.group(1).strip()
    # agent.py's check: a parsed reply must contain a number and must not say "done"
    return r if re.findall(r"[0-9]+", r) and "done" not in r else None


def instances(subtask, split, limit=None):
    start = _RANGES[split]
    return [str(s) for s in range(start, start + (limit or 200))]


class BoxingGymEnv(TextEnv):
    reads_full_reply = True
    bench = BENCH

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        import importlib
        cfg_name, mod, env_cls, goal_cls = SUBTASKS[subtask]
        cfg = yaml.safe_load(open(f"{_CONF}/{cfg_name}.yaml"))
        _pin_pymc()
        M = importlib.import_module(f"boxing_gym.envs.{mod}")
        self._seed = seed = int(instance)

        def build():
            env = getattr(M, env_cls)(**cfg.get("env_params", {}))
            env.include_prior = True
            return env, getattr(M, goal_cls)(env)
        self._env, self._goal = _seeded(seed, 0, build)
        self.higher_is_better = subtask in _ACCURACY
        self._n_evals = int(cfg["num_evals"])
        self.max_steps = 4 * (N_EXPERIMENTS + self._n_evals)    # room for retries
        self.instructions = self._goal.get_system_message(True) + "\n\n" + _FIRST
        self._experiments = 0
        self._last_obs = None
        self._final = None
        self._preds, self._gts = [], []
        self._question = None
        self.raw_score = None
        self.eval_error = None      # set when the goal's own scorer rejects the answers

    def _next_question(self):
        q, gt = _seeded(self._seed, 500 + len(self._gts), self._goal.get_goal_eval_question, True)
        self._gts.append(gt)
        self._question = self._final + "\n" + q + _ANSWER_FMT
        return self._question

    def _act(self, action):
        if self._experiments < N_EXPERIMENTS:
            obs_in = _parse(action, "observe")
            if obs_in is None:
                return _RETRY_OBS
            try:
                obs, ok = _seeded(self._seed, 1 + self._experiments, self._env.run_experiment, obs_in)
            except Exception as e:     # BoxingGym's own validators raise on some malformed inputs
                obs, ok = f"Invalid input {obs_in!r}: {e}", False
            if not ok:
                return str(obs)
            self._experiments += 1
            self._last_obs = obs
            if self._experiments < N_EXPERIMENTS:
                return _NEXT.format(r=obs)
            self._final = f"The final result is {obs}."
            self._goal.eval_pointer = 0
            return self._next_question()
        ans = _parse(action, "answer")
        if ans is None:
            return _RETRY_ANS
        self._preds.append(ans)
        if len(self._preds) < self._n_evals:
            return self._next_question()
        self.done = True
        try:
            self.raw_score = float(self._goal.evaluate_predictions(self._preds, self._gts)[0])
        except Exception as e:      # e.g. location/source raises on an answer of the wrong shape
            self.eval_error = repr(e)
        return "Evaluation complete."

    def score(self):
        if self.raw_score is None:
            return float("nan")
        return (self.raw_score - self._goal.norm_mu) / self._goal.norm_sigma

    def success(self):
        return None

    def result(self):
        return {**super().result(), "raw_score": self.raw_score, "eval_error": self.eval_error,
                "experiments": self._experiments, "answers": len(self._preds)}

    def _oracle_next(self):
        if self._experiments < N_EXPERIMENTS:
            x = _seeded(self._seed, 900 + self._experiments, self._env.sample_random_input)
            if isinstance(x, (tuple, list, np.ndarray)):
                x = "[" + ", ".join(str(v) for v in np.ravel(x)) + "]"
            return f"<observe>{x}</observe>"
        gt = self._gts[-1]
        if isinstance(gt, (tuple, list, np.ndarray)):
            gt = str(np.asarray(gt).tolist())
        return f"<answer>{gt}</answer>"


def make(subtask, instance, responder=None):
    return BoxingGymEnv(subtask, instance, responder)
