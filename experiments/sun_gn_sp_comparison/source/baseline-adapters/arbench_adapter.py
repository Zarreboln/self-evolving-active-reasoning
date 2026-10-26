"""Local AR-Bench boundary adapter. No network client or model imports.

Loads pinned original definitions without their import/CLI side effects. The
caller supplies transport; importing this module performs no inference.
Native bugs are preserved and surfaced as execution_failure, never 0% scores.
"""
from __future__ import annotations

import ast
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import inspect
import io
import itertools
import json
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace
import typing

PIN = '9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1'
VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'arbench'
METHODS = ('zero_shot', 'few_shot', 'few_shot_inst', 'tot', 'proactive_cot', 'uot')
SOURCE_SHA256 = {
    'arbench/reasoner/gn/gn_evaluator.py': 'aa6643d004ed7b510452ce9fe581b371112616a6702dfee3a26132759cb221a9',
    'arbench/reasoner/gn/prompt.py': '5dafd196d5542fad163898fbfecca411fb13b1b30b636ebc0577d38a7bbcc713',
    'arbench/utils/utils_gn.py': 'f57cc0a77615859ea44fad07a0e5b849aa95e6d428963d72d3c49405c20274e7',
    'arbench/utils/utils_sp.py': '03c29648f308b697767009504ea3f9dd01009883e96a33c7e7707aea30125785',
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CallLimits:
    max_calls: int
    max_tokens_per_call: int

    def __post_init__(self):
        if type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError('max_calls must be a positive integer')
        if type(self.max_tokens_per_call) is not int or self.max_tokens_per_call < 1:
            raise ValueError('max_tokens_per_call must be a positive integer')


def _tree(path):
    data = (VENDOR / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256[path]:
        raise ValueError('Pinned reference hash mismatch: ' + path)
    return ast.parse(data.decode('utf-8'), filename=path)


def _definitions(path, namespace, *, names=None, assignments=False):
    nodes = []
    for node in _tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if names is None or node.name in names:
                nodes.append(node)
        elif assignments and isinstance(node, ast.Assign):
            # Prompt module contains literal strings only. Reject new executable code.
            ast.literal_eval(node.value)
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)


REPAIR_NAME = 'gn-deterministic-parser-repair-01'


def _environment(seed, inference, repair_mode=None, instrumentation=None):
    class RecordedRandom(random.Random):
        def _note(self, operation):
            caller = inspect.currentframe().f_back.f_back.f_code.co_name
            instrumentation.setdefault('rng_operations', []).append({'operation': operation, 'caller': caller})

        def choice(self, population):
            self._note('choice')
            return super().choice(population)

        def sample(self, population, k, **kwargs):
            self._note('sample')
            return super().sample(population, k, **kwargs)

    ns = {name: getattr(typing, name) for name in
          ('Any', 'Dict', 'Generator', 'List', 'Union', 'Optional', 'Tuple')}
    ns.update(Counter=Counter, random=RecordedRandom(seed), re=re, math=math,
              itertools=itertools, json=json, inference=inference,
              tqdm=lambda value: value, print=lambda *args, **kwargs: None,
              POLICY_API_KEY=None, POLICY_BASE_URL=None, MAX_RETRIES=5,
              TARGET_FEEDBACK=[4, 0], CORRECT_POSITION_SCORE=2,
              DIFFERENT_POSITION_SCORE=1)
    # Upstream writes accumulated logs after each case. Keep that incidental sink
    # in memory; no user path or file is opened by the evaluator.
    def log_sink(path, mode='r', *args, **kwargs):
        if path != '__memory_only_log__' or mode != 'w':
            raise ValueError('Unexpected upstream filesystem access')
        return io.StringIO()
    ns['open'] = log_sink
    _definitions('arbench/utils/utils_gn.py', ns)
    _definitions('arbench/reasoner/gn/prompt.py', ns, assignments=True)
    if repair_mode is None:
        _definitions('arbench/reasoner/gn/gn_evaluator.py', ns)
    else:
        if repair_mode != REPAIR_NAME:
            raise ValueError('Unknown parser repair')
        # Separate adapter AST overlay. The original source file and its hash
        # remain unchanged; native scoring assignments are not transformed.
        class NarrowRepair(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if (isinstance(node.func, ast.Name) and node.func.id == 'extract_four_digit_numbers'
                        and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == 'guess_response'):
                    node.func.id = '_repair_tot_candidates'
                if isinstance(node.func, ast.Name) and node.func.id == 'extract_and_convert_guess':
                    node.func.id = '_repair_checked_guess'
                return node

            def visit_FunctionDef(self, node):
                node = self.generic_visit(node)
                if node.name == 'select':
                    expanded = []
                    for statement in node.body:
                        expanded.append(statement)
                        if (isinstance(statement, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == 'final_number' for t in statement.targets)):
                            expanded.append(ast.Expr(value=ast.Call(func=ast.Name(id='_capture_tot_final', ctx=ast.Load()),
                                                                     args=[ast.Name(id='final_number', ctx=ast.Load())], keywords=[])))
                    node.body = expanded
                return node

        def checked_guess(text):
            digits = ns['extract_and_convert_guess'](text)
            if len(digits) != 4:
                raise ns['NotNumberError']()
            return digits

        ns['_repair_checked_guess'] = checked_guess
        ns['_repair_tot_candidates'] = lambda text: re.findall(r'[0-9]{4}', text)
        ns['_capture_tot_final'] = lambda value: instrumentation.update(tot_final_actor_digits=list(value))
        tree = _tree('arbench/reasoner/gn/gn_evaluator.py')
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        overlay = NarrowRepair().visit(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(ast.fix_missing_locations(overlay), REPAIR_NAME, 'exec'), ns)
    ns['METHOD_DICT'] = {
        'zero_shot': ns['propose_template'],
        'few_shot': ns['propose_template_with_1_shot'],
        'few_shot_inst': ns['propose_template_with_1_shot_inst'],
        'proactive_cot': ns['propose_template'],
    }
    return ns


def gn_call_bound(method, max_turn=25, simulate_depth=3, repair_mode=None):
    """Inference invocations, not HTTP retries, wall time, or GPU seconds."""
    if method not in METHODS or type(max_turn) is not int or max_turn < 1:
        raise ValueError('Invalid method or turns')
    if type(simulate_depth) is not int or simulate_depth < 0:
        raise ValueError('Invalid simulation depth')
    if method in ('zero_shot', 'few_shot', 'few_shot_inst'):
        return max_turn + 1
    if method == 'proactive_cot':
        return 2 * max_turn
    if method == 'tot':
        return 5 if repair_mode is None else 5 * max_turn + 1
    simulations = 0 if simulate_depth == 0 else 14
    for _ in range(3, simulate_depth + 1):
        simulations = 14 * (1 + simulations)
    return max_turn * (1 + 3 * simulations)


def _normalize_gn(native):
    guesses = native['guess_list']
    final = native.get('prediction') or (guesses[-1] if guesses else ['', 0, 0])
    return {
        'official_success': bool(native['correctness']),
        'final_EM': bool(final[1] == 4),
        'hit_at_any': any(g[1] == 4 for g in guesses),
        'logged_guesses': len(guesses),
        'final_answer': str(final[0]),
        'final_answer_definition': 'last submitted/logged harness guess, not necessarily actor-generated',
    }


def _natural_interactions(native, method, events):
    guesses = native['guess_list']
    if method in ('zero_shot', 'few_shot', 'few_shot_inst'):
        # The final prediction is scored privately, not answered as a question.
        return guesses[:-1]
    if method == 'uot':
        queries = sum(e['metadata'].get('upstream_caller') == '_run_uot_evaluation'
                      for e in events)
        # A singleton may be automatically appended without an actor query.
        return guesses[:queries]
    return guesses


def run_gn_episode(*, method, secret, chat, seed, limits, max_turn=25,
                   temperature=0.7, top_p=0.7, simulate_depth=3,
                   memory_context='', case_id=None, repair_mode=None,
                   matched_finalization=False):
    """Run one original GN method with an injected, caller-owned transport.

    chat(messages, max_tokens, seed, metadata) returns text, finish_reason,
    usage {prompt_tokens, completion_tokens}. The caller must enforce timeout,
    zero HTTP retries, model identity, token cap, and outer allocation budget.
    These explicit transport caps differ from unbounded upstream defaults.
    The adapter does not retry a failed case or create any server/process.
    """
    if method not in METHODS:
        raise ValueError('Unsupported method')
    if not isinstance(secret, str) or re.fullmatch(r'[0-9]{4}', secret) is None or len(set(secret)) != 4:
        raise ValueError('secret must be four distinct ASCII digits, preserving leading zero')
    if type(seed) is not int or not isinstance(limits, CallLimits):
        raise ValueError('Explicit integer seed and CallLimits required')
    if repair_mode not in (None, REPAIR_NAME):
        raise ValueError('Unknown parser repair')
    gn_call_bound(method, max_turn, simulate_depth, repair_mode)
    events = []
    attempted_calls = 0
    phase = 'native_algorithm'

    def inference(messages, **kwargs):
        nonlocal attempted_calls
        if attempted_calls >= limits.max_calls:
            raise BudgetExceeded('Model call cap reached before next transport call')
        attempted_calls += 1
        submitted = copy.deepcopy(messages)
        if memory_context:
            submitted.append({'role': 'system', 'content': 'Learned context:\n' + str(memory_context)})
        metadata = {
            'task': 'GN', 'method': method, 'call_index': attempted_calls,
            'temperature': kwargs.get('temperature', temperature),
            'top_p': kwargs.get('top_p', top_p),
            'json_format': kwargs.get('json_format', False),
            'memory_injected': bool(memory_context),
            'phase': phase,
            'upstream_caller': inspect.currentframe().f_back.f_code.co_name,
        }
        event = {'call_index': attempted_calls, 'messages': submitted, 'metadata': metadata,
                 'response': None}
        events.append(event)
        response = chat(messages=copy.deepcopy(submitted),
                        max_tokens=limits.max_tokens_per_call, seed=seed,
                        metadata=copy.deepcopy(metadata))
        if not isinstance(response, dict) or not isinstance(response.get('text'), str):
            raise ValueError('Transport must return a text response')
        usage = response.get('usage')
        if not isinstance(usage, dict):
            raise ValueError('Transport must return usage')
        for key in ('prompt_tokens', 'completion_tokens'):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError('Invalid token usage: ' + key)
        event['response'] = copy.deepcopy(response)
        if usage['completion_tokens'] > limits.max_tokens_per_call:
            raise BudgetExceeded('Transport exceeded requested output-token cap')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response['text']))],
            usage=SimpleNamespace(**usage))

    instrumentation = {}
    ns = _environment(seed, inference, repair_mode, instrumentation)
    logs = []
    error = None
    try:
        args = ([secret], logs, '__memory_only_log__', 'caller_frozen_model')
        if method == 'tot':
            ns['_run_tot_evaluation'](*args, max_turn, temperature, top_p)
        elif method == 'proactive_cot':
            ns['_run_proactive_cot_evaluation'](*args, max_turn, temperature, top_p)
        elif method == 'uot':
            ns['_run_uot_evaluation'](*args, max_turn, temperature, top_p, simulate_depth)
        else:
            ns['_run_traditional_evaluation'](*args, method, max_turn, temperature, top_p)
        if matched_finalization:
            # Explicit additional variant: one actor answer after the native run.
            # Feed only questions and natural oracle replies already observed.
            phase = 'matched_finalization_01'
            history = '\n'.join(
                'Guess ' + str(g[0]) + ': ' + str(g[1]) + ' correct positions, ' + str(g[2]) + ' correct digits in wrong positions.'
                for g in _natural_interactions(logs[0], method, events))
            messages = [
                {'role': 'system', 'content': ns['propose_template'].format(turn=max_turn)},
                {'role': 'user', 'content': history + '\n' + ns['final_guess_prompt']},
            ]
            final_response = inference(messages, temperature=temperature, top_p=top_p)
            final_digits = ns['extract_and_convert_guess'](final_response.choices[0].message.content)
            instrumentation['matched_final_actor_digits'] = final_digits
            instrumentation['matched_final_actor_text'] = final_response.choices[0].message.content
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    status = 'complete' if len(logs) == 1 and error is None else 'execution_failure'
    native = logs[0] if len(logs) == 1 else None
    metrics = _normalize_gn(native) if native is not None else None
    if metrics is not None:
        metrics['query_count_definition'] = 'logged guesses include final guess and UoT automatic singleton completion; not total simulation/oracle operations'
        metrics['selected_path_environment_queries'] = len(_natural_interactions(native, method, events))
        metrics['selected_path_count_caveat'] = 'ToT also evaluates candidate branches against the oracle; selected path count is not total search-oracle cost'
        metrics['harness_final_EM'] = metrics['final_EM']
        metrics['actor_final_EM'] = None
        metrics['actor_final_reason'] = 'not uniformly identifiable from native method output; do not impute actor success from harness singleton completion'
        if method in ('zero_shot', 'few_shot', 'few_shot_inst'):
            metrics['actor_final_EM'] = metrics['final_EM']
            metrics['actor_final_reason'] = 'explicit final actor response parsed by original method'
        elif 'tot_final_actor_digits' in instrumentation:
            metrics['actor_final_EM'] = instrumentation['tot_final_actor_digits'] == [int(x) for x in secret]
            metrics['actor_final_reason'] = 'discarded native final_number captured without changing native history-based scoring'
        if matched_finalization:
            digits = instrumentation.get('matched_final_actor_digits')
            metrics['matched_final_actor_EM'] = (digits == [int(x) for x in secret]) if digits is not None else None
            metrics['matched_final_answer'] = ''.join(str(x) for x in digits) if digits is not None else None
    return {
        'schema_version': 1, 'task': 'GN', 'method': method, 'case_id': case_id,
        'seed': seed, 'status': status, 'error': error,
        'native_status': 'complete' if native is not None else 'execution_failure',
        'variant_suffix': '_matched_finalization_01' if matched_finalization else '',
        'source_commit': PIN, 'source_sha256': SOURCE_SHA256,
        'parser_repair': repair_mode,
        'run_kind': 'adapted_context_run' if memory_context else 'native_algorithm_with_explicit_transport_caps',
        'official_result': native, 'metrics': metrics, 'events': events,
        'accounting': {
            'attempted_model_calls': attempted_calls,
            'completed_model_calls': sum(e['response'] is not None for e in events),
            'input_tokens': sum(e['response']['usage']['prompt_tokens'] for e in events if e['response']),
            'output_tokens': sum(e['response']['usage']['completion_tokens'] for e in events if e['response']),
            'truncated_responses': sum(e['response'].get('finish_reason') == 'length' for e in events if e['response']),
            'rng_operations': instrumentation.get('rng_operations', []),
            'proactive_random_fallbacks': sum(e['caller'] == '_run_proactive_cot_evaluation' for e in instrumentation.get('rng_operations', [])),
            'uot_random_candidate_fills': sum(e['caller'] == '_run_uot_evaluation' for e in instrumentation.get('rng_operations', [])),
            'requested_max_calls': limits.max_calls,
            'requested_max_output_tokens_per_call': limits.max_tokens_per_call,
            'model_call_bound': gn_call_bound(method, max_turn, simulate_depth, repair_mode) + int(matched_finalization),
            'max_turn': max_turn, 'simulate_depth': simulate_depth,
            'seed_policy': 'same explicit episode seed for every model call; local random.Random uses same seed',
            'budget_guarantee': 'call cap only; transport/outer runner must enforce time, token, retries and allocation constraints',
        },
    }


def public_projection(result):
    """Label-free update input; never passes secret, metrics, IDs or evaluator log."""
    trajectory = []
    for event in result['events']:
        # Complete actor-visible request/response trace. This contains repeated
        # context by design, and only feedback already presented to the actor.
        trajectory.extend(copy.deepcopy(event['messages']))
        if event['response'] is not None:
            trajectory.append({'role': 'assistant', 'content': event['response']['text']})
    prediction = ''
    if result.get('metrics') is not None:
        prediction = result['metrics'].get('matched_final_answer') or result['metrics']['final_answer']
    return {'task': 'GN',
            'question': 'Guess a four-digit secret with unique digits from 0..9.',
            'trajectory': trajectory, 'predicted_answer': prediction}


def compact_zero_shot_public_projection(result):
    """Once-per-interaction GN zero-shot dialogue for DC/ACE training.

    Reconstruct append-only native request deltas, retaining each complete actor
    response and every already-presented public message without truncation. The
    identical learned-context instruction appended to each request is retained
    once, in its first-request position. Raw events are never changed. Refuse
    alternate methods, incomplete runs, or a history shape this mapping cannot
    prove lossless. No evaluator log, secret, score, ID or metrics are accessed.
    """
    if (result.get('task') != 'GN' or result.get('method') != 'zero_shot'
            or result.get('status') != 'complete' or result.get('variant_suffix')):
        raise ValueError('Compact projection requires completed native zero_shot')
    events = result['events']
    turns = result['accounting']['max_turn']
    if type(turns) is not int or turns < 1 or len(events) != turns + 1:
        raise ValueError('Unexpected traditional event count')
    native_history, trajectory = [], []
    first_context = None
    context_present = None
    final_text = None
    for index, event in enumerate(events):
        meta = event['metadata']
        if (meta.get('task') != 'GN' or meta.get('method') != 'zero_shot'
                or meta.get('phase') != 'native_algorithm'
                or meta.get('upstream_caller') != '_run_traditional_evaluation'):
            raise ValueError('Unexpected actor event in compact projection')
        request = copy.deepcopy(event['messages'])
        if any(set(m) != {'role', 'content'} or not isinstance(m['content'], str)
               for m in request):
            raise ValueError('Unexpected public message schema')
        injected = meta.get('memory_injected')
        if type(injected) is not bool:
            raise ValueError('Missing explicit memory-injection marker')
        context = None
        if injected:
            if (not request or request[-1]['role'] != 'system'
                    or not request[-1]['content'].startswith('Learned context:\n')):
                raise ValueError('Unexpected learned-context injection')
            context = request.pop()
        if index == 0:
            first_context, context_present = context, injected
        elif injected != context_present or context != first_context:
            raise ValueError('Learned context changed within episode')
        if request[:len(native_history)] != native_history:
            raise ValueError('Native dialogue ceased to be append-only')
        delta = request[len(native_history):]
        expected_roles = ['system', 'user'] if index == 0 else ['user', 'user']
        if [m['role'] for m in delta] != expected_roles:
            raise ValueError('Unexpected traditional feedback/turn message shape')
        response = event['response']
        if not isinstance(response, dict) or not isinstance(response.get('text'), str):
            raise ValueError('Missing full actor response')
        final_text = response['text']
        answer = {'role': 'assistant', 'content': final_text}
        trajectory.extend(copy.deepcopy(delta))
        if index == 0 and context is not None:
            trajectory.append(copy.deepcopy(context))
        trajectory.append(copy.deepcopy(answer))
        native_history = request + [answer]
    # Identical text-only rule to pinned utils_gn.extract_and_convert_guess:
    # last four-digit match, integer digits normalized back to text. The raw
    # final answer and all reasoning remain in trajectory even if unparseable.
    matches = re.findall(r'\d{4}', final_text)
    prediction = ''.join(str(int(c)) for c in matches[-1]) if matches else ''
    return {'task': 'GN',
            'question': 'Guess a four-digit secret with unique digits from 0..9.',
            'trajectory': trajectory, 'predicted_answer': prediction}


def score_sp_prediction(prediction, hidden_bottom):
    """Scoring-only endpoint. Never expose hidden_bottom or this score to updater."""
    if not isinstance(prediction, str) or not isinstance(hidden_bottom, str):
        raise TypeError('SP predictions and references must be strings')
    ns = {'Counter': Counter, 'List': typing.List}
    _definitions('arbench/utils/utils_sp.py', ns, names={'f1_score'})
    return {'official_f1_char': ns['f1_score'](prediction, hidden_bottom),
            'diagnostic_f1_word': ns['f1_score'](prediction.split(), hidden_bottom.split())}


def prepare_sp_interface(*, method, actor_model, referee_model, referee_endpoint_verified=False):
    if method not in METHODS:
        raise ValueError('Unsupported method')
    return {
        'task': 'SP', 'method': method, 'source_commit': PIN,
        'actor_model': actor_model, 'referee_model': referee_model,
        'status': 'interface_only_not_executable',
        'requires': ['independent Llama-3.1-405B referee endpoint',
                     'explicitly disclosed and calibrated quantization if substituted',
                     'separate policy and response transport',
                     'reviewed caller budget and every-method termination bounds'],
        'referee_claim_verified_by_caller': bool(referee_endpoint_verified),
        'self_judging_allowed': False,
        'blocker': 'No SP execution implemented or authorized by this local interface',
    }
