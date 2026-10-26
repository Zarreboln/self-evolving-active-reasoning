"""Injected GN comparison controller; no provider, model, CLI or job launcher.

The caller supplies a frozen protocol, private cases, transport, real tokenizer
and explicit admission receipt. Importing this file cannot start a real run.
All output is private evidence, not a sanitized publication bundle.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import traceback

HERE = Path(__file__).resolve().parent
for folder in ('baseline-adapters', 'method-adapters'):
    location = str(HERE / folder)
    if location not in sys.path:
        sys.path.insert(0, location)
BASELINE = importlib.import_module('arbench_adapter')
METHODS = importlib.import_module('method_adapters')


class SharedBoundaryHalt(RuntimeError):
    """Any uncertain transport or integrity/budget failure stops all arms."""


class TrainingFailure(RuntimeError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(where + ' has unexpected or missing fields')


def validate_protocol(protocol, cases):
    """Return a copied protocol plus conservative whole-plan request bounds."""
    p = json.loads(_json(protocol))
    _keys(p, ('schema_version', 'task', 'run_kind', 'splits', 'checkpoints',
              'arms', 'actor', 'dc', 'ace', 'seeds', 'budget'), 'protocol')
    if p['schema_version'] != 1 or p['task'] != 'GN':
        raise ValueError('Only GN protocol schema 1 is supported')
    if p['run_kind'] not in ('synthetic', 'research_run'):
        raise ValueError('Explicit synthetic or research_run label required')
    _keys(p['splits'], ('train', 'dev', 'final'), 'splits')
    expected_sizes = {'train': 6, 'dev': 5, 'final': 20}
    all_ids = []
    for split, size in expected_sizes.items():
        ids = p['splits'][split]
        if not isinstance(ids, list) or len(ids) != size:
            raise ValueError(split + ' must explicitly contain ' + str(size) + ' case IDs')
        if any(not isinstance(x, str) or not x for x in ids):
            raise ValueError('Case IDs must be nonempty strings')
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Train/dev/final case IDs must be disjoint')
    if not isinstance(cases, dict) or set(cases) != set(all_ids):
        raise ValueError('Supply exactly the frozen private case mapping')
    for secret in cases.values():
        if not isinstance(secret, str) or not re.fullmatch(r'[0-9]{4}', secret) or len(set(secret)) != 4:
            raise ValueError('Each private case must be a distinct-digit four-digit string')
    if len(set(cases.values())) != len(cases):
        raise ValueError('Distinct IDs must not reuse a secret across the frozen splits')
    if p['checkpoints'] != [0, 3, 6]:
        raise ValueError('This comparison requires explicit checkpoints [0,3,6]')
    if not isinstance(p['arms'], list) or not p['arms']:
        raise ValueError('The complete selected arm matrix must be explicit')
    names = []
    for arm in p['arms']:
        _keys(arm, ('name', 'kind', 'baseline_method'), 'arm')
        if not isinstance(arm['name'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', arm['name']):
            raise ValueError('Arm names must be safe stable identifiers')
        names.append(arm['name'])
        if arm['kind'] not in ('baseline', 'DC', 'ACE'):
            raise ValueError('Unknown arm kind')
        if arm['baseline_method'] not in BASELINE.METHODS:
            raise ValueError('Unknown native baseline method')
        if arm['kind'] != 'baseline' and arm['baseline_method'] != 'zero_shot':
            raise ValueError('DC and ACE must share the same native zero_shot actor')
    if len(set(names)) != len(names):
        raise ValueError('Arm names must be unique')
    if sum(a['kind'] == 'DC' for a in p['arms']) != 1 or sum(a['kind'] == 'ACE' for a in p['arms']) != 1:
        raise ValueError('Explicitly select exactly one DC and one ACE arm')
    if not any(a['kind'] == 'baseline' and a['baseline_method'] == 'zero_shot' for a in p['arms']):
        raise ValueError('The matched zero_shot baseline must be explicitly selected')
    _keys(p['actor'], ('max_turn', 'max_tokens', 'temperature', 'top_p',
                       'simulate_depth', 'repair_mode'), 'actor')
    for name in ('max_turn', 'max_tokens'):
        _positive('actor.' + name, p['actor'][name])
    if type(p['actor']['simulate_depth']) is not int or p['actor']['simulate_depth'] < 0:
        raise ValueError('simulate_depth must be a nonnegative integer')
    for name in ('temperature', 'top_p'):
        if type(p['actor'][name]) not in (int, float):
            raise TypeError('Sampling values must be numeric')
    if not 0 <= p['actor']['temperature'] <= 2 or not 0 < p['actor']['top_p'] <= 1:
        raise ValueError('Invalid actor sampling')
    if p['actor']['repair_mode'] not in (None, BASELINE.REPAIR_NAME):
        raise ValueError('Unknown explicit parser variant')
    _keys(p['dc'], ('temperature',), 'dc')
    if type(p['dc']['temperature']) not in (int, float) or not 0 <= p['dc']['temperature'] <= 2:
        raise ValueError('Invalid DC temperature')
    _keys(p['ace'], ('reflector_max_tokens', 'curator_max_tokens', 'max_num_rounds',
                     'curator_frequency', 'token_budget', 'no_ground_truth', 'use_json_mode'), 'ace')
    for name in ('reflector_max_tokens', 'curator_max_tokens', 'max_num_rounds',
                 'curator_frequency', 'token_budget'):
        _positive('ace.' + name, p['ace'][name])
    for name in ('no_ground_truth', 'use_json_mode'):
        if type(p['ace'][name]) is not bool:
            raise TypeError('ACE flags must be bool')
    _keys(p['seeds'], ('actor', 'update'), 'seeds')
    if any(type(x) is not int for x in p['seeds'].values()):
        raise TypeError('Explicit integer seed bases are required')
    _keys(p['budget'], ('max_calls', 'max_output_tokens', 'default_uot_in_scope'), 'budget')
    for name in ('max_calls', 'max_output_tokens'):
        _positive('budget.' + name, p['budget'][name])
    if type(p['budget']['default_uot_in_scope']) is not bool:
        raise TypeError('default_uot_in_scope must be explicit bool')
    if any(a['baseline_method'] == 'uot' for a in p['arms']) and p['actor']['simulate_depth'] == 3:
        if not p['budget']['default_uot_in_scope']:
            raise ValueError('Default-depth UoT requires explicit cost-scope inclusion')
    bounds = {a['name']: BASELINE.gn_call_bound(a['baseline_method'],
               p['actor']['max_turn'], p['actor']['simulate_depth'], p['actor']['repair_mode'])
              for a in p['arms']}
    # Static baselines are evaluated only on the final 20 cases. Development
    # checkpoints assess the two learning methods and never select a policy.
    learning_names = [a['name'] for a in p['arms'] if a['kind'] != 'baseline']
    evaluation_calls = 20 * sum(bounds.values()) + 5 * 3 * sum(bounds[name] for name in learning_names)
    b = BASELINE.gn_call_bound('zero_shot', p['actor']['max_turn'],
                               p['actor']['simulate_depth'], p['actor']['repair_mode'])
    r = p['ace']['max_num_rounds']
    curator_calls = sum(step % p['ace']['curator_frequency'] == 0 for step in range(1, 7))
    actor_calls = evaluation_calls + 6 * b + 6 * (r + 2) * b
    role_calls = {'actor': actor_calls, 'DC_curator': 6,
                  'ACE_reflector': 6 * r, 'ACE_curator': curator_calls}
    output_bound = (actor_calls * p['actor']['max_tokens'] +
                    12 * p['actor']['max_tokens'] +
                    6 * r * p['ace']['reflector_max_tokens'] +
                    curator_calls * p['ace']['curator_max_tokens'])
    plan = {'episode_call_bounds': bounds, 'role_call_bounds': role_calls,
            'max_calls': sum(role_calls.values()), 'max_output_tokens': output_bound,
            'evaluation_slots': len(p['arms']) * 20 + 30, 'training_updates': 12,
            'evaluation_matrix': {'dev': {'arms': learning_names, 'checkpoints': [0, 3, 6], 'cases_each': 5},
                                  'final': {'arms': [a['name'] for a in p['arms']], 'checkpoint': 6, 'cases': 20}},
            'seed_derivation': 'sha256(canonical JSON [base, key...]) first 8 hex modulo 2147483647; actor evaluation keys omit arm/checkpoint',
            'training_projection': 'compact_zero_shot_public_projection',
            'selected_matrix': p['arms'],
            'unselected_native_methods': [m for m in BASELINE.METHODS if not any(
                a['kind'] == 'baseline' and a['baseline_method'] == m for a in p['arms'])]}
    if p['budget']['max_calls'] < plan['max_calls'] or p['budget']['max_output_tokens'] < output_bound:
        raise ValueError('Declared budget does not cover the full explicit worst-case plan')
    return p, plan


def seed_value(base, *parts):
    return int(_hash([base, *parts])[:8], 16) % 2147483647


class Evidence:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.index = 0
        self.hashes = {}

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(data)
        sha = hashlib.sha256(data.encode('utf-8')).hexdigest()
        self.hashes[relative] = sha
        return {'path': relative, 'sha256': sha}

    def event(self, kind, payload):
        self.index += 1
        entry = {'sequence': self.index, 'kind': kind, 'payload': payload}
        with (self.root / 'journal.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(_json(entry) + '\n')
        return entry


def run_comparison(protocol, cases, *, chat, count_tokens, admission, output_dir):
    """Run only through caller-injected capabilities; no real default transport.

    admission(protocol_copy, plan_copy) -> {admitted: bool, ...receipt fields}.
    Any transport exception, malformed usage, or budget excess sets a shared
    halt flag, even if a native evaluator catches the exception internally.
    The tokenizer callback is supplied to the official ACE bookkeeping. Actual
    chat-template context admission belongs to the caller's shared transport.
    """
    p, plan = validate_protocol(protocol, cases)
    if not all(callable(x) for x in (chat, count_tokens, admission)):
        raise TypeError('Explicit chat, tokenizer and admission callbacks required')
    cases = json.loads(_json(cases))
    evidence = Evidence(output_dir)
    protocol_ref = evidence.save('protocol.json', p)
    evidence.save('plan.json', plan)
    evidence.save('private-cases.json', cases)
    receipt = admission(json.loads(_json(p)), json.loads(_json(plan)))
    if not isinstance(receipt, dict) or type(receipt.get('admitted')) is not bool:
        raise ValueError('Admission must return an explicit admitted bool')
    evidence.save('admission.json', receipt)
    evidence.event('admission', receipt)
    if not receipt['admitted']:
        result = {'status': 'not_admitted', 'run_kind': p['run_kind'], 'model_calls': 0,
                  'protocol': protocol_ref, 'admission': receipt}
        evidence.save('result.json', result)
        return result
    state = {'halt': None, 'attempted': 0, 'completed': 0,
             'input_tokens': 0, 'output_tokens': 0, 'roles': Counter(),
             'attempted_roles': Counter(), 'reported_input_tokens': 0,
             'reported_output_tokens': 0}
    memories = {a['name']: METHODS.initial_memory(a['kind'])
                for a in p['arms'] if a['kind'] != 'baseline'}
    evaluations, updates = [], []
    frozen = {}
    episode_index = 0

    def boundary(scope):
        def call(*, messages, max_tokens, seed, metadata):
            if state['halt'] is not None:
                raise SharedBoundaryHalt('Shared boundary already halted')
            request = {'scope': scope, 'messages': messages, 'max_tokens': max_tokens,
                       'seed': seed, 'metadata': dict(metadata, **scope)}
            attempt = state['attempted'] + 1
            response = None
            try:
                _positive('requested max_tokens', max_tokens)
                if state['attempted'] >= p['budget']['max_calls']:
                    raise SharedBoundaryHalt('Global request cap would be exceeded')
                if state['output_tokens'] + max_tokens > p['budget']['max_output_tokens']:
                    raise SharedBoundaryHalt('Global output-token reserve would be exceeded')
                state['attempted'] += 1
                role = scope['arm'] + '/' + metadata.get('role', 'actor')
                state['attempted_roles'][role] += 1
                evidence.event('call_started', {'attempt': attempt, **request})
                response = chat(messages=json.loads(_json(messages)), max_tokens=max_tokens,
                                seed=seed, metadata=json.loads(_json(request['metadata'])))
                if not isinstance(response, dict) or not isinstance(response.get('text'), str):
                    raise SharedBoundaryHalt('Invalid transport response text')
                if 'finish_reason' not in response or not isinstance(response.get('usage'), dict):
                    raise SharedBoundaryHalt('Missing transport finish_reason or usage')
                usage = response['usage']
                for field in ('prompt_tokens', 'completion_tokens'):
                    if type(usage.get(field)) is not int or usage[field] < 0:
                        raise SharedBoundaryHalt('Invalid token usage: ' + field)
                # Preserve observed usage even when the response violates its cap.
                state['reported_input_tokens'] += usage['prompt_tokens']
                state['reported_output_tokens'] += usage['completion_tokens']
                if usage['completion_tokens'] > max_tokens:
                    raise SharedBoundaryHalt('Transport exceeded requested output cap')
                state['completed'] += 1
                state['input_tokens'] += usage['prompt_tokens']
                state['output_tokens'] += usage['completion_tokens']
                state['roles'][role] += 1
                evidence.save('calls/' + str(attempt).zfill(6) + '.json',
                              {'attempt': attempt, **request, 'response': response})
                evidence.event('call_completed', {'attempt': attempt, 'role': role, 'usage': usage})
                return response
            except Exception as exc:
                state['halt'] = {'type': type(exc).__name__, 'message': str(exc),
                                 'attempt': attempt, 'scope': scope,
                                 'traceback': traceback.format_exc()}
                evidence.save('calls/failed-' + str(attempt).zfill(6) + '.json',
                              {'attempt': attempt, **request, 'response': response,
                               'error': state['halt']})
                evidence.event('global_halt', state['halt'])
                raise SharedBoundaryHalt(str(exc)) from exc
        return call

    def actor(arm, case_id, phase, checkpoint, memory, seed, *, reflection='(empty)', call_id=None):
        nonlocal episode_index
        if state['halt'] is not None:
            raise SharedBoundaryHalt('No new episode after global halt')
        episode_index += 1
        context = METHODS.frozen_context(arm['kind'], memory) if arm['kind'] != 'baseline' else ''
        if reflection != '(empty)':
            context += '\nReflection:\n' + reflection
        scope = {'arm': arm['name'], 'comparison_phase': phase, 'checkpoint': checkpoint,
                 'case_id': case_id, 'generator_call_id': call_id,
                 'oracle_training_feedback': False}
        evidence.event('episode_started', {**scope, 'memory_sha256': METHODS.digest(memory),
                                          'episode_index': episode_index})
        result = BASELINE.run_gn_episode(
            method=arm['baseline_method'], secret=cases[case_id],
            chat=boundary(scope), seed=seed,
            limits=BASELINE.CallLimits(plan['episode_call_bounds'][arm['name']], p['actor']['max_tokens']),
            max_turn=p['actor']['max_turn'], temperature=p['actor']['temperature'],
            top_p=p['actor']['top_p'], simulate_depth=p['actor']['simulate_depth'],
            memory_context=context, case_id=case_id, repair_mode=p['actor']['repair_mode'],
            matched_finalization=False)
        ref = evidence.save('episodes/' + str(episode_index).zfill(5) + '.json', result)
        evidence.event('episode_completed', {**scope, 'result': ref, 'status': result['status']})
        if state['halt'] is not None:
            raise SharedBoundaryHalt('Native evaluator caught a shared-boundary failure')
        if result['status'] != 'complete' and result['error'].get('type') not in (
                'ValueError', 'IndexError', 'NotNumberError'):
            state['halt'] = {'type': 'UnclassifiedNativeFailure', 'native_error': result['error'],
                             'scope': scope, 'message': 'Only known deterministic native/parser failures may continue'}
            evidence.event('global_halt', state['halt'])
            raise SharedBoundaryHalt('Unclassified native failure')
        if phase == 'train' and result['status'] != 'complete':
            raise TrainingFailure('Training episode failed without replacement: ' + _json(result['error']))
        return result, ref

    def rotated(values, offset):
        k = offset % len(values)
        return values[k:] + values[:k]

    def evaluate(split, checkpoint):
        before = dict(memories)
        evaluated_arms = p['arms'] if split == 'final' else [a for a in p['arms'] if a['kind'] != 'baseline']
        for case_index, case_id in enumerate(p['splits'][split]):
            order = rotated(evaluated_arms, case_index + checkpoint)
            evidence.event('evaluation_arm_order', {'split': split, 'checkpoint': checkpoint,
                                                    'case_id': case_id,
                                                    'arms': [a['name'] for a in order]})
            for arm in order:
                memory = memories.get(arm['name'], '')
                result, ref = actor(arm, case_id, split, checkpoint, memory,
                    seed_value(p['seeds']['actor'], 'evaluation', split, case_id))
                evaluations.append({'split': split, 'checkpoint': checkpoint, 'arm': arm['name'],
                    'case_id': case_id, 'status': result['status'], 'metrics': result['metrics'],
                    'error': result['error'], 'episode': ref, 'memory_sha256': METHODS.digest(memory)})
        if memories != before:
            raise AssertionError('Evaluation changed a memory state')

    status, failure = 'complete', None
    try:
        for name, memory in memories.items():
            evidence.save('states/' + name + '-initial.json', {'memory': memory, 'sha256': METHODS.digest(memory)})
        evaluate('dev', 0)
        learning_arms = [a for a in p['arms'] if a['kind'] != 'baseline']
        for step, case_id in enumerate(p['splits']['train'], 1):
            for arm in rotated(learning_arms, step - 1):
                memory = memories[arm['name']]
                initial_seed = seed_value(p['seeds']['actor'], 'train', case_id, 'initial')
                initial, initial_ref = actor(arm, case_id, 'train', step, memory, initial_seed,
                                             call_id='train_step_' + str(step) + '_gen_initial')
                public = BASELINE.compact_zero_shot_public_projection(initial)
                scope = {'arm': arm['name'], 'comparison_phase': 'train', 'checkpoint': step,
                         'case_id': case_id, 'oracle_training_feedback': arm['kind'] == 'ACE'}
                if arm['kind'] == 'DC':
                    update = METHODS.dc_step(memory, public, chat=boundary(scope),
                        max_tokens=p['actor']['max_tokens'], phase='train',
                        temperature=p['dc']['temperature'],
                        seed=seed_value(p['seeds']['update'], 'DC', case_id, step))
                else:
                    # Private closure binds the exact same secret for every retry.
                    def regenerate(*, memory_context, reflection, seed, metadata):
                        # The adapter already rendered memory_context; use it verbatim.
                        nonlocal episode_index
                        if state['halt'] is not None:
                            raise SharedBoundaryHalt('No regeneration after boundary halt')
                        episode_index += 1
                        combined = memory_context
                        if reflection != '(empty)':
                            combined += '\nReflection:\n' + reflection
                        regen_scope = dict(scope, oracle_training_feedback=False,
                                           generator_call_id=metadata['call_id'])
                        evidence.event('episode_started', {**regen_scope, 'episode_index': episode_index,
                                                           'memory_context_sha256': METHODS.digest(memory_context)})
                        regenerated = BASELINE.run_gn_episode(
                            method='zero_shot', secret=cases[case_id], chat=boundary(regen_scope),
                            seed=seed, limits=BASELINE.CallLimits(plan['episode_call_bounds'][arm['name']],
                                                                 p['actor']['max_tokens']),
                            max_turn=p['actor']['max_turn'], temperature=p['actor']['temperature'],
                            top_p=p['actor']['top_p'], simulate_depth=p['actor']['simulate_depth'],
                            memory_context=combined, case_id=case_id, repair_mode=p['actor']['repair_mode'],
                            matched_finalization=False)
                        ref = evidence.save('episodes/' + str(episode_index).zfill(5) + '.json', regenerated)
                        evidence.event('episode_completed', {**regen_scope, 'result': ref,
                                                            'status': regenerated['status']})
                        if state['halt'] is not None:
                            raise SharedBoundaryHalt('Regeneration caught boundary failure')
                        if regenerated['status'] != 'complete':
                            raise TrainingFailure('ACE regeneration failed without replacement: ' + _json(regenerated['error']))
                        return BASELINE.compact_zero_shot_public_projection(regenerated)

                    def seed_for(role, call_id):
                        if role == 'generator' and call_id.endswith('_gen_initial'):
                            return initial_seed
                        return seed_value(p['seeds']['update'], 'ACE', case_id, role, call_id)

                    update = METHODS.ace_step(memory, public, training_target=cases[case_id],
                        answer_is_correct=lambda prediction, target: prediction == target,
                        regenerate=regenerate, chat=boundary(scope), count_tokens=count_tokens,
                        seed_for=seed_for, phase='train', step=step, total_samples=6, **p['ace'])
                if state['halt'] is not None:
                    raise SharedBoundaryHalt('Update caught shared boundary failure')
                update['record']['public_projection'] = 'compact_zero_shot_public_projection'
                memories[arm['name']] = update['memory']
                ref = evidence.save('updates/' + arm['name'] + '-' + str(step) + '.json', update)
                snapshot = evidence.save('states/' + arm['name'] + '-step-' + str(step) + '.json',
                    {'memory': update['memory'], 'sha256': METHODS.digest(update['memory'])})
                updates.append({'arm': arm['name'], 'step': step, 'case_id': case_id,
                                'initial_episode': initial_ref, 'update': ref, 'state': snapshot})
                evidence.event('memory_updated', updates[-1])
            if step in p['checkpoints']:
                if step == 6:
                    for name, memory in memories.items():
                        frozen[name] = evidence.save('states/' + name + '-final.json',
                            {'memory': memory, 'sha256': METHODS.digest(memory), 'training_complete': True})
                    evidence.event('all_final_memories_frozen', frozen)
                evaluate('dev', step)
        if len(frozen) != len(memories):
            raise AssertionError('Final evaluation requires all final memory files')
        evaluate('final', 6)
    except Exception as exc:
        status = 'global_halt' if state['halt'] is not None else 'incomplete_training_or_controller_failure'
        failure = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        evidence.event('run_failed', failure)
    failure_count = sum(x['status'] != 'complete' for x in evaluations)
    if status == 'complete' and failure_count:
        status = 'complete_schedule_with_execution_failures'
    result = {'schema_version': 1, 'task': 'GN', 'run_kind': p['run_kind'],
              'status': status, 'scientific_success_claimed': False,
              'protocol': protocol_ref, 'plan': plan, 'failure': failure,
              'shared_boundary_halt': state['halt'], 'accounting': dict(state, roles=dict(state['roles'])),
              'evaluations': evaluations, 'updates': updates, 'final_states': frozen,
              'coverage': {'expected_evaluation_slots': plan['evaluation_slots'],
                           'recorded_evaluation_slots': len(evaluations),
                           'execution_failures': failure_count,
                           'expected_updates': plan['training_updates'], 'completed_updates': len(updates)},
              'artifact_sha256': dict(evidence.hashes)}
    evidence.save('result.json', result)
    return result
