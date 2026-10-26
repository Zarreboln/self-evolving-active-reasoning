"""Pinned DC/ACE memory-core adaptations for an injected GN/SP episode engine.

Only training entry points can update memory. Evaluation calls frozen_context
and the external task engine; this module never scores held-out targets.
See README.md for the explicit generator and feedback adaptations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from upstream_loader import definitions, source_hashes, verified_source

PUBLIC_FIELDS = {'task', 'question', 'trajectory', 'predicted_answer'}


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def public_episode(episode):
    """Strict projection: pass no metrics, case identifiers, target or raw result."""
    if not isinstance(episode, dict) or set(episode) != PUBLIC_FIELDS:
        raise ValueError('Expected only task, question, trajectory, predicted_answer')
    if episode['task'] not in ('GN', 'SP'):
        raise ValueError('Unsupported task')
    for field in ('question', 'predicted_answer'):
        if not isinstance(episode[field], str):
            raise TypeError(field + ' must be text')
    if not isinstance(episode['trajectory'], list):
        raise TypeError('trajectory must be a list')
    for event in episode['trajectory']:
        if not isinstance(event, dict) or set(event) != {'role', 'content'}:
            raise ValueError('Each public event must contain only role and content')
        if event['role'] not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Invalid public event role')
        if not isinstance(event['content'], str):
            raise TypeError('Public content must be text')
    return json.loads(json.dumps(episode, ensure_ascii=False))


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')


def _train(phase, memory):
    if phase != 'train':
        raise ValueError('Memory updates are training-only')
    if not isinstance(memory, str):
        raise TypeError('memory must be text')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _invoke(chat, messages, max_tokens, seed, metadata, calls):
    _positive('max_tokens', max_tokens)
    if type(seed) is not int:
        raise TypeError('seed must be an integer')
    # Copies prevent an injected transport from changing the audit record.
    sent = json.loads(_json(messages))
    meta = json.loads(_json(metadata))
    response = chat(messages=json.loads(_json(sent)), max_tokens=max_tokens,
                    seed=seed, metadata=json.loads(_json(meta)))
    if not isinstance(response, dict) or not isinstance(response.get('text'), str):
        raise ValueError('chat must return a dict containing text')
    if 'finish_reason' not in response or not isinstance(response.get('usage'), dict):
        raise ValueError('chat response must include finish_reason and usage')
    record = {'messages': sent, 'max_tokens': max_tokens, 'seed': seed,
              'metadata': meta, 'response': json.loads(_json(response))}
    calls.append(record)
    return response['text'], record


def initial_memory(method):
    if method == 'DC':
        return ''
    if method == 'ACE':
        env = definitions('ace/ace/ace.py', ['_initialize_empty_playbook'],
                          class_name='ACE')
        return env['_initialize_empty_playbook'](None)
    raise ValueError('method must be DC or ACE')


def frozen_context(method, memory):
    """Memory-only task adaptation; native actor output format stays authoritative.

    This deliberately does not inject the upstream single-answer JSON/tag format
    into an interactive benchmark. It does not call a model or change memory.
    """
    if method not in ('DC', 'ACE') or not isinstance(memory, str):
        raise ValueError('Invalid frozen memory')
    if not memory:
        return ''
    label = 'cheatsheet' if method == 'DC' else 'playbook'
    return ('Use relevant strategies from the following ' + label +
            '. Preserve the task instructions and required response format.\n\n' + memory)


def dc_step(memory, episode, *, chat, max_tokens, seed, phase, temperature=0.0):
    """One exact upstream cumulative update after an external public episode.

    max_tokens is the upstream generator cap; its curator receives 2*max_tokens.
    The first upstream generate call replays the completed task episode. It is
    explicitly recorded as external generation, not counted as a new model call.
    Generated code execution is disabled. No label/score argument is accepted.
    """
    _train(phase, memory)
    _positive('max_tokens', max_tokens)
    episode = public_episode(episode)
    extractors = definitions('dc/dynamic_cheatsheet/utils/extractor.py',
                             ['extract_answer', 'extract_cheatsheet'])
    env = definitions('dc/dynamic_cheatsheet/language_model.py',
                      ['advanced_generate'], extractors, class_name='LanguageModel')
    generator_template = verified_source('dc/prompts/generator_prompt.txt')[1]
    curator_template = verified_source('dc/prompts/curator_prompt_for_dc_cumulative.txt')[1]
    # The entire public transcript is the generator's task-adapted output.
    output = _json(episode['trajectory']) + '\n<answer>' + episode['predicted_answer'] + '</answer>'
    calls, generation_requests = [], []

    def generate(**kwargs):
        generation_requests.append(kwargs)
        if len(generation_requests) == 1:
            return output
        if len(generation_requests) != 2:
            raise AssertionError('Unexpected DC call beyond one cumulative update')
        return _invoke(chat, kwargs['history'], kwargs['max_tokens'], seed,
                       {'method': 'DC', 'role': 'curator',
                        'temperature': kwargs['temperature'],
                        'allow_code_execution': False}, calls)[0]

    native = env['advanced_generate'](
        SimpleNamespace(generate=generate), approach_name='DynamicCheatsheet_Cumulative',
        input_txt=episode['question'], cheatsheet=memory,
        generator_template=generator_template, cheatsheet_template=curator_template,
        temperature=temperature, max_tokens=max_tokens, max_num_rounds=1,
        allow_code_execution=False, use_code_interpreter=False)
    updated = native['final_cheatsheet']
    return {'memory': updated, 'record': {
        'schema_version': 1, 'method': 'DC_Cumulative_task_adaptation',
        'phase': phase, 'before_sha256': digest(memory), 'after_sha256': digest(updated),
        'episode': episode, 'oracle_feedback_used': False,
        'generator_replayed_from_public_episode': True,
        'upstream_generation_requests': generation_requests,
        'calls': calls, 'upstream_result': native, 'source_hashes': source_hashes()}}


def ace_request_bound(actor_episode_bound, max_num_rounds, curator_due=True):
    """Full training step including initial and mandatory post-curator episodes."""
    _positive('actor_episode_bound', actor_episode_bound)
    _positive('max_num_rounds', max_num_rounds)
    if type(curator_due) is not bool:
        raise TypeError('curator_due must be bool')
    return {'actor_episodes': max_num_rounds + 2,
            'reflector_calls': max_num_rounds,
            'curator_calls': int(curator_due),
            'total_calls': (max_num_rounds + 2) * actor_episode_bound +
                           max_num_rounds + int(curator_due),
            'additional_after_initial': (max_num_rounds + 1) * actor_episode_bound +
                                       max_num_rounds + int(curator_due)}


def ace_step(memory, episode, *, training_target, answer_is_correct,
             regenerate, chat, count_tokens, seed_for, reflector_max_tokens,
             curator_max_tokens, max_num_rounds, curator_frequency,
             token_budget, step, total_samples, phase,
             no_ground_truth=False, use_json_mode=False):
    """Execute exact upstream ACE._train_single_sample with injected task episodes.

    Supervised training only: answer_is_correct(prediction, training_target)
    determines upstream retry/stop branches. no_ground_truth hides target text
    from reflection, NOT target-derived correctness. For SP, supply a separately
    justified training predicate; this adapter invents no semantic threshold.

    regenerate(*, memory_context, reflection, seed, metadata) returns the strict
    public episode schema; the caller owns native task execution and its logs.
    seed_for(role, call_id) returns an int. count_tokens is a real tokenizer
    callback. Neither an experiment size nor any seeds are selected here.
    """
    _train(phase, memory)
    episode = public_episode(episode)
    if not isinstance(training_target, str):
        raise TypeError('Explicit private training_target must be text')
    for name, value in [('reflector_max_tokens', reflector_max_tokens),
                        ('curator_max_tokens', curator_max_tokens),
                        ('max_num_rounds', max_num_rounds),
                        ('curator_frequency', curator_frequency),
                        ('token_budget', token_budget), ('step', step),
                        ('total_samples', total_samples)]:
        _positive(name, value)
    if step > total_samples:
        raise ValueError('step exceeds total_samples')
    if type(no_ground_truth) is not bool or type(use_json_mode) is not bool:
        raise TypeError('ACE flags must be bool')
    for callback in (answer_is_correct, regenerate, chat, count_tokens, seed_for):
        if not callable(callback):
            raise TypeError('ACE requires explicit callable dependencies')
    calls, generations, logs, predicates = [], [], [], []

    def timed_llm_call(_client, _provider, _model, prompt, *, role, call_id,
                       max_tokens, log_dir=None, use_json_mode=False):
        return _invoke(chat, [{'role': 'user', 'content': prompt}], max_tokens,
                       seed_for(role, call_id), {'method': 'ACE', 'role': role,
                       'call_id': call_id, 'temperature': 0.0,
                       'use_json_mode': use_json_mode}, calls)

    common = {'json': json, 're': re, 'Path': Path, 'timed_llm_call': timed_llm_call}
    slug = definitions('ace/utils.py', ['get_section_slug'])['get_section_slug']
    helpers = definitions('ace/playbook_utils.py', [
        'parse_playbook_line', 'get_next_global_id', 'format_playbook_line',
        'update_bullet_counts', 'apply_curator_operations', 'get_playbook_stats',
        'extract_json_from_text', 'extract_playbook_bullets'], dict(common, get_section_slug=slug))
    common.update({key: value for key, value in helpers.items() if not key.startswith('__')})
    prompt_names = {
        'generator': ['GENERATOR_PROMPT'],
        'reflector': ['REFLECTOR_PROMPT', 'REFLECTOR_PROMPT_NO_GT'],
        'curator': ['CURATOR_PROMPT', 'CURATOR_PROMPT_NO_GT', 'CURATOR_OPERATIONS_AGGREGATION_PROMPT']}
    for role, names in prompt_names.items():
        loaded = definitions('ace/ace/prompts/' + role + '.py', names)
        common.update({name: loaded[name] for name in names})
    # Upstream logging is retained as in-memory audit events, never filesystem IO.
    common['log_curator_operation_diff'] = lambda *args: logs.append(
        {'kind': 'curator_operation_diff', 'operation': args[1]})
    common['log_curator_failure'] = lambda *args: logs.append(
        {'kind': 'curator_failure', 'step': args[1], 'reason': args[2],
         'details': list(args[3:])})
    classes = {}
    for role in ('Generator', 'Reflector', 'Curator'):
        classes[role] = definitions('ace/ace/core/' + role.lower() + '.py',
                                    [role], common)[role]
    parser = classes['Generator'](None, 'injected', 'external-task-engine')

    def external_generate(**kwargs):
        replayed = not generations
        seed = seed_for('generator', kwargs['call_id'])
        if type(seed) is not int:
            raise TypeError('seed_for must return an integer')
        if replayed:
            result = episode
        else:
            result = public_episode(regenerate(
                memory_context=frozen_context('ACE', kwargs['playbook']),
                reflection=kwargs['reflection'], seed=seed,
                metadata={'method': 'ACE', 'role': 'generator',
                          'call_id': kwargs['call_id'], 'phase': 'train'}))
            if result['task'] != episode['task'] or result['question'] != episode['question']:
                raise ValueError('ACE regeneration must use the same training task')
        actor_text = '\n'.join(e['content'] for e in result['trajectory']
                               if e['role'] == 'assistant')
        bullet_ids = parser._extract_bullet_ids(actor_text, False)
        response = _json({'reasoning': result['trajectory'], 'bullet_ids': bullet_ids,
                          'final_answer': result['predicted_answer']})
        info = {'call_id': kwargs['call_id'], 'seed': seed,
                'replayed_initial_episode': replayed,
                'playbook_sha256': digest(kwargs['playbook']),
                'reflection': kwargs['reflection'], 'episode': result}
        generations.append(info)
        return response, bullet_ids, info

    def predicate(answer, target):
        result = answer_is_correct(answer, target)
        if type(result) is not bool:
            raise TypeError('Training correctness predicate must return bool')
        predicates.append({'prediction': answer, 'correct': result})
        return result

    def token_counter(text):
        result = count_tokens(text)
        if type(result) is not int or result < 0:
            raise ValueError('count_tokens must return a nonnegative integer')
        return result

    common.update(definitions('ace/utils.py', ['extract_answer'], {'json': json, 're': re}))
    common['count_tokens'] = token_counter
    common['log_bullet_usage'] = lambda *args, **kwargs: logs.append({
        'kind': 'bullet_usage', 'bullet_ids': args[4],
        'correct': kwargs['is_correct'], 'reflection': kwargs.get('reflection_content')})
    trainer = definitions('ace/ace/ace.py', ['_train_single_sample'], common,
                          class_name='ACE')['_train_single_sample']
    state = SimpleNamespace(
        playbook=memory, next_global_id=helpers['get_next_global_id'](memory),
        generator=SimpleNamespace(generate=external_generate),
        reflector=classes['Reflector'](None, 'injected', 'reflector', reflector_max_tokens),
        curator=classes['Curator'](None, 'injected', 'curator', curator_max_tokens),
        use_bulletpoint_analyzer=False, bulletpoint_analyzer=None)
    config = {'max_num_rounds': max_num_rounds, 'curator_frequency': curator_frequency,
              'token_budget': token_budget, 'use_json_mode': use_json_mode,
              'no_ground_truth': no_ground_truth}
    native = trainer(
        state, {'question': episode['question'], 'context': '', 'target': training_target},
        SimpleNamespace(answer_is_correct=predicate), 'train_step_' + str(step),
        1, step, None, '.', config, total_samples)
    return {'memory': state.playbook, 'record': {
        'schema_version': 1, 'method': 'ACE_offline_supervised_task_adaptation',
        'phase': phase, 'task': episode['task'], 'config': config,
        'before_sha256': digest(memory), 'after_sha256': digest(state.playbook),
        'oracle_feedback_used': True, 'target_text_visible': not no_ground_truth,
        'target_sha256': digest(training_target),
        'calls': calls, 'generator_episodes': generations, 'correctness_checks': predicates,
        'logs': logs, 'upstream_result': native, 'next_global_id': state.next_global_id,
        'source_hashes': source_hashes()}}
