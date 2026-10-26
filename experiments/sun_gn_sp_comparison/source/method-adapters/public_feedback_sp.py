"""ACE-Public-Feedback-SP: one public-only reflection and one curation.

This is a named task adaptation, not the supervised ACE training loop. The
caller completes an SP episode, projects only actor-visible messages, and then
calls this function once. It cannot receive labels, scores, or a regeneration
callback. Vendored Reflector, Curator, counters, IDs, and merge stay unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from method_adapters import _invoke, _json, _positive, _train, digest, public_episode
from upstream_loader import definitions, source_hashes


METHOD = 'ACE-Public-Feedback-SP'
PUBLIC_FEEDBACK = (
    'Completed SP public interaction. Final answer correctness: unassessed. '
    'Public referee replies concern individual questions only and do not certify '
    'the final answer. Analyze only the public transcript and current playbook; '
    'do not treat final correctness as observed. The full playbook is supplied '
    'as context; exposure does not establish which bullets were used.'
)
DEVIATIONS = (
    'External completed SP episode replaces the upstream Generator invocation.',
    'Exactly one no-ground-truth reflection and one no-ground-truth curation; '
    'no correctness predicate, correctness-triggered regeneration, or post-curation replay.',
    'The full current playbook is visible to the Reflector, not a verified used-bullet subset.',
    'Helpful/harmful tags are model judgments from public interaction, not measured final correctness.',
    'Upstream no-ground-truth prompt still uses error-oriented language; '
    'fixed feedback explicitly leaves final correctness unassessed.',
    'Upstream logging is captured in memory; token_budget remains upstream prompt guidance.'
)


def ace_public_feedback_sp_request_bound():
    """Additional requests after the separately completed task episode."""
    return {'actor_episodes': 0, 'reflector_calls': 1, 'curator_calls': 1,
            'total_calls': 2, 'automatic_retries': 0}


def ace_public_feedback_sp_step(
        memory, episode, *, episode_completed, phase, chat, count_tokens,
        seed_for, reflector_max_tokens, curator_max_tokens, token_budget,
        step, total_samples, use_json_mode=False):
    """Consume a completed, strict public SP projection without assessing it.

    episode_completed must be the caller's actual completion flag, not a
    correctness flag. Call with sp_public_projection(result) only after the
    native result has status='complete'. Never pass the full SP result. Exact
    schema validation cannot detect secrets already embedded in arbitrary text.

    A successful call makes two requests. Transport/parser failures propagate
    without retry; native Curator parse/operation failures retain its fallback.
    No actor call occurs here, including after a playbook update.
    """
    _train(phase, memory)
    episode = public_episode(episode)
    if episode['task'] != 'SP':
        raise ValueError('ACE-Public-Feedback-SP accepts only SP episodes')
    if type(episode_completed) is not bool or not episode_completed:
        raise ValueError('A completed public episode is required')
    for name, value in [('reflector_max_tokens', reflector_max_tokens),
                        ('curator_max_tokens', curator_max_tokens),
                        ('token_budget', token_budget), ('step', step),
                        ('total_samples', total_samples)]:
        _positive(name, value)
    if step > total_samples:
        raise ValueError('step exceeds total_samples')
    if type(use_json_mode) is not bool:
        raise TypeError('use_json_mode must be bool')
    for callback in (chat, count_tokens, seed_for):
        if not callable(callback):
            raise TypeError('Explicit callable dependencies are required')

    def token_count(text):
        count = count_tokens(text)
        if type(count) is not int or count < 0:
            raise ValueError('count_tokens must return a nonnegative integer')
        return count

    tokens_before = token_count(memory)
    calls, logs = [], []

    def timed_llm_call(_client, _provider, _model, prompt, *, role, call_id,
                       max_tokens, log_dir=None, use_json_mode=False):
        # A source change that introduces additional calls cannot consume them.
        expected_roles = ('reflector', 'curator')
        if len(calls) >= 2 or role != expected_roles[len(calls)]:
            raise ValueError('Unexpected public-feedback call sequence')
        return _invoke(chat, [{'role': 'user', 'content': prompt}], max_tokens,
                       seed_for(role, call_id),
                       {'method': METHOD, 'role': role, 'call_id': call_id,
                        'temperature': 0.0, 'use_json_mode': use_json_mode,
                        'final_answer_correctness': 'unassessed'}, calls)

    common = {'json': json, 're': re, 'Path': Path, 'timed_llm_call': timed_llm_call}
    slug = definitions('ace/utils.py', ['get_section_slug'])['get_section_slug']
    helpers = definitions('ace/playbook_utils.py', [
        'parse_playbook_line', 'get_next_global_id', 'format_playbook_line',
        'update_bullet_counts', 'apply_curator_operations', 'get_playbook_stats',
        'extract_json_from_text'], dict(common, get_section_slug=slug))
    common.update({key: value for key, value in helpers.items()
                   if not key.startswith('__')})
    for role, names in {
        'reflector': ['REFLECTOR_PROMPT', 'REFLECTOR_PROMPT_NO_GT'],
        'curator': ['CURATOR_PROMPT', 'CURATOR_PROMPT_NO_GT',
                    'CURATOR_OPERATIONS_AGGREGATION_PROMPT']
    }.items():
        loaded = definitions('ace/ace/prompts/' + role + '.py', names)
        common.update({name: loaded[name] for name in names})
    common['log_curator_operation_diff'] = lambda *args: logs.append(
        {'kind': 'curator_operation_diff', 'operation': args[1]})
    common['log_curator_failure'] = lambda *args: logs.append(
        {'kind': 'curator_failure', 'step': args[1], 'reason': args[2],
         'details': list(args[3:])})
    reflector_class = definitions('ace/ace/core/reflector.py', ['Reflector'], common)['Reflector']
    curator_class = definitions('ace/ace/core/curator.py', ['Curator'], common)['Curator']
    reflector = reflector_class(None, 'injected', 'reflector', reflector_max_tokens)
    curator = curator_class(None, 'injected', 'curator', curator_max_tokens)
    prefix = 'sp_public_step_' + str(step)
    reflection, tags, _ = reflector.reflect(
        question=episode['question'], reasoning_trace=_json(episode['trajectory']),
        predicted_answer=episode['predicted_answer'], ground_truth=None,
        environment_feedback=PUBLIC_FEEDBACK, bullets_used=memory,
        use_ground_truth=False, use_json_mode=use_json_mode,
        call_id=prefix + '_reflect', log_dir=None)
    counted = helpers['update_bullet_counts'](memory, tags) if tags else memory
    updated, next_id, operations, _ = curator.curate(
        current_playbook=counted, recent_reflection=reflection,
        question_context=_json({'surface': episode['question'],
                                'final_answer_correctness': 'unassessed'}),
        current_step=step, total_samples=total_samples, token_budget=token_budget,
        playbook_stats=helpers['get_playbook_stats'](counted),
        use_ground_truth=False, use_json_mode=use_json_mode,
        call_id=prefix + '_curate', log_dir='.',
        next_global_id=helpers['get_next_global_id'](memory))
    if len(calls) != 2:
        raise ValueError('Incomplete public-feedback call sequence')
    tokens_after = token_count(updated)
    return {'memory': updated, 'record': {
        'schema_version': 1, 'method': METHOD, 'phase': phase, 'task': 'SP',
        'episode_completed': True, 'public_episode': episode,
        'final_answer_correctness': 'unassessed',
        'oracle_feedback_used': False, 'target_text_visible': False,
        'correctness_checks': [], 'generator_episodes': [],
        'additional_actor_episodes': 0, 'automatic_retries': 0,
        'before_sha256': digest(memory), 'counter_updated_sha256': digest(counted),
        'after_sha256': digest(updated), 'reflection': reflection,
        'bullet_tags': tags, 'operations': operations, 'next_global_id': next_id,
        'calls': calls, 'logs': logs,
        'token_counts': {'before': tokens_before, 'after': tokens_after},
        'token_budget_exceeded': tokens_after > token_budget,
        'config': {'reflector_max_tokens': reflector_max_tokens,
                   'curator_max_tokens': curator_max_tokens,
                   'token_budget': token_budget, 'step': step,
                   'total_samples': total_samples, 'use_json_mode': use_json_mode,
                   'use_ground_truth': False},
        'deviations': list(DEVIATIONS), 'source_hashes': source_hashes()}}
