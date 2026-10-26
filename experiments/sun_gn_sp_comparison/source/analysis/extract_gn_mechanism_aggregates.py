"""Recompute descriptive GN update aggregates from separately supplied local evidence.

This does not run an audit, update a memory, or invoke a model. No private data
locations are embedded. The caller must separately establish result provenance.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(controller):
    result_path = controller / 'result.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    output = {'original_result_sha256': digest(result_path), 'source_sha256': {},
              'dc_updates': [], 'ace_training_stages': []}
    previous = None
    for step in range(1, 7):
        updates = {}
        for arm in ('DC_Cumulative', 'ACE_Supervised'):
            relative = f'updates/{arm}-{step}.json'
            path = controller / relative
            binding = digest(path)
            if binding != result['artifact_sha256'][relative]:
                raise ValueError('Update bytes differ from original result binding')
            output['source_sha256'][relative] = binding
            updates[arm] = json.loads(path.read_text(encoding='utf-8'))
        dc = updates['DC_Cumulative']
        if len(dc['record']['calls']) != 1:
            raise ValueError('Expected one DC curator call')
        call = dc['record']['calls'][0]
        body = re.sub(r'^Version: [^\n]+', 'Version: X', dc['memory'])
        output['dc_updates'].append({'step': step,
            'finish_reason': call['response']['finish_reason'],
            'output_tokens': call['response']['usage']['completion_tokens'],
            'max_tokens': call['max_tokens'],
            'body_changed_after_removing_version_line': None if previous is None else body != previous})
        previous = body
        ace = updates['ACE_Supervised']['record']
        expected = [f'train_step_{step}_' + s for s in
                    ('gen_initial', 'post_reflect_round_0', 'post_curate')]
        if [x['call_id'] for x in ace['generator_episodes']] != expected:
            raise ValueError('Unexpected ACE training stage ordering')
        values = [x['correct'] for x in ace['correctness_checks']]
        if len(values) != 3 or any(type(v) is not bool for v in values):
            raise ValueError('Expected three assessed correctness checks')
        output['ace_training_stages'].append({'step': step,
            'initial_correct': values[0], 'post_reflection_correct': values[1],
            'post_curation_correct': values[2],
            'reflector_target_visible': ace['target_text_visible']})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = extract(args.controller)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
