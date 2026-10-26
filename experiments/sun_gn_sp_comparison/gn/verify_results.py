"""Replay anonymous partial-run arithmetic; no models, private data or dependencies."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ARMS = ('ZeroShot', 'FewShot', 'FewShotInst', 'ToT_parser_repair01',
        'ProactiveCoT', 'DC_Cumulative', 'ACE_Supervised')
LEARNERS = ARMS[-2:]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive(data):
    require(set(data) == {'schema_version', 'status', 'coverage', 'source_sha256',
                         'rows', 'dc_updates', 'ace_training_stages'}, 'Unexpected schema')
    require(data['schema_version'] == 1 and data['status'] == 'PARTIAL_VERIFIED', 'Status')
    coverage = data['coverage']
    require(coverage == {'dev_planned': 30, 'dev_recorded': 30,
        'final_planned': 140, 'final_recorded': 97, 'final_exported_common_prefix': 91,
        'final_recorded_outside_prefix': 6, 'final_interrupted': 1,
        'final_not_started': 42, 'full_comparison_complete': False}, 'Coverage')
    require(set(data['source_sha256']) == {'frozen_protocol', 'original_result',
        'allocation_receipt', 'partial_integrity_report', 'partial_integrity_code'}, 'Bindings')
    require(all(re.fullmatch(r'[0-9a-f]{64}', v) for v in data['source_sha256'].values()), 'Digest')
    expected = {('dev', c, a, f'D{i:02d}') for c in (0, 3, 6)
                for a in LEARNERS for i in range(1, 6)}
    expected |= {('final_common_prefix', 6, a, f'F{i:02d}')
                 for a in ARMS for i in range(1, 14)}
    seen, groups = set(), defaultdict(list)
    for row in data['rows']:
        require(set(row) == {'split', 'checkpoint', 'arm', 'case',
            'native_success', 'actor_final_em', 'hit_at_any'}, 'Row fields')
        key = (row['split'], row['checkpoint'], row['arm'], row['case'])
        require(key in expected and key not in seen, 'Missing, duplicate or nonanonymous row')
        require(type(row['native_success']) is bool and type(row['hit_at_any']) is bool,
                'Native binary metrics must be assessed')
        require(row['actor_final_em'] is None or type(row['actor_final_em']) is bool,
                'Actor metric must be boolean or null')
        require(row['arm'] in ('ProactiveCoT', 'ToT_parser_repair01') or
                type(row['actor_final_em']) is bool, 'Required explicit actor readout')
        seen.add(key)
        groups[key[:3]].append(row)
    require(seen == expected, 'Incomplete exported matrix')
    tables = []
    for (split, checkpoint, arm), rows in sorted(groups.items()):
        assessed = [r['actor_final_em'] for r in rows if r['actor_final_em'] is not None]
        tables.append({'split': split, 'checkpoint': checkpoint, 'arm': arm,
            'n': len(rows), 'native_successes': sum(r['native_success'] for r in rows),
            'actor_final_successes': sum(assessed), 'actor_final_assessed': len(assessed),
            'actor_final_unassessed': len(rows) - len(assessed),
            'hit_at_any': sum(r['hit_at_any'] for r in rows)})
    dc, ace = data['dc_updates'], data['ace_training_stages']
    require(len(dc) == len(ace) == 6, 'Six training steps required')
    for i, row in enumerate(dc, 1):
        require(set(row) == {'step', 'finish_reason', 'output_tokens', 'max_tokens',
            'body_changed_after_removing_version_line'}, 'DC fields')
        require(row['step'] == i and row['finish_reason'] in ('stop', 'length') and
            type(row['output_tokens']) is int and 0 <= row['output_tokens'] <= row['max_tokens'] == 1024,
            'DC call facts')
        changed = row['body_changed_after_removing_version_line']
        require(changed is None if i == 1 else type(changed) is bool, 'DC comparison')
    for i, row in enumerate(ace, 1):
        require(set(row) == {'step', 'initial_correct', 'post_reflection_correct',
            'post_curation_correct', 'reflector_target_visible'}, 'ACE fields')
        require(row['step'] == i and all(type(row[k]) is bool for k in row if k != 'step'),
                'ACE stage facts')
        require(row['reflector_target_visible'] is True, 'Supervised training label')
    return {'status': 'PARTIAL_VERIFIED', 'full_comparison_complete': False,
        'arithmetic_scope': 'Anonymous rows and descriptive update aggregates only; not independent run provenance.',
        'exported_rows': len(seen), 'dev_rows': 30, 'common_final_rows': 91,
        'missing_final_slots': coverage['final_interrupted'] + coverage['final_not_started'],
        'tables': tables,
        'dc': {'curator_outputs': 6, 'length_limited': sum(x['finish_reason'] == 'length' for x in dc),
            'post_initial_transitions': 5,
            'body_changes_after_first': sum(x['body_changed_after_removing_version_line'] is True for x in dc[1:])},
        'ace': {'training_cases': 6, 'initial_correct': sum(x['initial_correct'] for x in ace),
            'post_reflection_correct': sum(x['post_reflection_correct'] for x in ace),
            'post_curation_correct': sum(x['post_curation_correct'] for x in ace),
            'reflector_target_visible': True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('anonymous-numerics.json'))
    parser.add_argument('--results', type=Path, default=Path('results.json'))
    args = parser.parse_args()
    actual = derive(json.loads(args.data.read_text(encoding='utf-8')))
    require(actual == json.loads(args.results.read_text(encoding='utf-8')), 'Result arithmetic differs')
    print(json.dumps({'status': 'ARITHMETIC_MATCH', 'rows': actual['exported_rows'],
        'full_comparison_complete': False, 'missing_final_slots': actual['missing_final_slots'],
        'provenance_independently_reverified_by_this_script': False}, indent=2))


if __name__ == '__main__':
    main()
