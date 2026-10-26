"""Check public descriptive joins and arithmetic; no private scoring/model replay."""
import json
import math
from decimal import Decimal
from pathlib import Path


def verify(base):
    read = lambda name: json.loads((base / name).read_text(encoding='utf-8'))
    data = read('final-answer-lengths.json')
    public = read('numeric/anonymous-episodes.json')['tasks']['SP']
    final = {(r['case'], r['arm']): r['metrics'] for r in public if r['phase'] == 'final'}
    rows = data['rows']
    assert len(rows) == len(final) == 15
    assert len({(r['case'], r['arm']) for r in rows}) == 15
    assert {(r['case'], r['arm']) for r in rows} == set(final)
    values = ('official_f1_char', 'diagnostic_f1_word', 'final_answer_characters',
              'final_answer_whitespace_words', 'final_answer_completion_tokens')
    for row in rows:
        metrics = final[row['case'], row['arm']]
        assert all(row[k] == metrics[k] for k in values[:2])
        assert all(type(row[k]) is int and row[k] >= 0 for k in values[2:])
        assert row['finish_reason'] == 'stop'
    assert len(data['means_by_arm']) == 3
    for mean in data['means_by_arm']:
        selected = [r for r in rows if r['arm'] == mean['arm']]
        assert len(selected) == mean['n'] == 5
        for key in values:
            assert math.isclose(sum(r[key] for r in selected) / 5, mean[key], rel_tol=0, abs_tol=1e-12)
    usage = read('figures/sp-actual-usage.json')
    prices = data['generation_pricing']
    total_price = Decimal(0)
    assert {r['role'] for r in data['all_phase_role_usage']} == set(usage['role_counts'])
    for row in data['all_phase_role_usage']:
        role = row['role']
        assert row['all_phase_generation_calls'] == usage['role_counts'][role]
        assert row['prompt_tokens'] == usage['token_usage_by_role'][role]['prompt_tokens']
        assert row['completion_tokens'] == usage['token_usage_by_role'][role]['completion_tokens']
        cost = (Decimal(row['prompt_tokens']) * Decimal(prices['input_per_million_usd'])
                + Decimal(row['completion_tokens']) * Decimal(prices['output_per_million_usd'])) / 1_000_000
        assert cost == Decimal(row['uncached_generation_usage_price_estimate_usd'])
        total_price += cost
    assert total_price == Decimal(usage['uncached_generation_usage_price_estimate_usd'])
    assert sum(usage['role_counts'].values()) == usage['paired_generation_calls'] == 1692
    review = read('feedback-source-review.json')
    assert review['counts'] == {'source_supported': 23, 'source_contradicted': 1, 'source_unresolved': 6}
    assert sum(review['counts'].values()) == 30
    assert review['qualification'] == 'UNMET' and review['publication_ready'] is False
    return {'status':'SUPPLEMENT_ARITHMETIC_AND_JOINS_MATCH', 'final_rows':15,
            'reviewed_fixed_subset':30, 'generation_calls':1692,
            'uncached_generation_usage_price_estimate_usd':str(total_price),
            'private_provenance_reverified':False, 'human_qualification_established':False}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
