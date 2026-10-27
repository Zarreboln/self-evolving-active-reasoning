"""Verify public arithmetic and hashes only; no private replay or model run."""
import csv,hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'manifest.json').read_bytes())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    report=json.loads((ROOT/'analysis.json').read_bytes())
    provenance=json.loads((ROOT/'provenance.json').read_bytes())
    audit=json.loads((ROOT/'audit-result.json').read_bytes())
    execution=json.loads((ROOT/'audit-execution.json').read_bytes())
    rows=list(csv.DictReader((ROOT/'numeric-cases.csv').open(encoding='utf-8',newline='')))
    assert len(rows)==500
    for r in rows:
        for k in r:
            if k!='method':r[k]=int(r[k])
    for s in report['summaries']:
        rr=[r for r in rows if r['method']==s['method']]
        assert len(rr)==s['cases']==100
        assert sorted(r['case_index'] for r in rr)==list(range(100))
        assert all(r['official_success'] in (0,1) and r['harness_final_EM'] in (0,1) and r['hit_at_any'] in (0,1) for r in rr)
        for out,field in [('official_successes','official_success'),('harness_final_EM','harness_final_EM'),('hit_at_any','hit_at_any'),('completed_model_calls','model_calls'),('input_tokens','input_tokens'),('output_tokens','output_tokens'),('truncated_responses','truncated_responses')]:
            assert sum(r[field] for r in rr)==s[out], (s['method'],out)
        for out,field in [('selected_path_queries_mean','selected_path_queries'),('logged_guesses_mean','logged_guesses')]:
            assert math.isclose(statistics.mean(r[field] for r in rr),s[out],abs_tol=1e-12)
        k=s['official_successes'];n=100;z=1.959963984540054;den=1+z*z/n
        center=(k/n+z*z/(2*n))/den;half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/den
        assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(s['success_wilson95'],[max(0,center-half),min(1,center+half)]))
    baseline={r['case_index']:r for r in rows if r['method']=='zero_shot'}
    for s in report['paired_success_comparisons']:
        rr=[r for r in rows if r['method']==s['method']]
        wins=sum(r['official_success'] and not baseline[r['case_index']]['official_success'] for r in rr)
        losses=sum(not r['official_success'] and baseline[r['case_index']]['official_success'] for r in rr)
        assert wins==s['wins'] and losses==s['losses']
        n=wins+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        assert p==s['mcnemar_exact_two_sided_p_uncorrected']
    assert sum(r['model_calls'] for r in rows)==audit['verified_model_responses']==12805
    assert sum(r['input_tokens'] for r in rows)==provenance['actual_input_tokens']
    assert sum(r['output_tokens'] for r in rows)==provenance['actual_output_tokens']
    assert sum(r['truncated_responses'] for r in rows)==13
    assert execution['exit_code']==0 and execution['timed_out'] is False
    assert audit['status']=='PASS_OFFLINE_GN_FIVE_INTEGRITY'
    assert audit['result_sha256']==report['source_result_sha256']==provenance['controller_result_sha256']
    assert execution['result_sha256']==report['source_audit_sha256']==manifest['sha256']['audit-result.json']
    print('PASS: 500 numeric rows, five complete methods, hashes and arithmetic; not a model rerun.')

if __name__=='__main__':main()
