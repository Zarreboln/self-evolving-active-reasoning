"""Portable stdlib arithmetic verification of anonymous GN/SP numeric rows.

No model, private labels, raw trajectories, vendor source or network is used.
This validates arithmetic and pairing, not the provenance of an experiment.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

METRICS=('official_success','harness_final_EM','hit_at_any','actor_final_EM')
SP_ARMS=('ZeroShot','DC_Cumulative','ACE_Public_Feedback')

def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):
    def unique(pairs):
        result={}
        for k,v in pairs:
            require(k not in result,'Duplicate JSON key');result[k]=v
        return result
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique)

def normalized(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))

def paired_difference(left,right,*,seed,draws=10000):
    common=sorted(set(left)&set(right))
    values=[int(left[k])-int(right[k]) for k in common if left[k] is not None and right[k] is not None]
    n=len(values)
    if not n:return dict(paired_n=0,difference=None,ci95=None,excluded=len(common))
    rng=random.Random(seed)
    bootstrap=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    def quantile(p):
        at=(draws-1)*p;low=int(at)
        return bootstrap[low]+(at-low)*(bootstrap[min(low+1,draws-1)]-bootstrap[low])
    return dict(paired_n=n,difference=sum(values)/n,ci95=[quantile(.025),quantile(.975)],
        excluded=len(common)-n,differences=Counter(values),
        method='posthoc descriptive paired-case percentile bootstrap; small N, no significance/success decision',
        draws=draws,seed=seed)

def validate_rows(rows,task,protocol):
    require(isinstance(rows,list),'Numeric rows must be a list')
    keys=[]
    for row in rows:
        require(set(row)=={'task','phase','arm','step','case','status','metrics'},'Unexpected public row fields')
        require(row['task']==task and row['phase'] in ('dev','final'),'Unexpected task/phase')
        require(re.fullmatch(task+r'-[0-9]{4}',row['case']) is not None,'Unsafe anonymous case alias')
        require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',row['arm']) is not None,'Unsafe method name')
        require(type(row['step']) is int,'Invalid checkpoint')
        keys.append((row['phase'],row['step'],row['arm'],row['case']))
        metrics=row['metrics']
        if task=='GN':
            require(row['status'] in ('complete','execution_failure'),'Unknown GN outcome status')
            require(metrics is None if row['status']=='execution_failure' else isinstance(metrics,dict),
                    'Failure must stay missing, never false')
            if metrics is not None:
                require(set(metrics)==set(METRICS),'Unexpected GN metrics')
                require(all(v is None or type(v) is bool for v in metrics.values()),'GN metrics must be bool/null')
        else:
            require(row['status']=='complete' and set(metrics)=={'official_f1_char','diagnostic_f1_word'},
                    'Only completed audited SP rows may be exported')
            require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in metrics.values()),
                    'Invalid or missing SP overlap metric')
    require(len(keys)==len(set(keys)),'Duplicate evaluated slot')
    splits=protocol['evaluation_cases'];flat=splits['dev']+splits['final']
    expected_sizes=(5,20) if task=='GN' else (2,5)
    require(tuple(len(splits[s]) for s in ('dev','final'))==expected_sizes and len(set(flat))==sum(expected_sizes),
            'Wrong or overlapping anonymous evaluation case sets')
    if task=='GN':
        arms=protocol['arms'];learning=[a['name'] for a in arms if a['kind']!='baseline']
        require(len(arms)==7 and len({a['name'] for a in arms})==7 and
                Counter(a['kind'] for a in arms)=={'baseline':5,'DC':1,'ACE':1},'Wrong GN arm matrix')
        expected={(phase,step,arm,case) for phase,steps,selected in
                  [('dev',(0,3,6),learning),('final',(6,),[a['name'] for a in arms])]
                  for step in steps for arm in selected for case in splits[phase]}
    else:
        require(protocol['qualification']=='UNMET' and protocol['condition']=='EXPLORATORY_UNQUALIFIED_SAME_FAMILY_REFEREE',
                'SP qualification limitation changed')
        expected={('dev',step,arm,case) for arm in SP_ARMS[1:] for step in (0,1,3) for case in splits['dev']}
        expected|={('final',0 if arm=='ZeroShot' else 3,arm,case) for arm in SP_ARMS for case in splits['final']}
    require(set(keys)==expected,'Planned/observed evaluation slots mismatch')

def summarize_gn(rows,protocol):
    validate_rows(rows,'GN',protocol)
    groups=defaultdict(list)
    for row in rows:groups[(row['phase'],row['step'],row['arm'])].append(row)
    tables=[]
    for key,items in sorted(groups.items()):
        table=dict(split=key[0],checkpoint=key[1],arm=key[2],scheduled=len(items),
                   execution_failures=sum(x['metrics'] is None for x in items),metrics={})
        for metric in METRICS:
            values=[x['metrics'][metric] for x in items if x['metrics'] is not None and x['metrics'][metric] is not None]
            table['metrics'][metric]=dict(successes=sum(values),defined_n=len(values),
                rate=sum(values)/len(values) if values else None,missing=len(items)-len(values))
        tables.append(table)
    arms=protocol['arms']
    baseline=next(a['name'] for a in arms if a['kind']=='baseline' and a['baseline_method']=='zero_shot')
    requested=[(('final',6,a['name']),('final',6,baseline)) for a in arms if a['name']!=baseline]
    requested += [(('dev',step,a['name']),('dev',0,a['name'])) for a in arms if a['kind']!='baseline' for step in (3,6)]
    contrasts=[]
    for index,(left,right) in enumerate(requested):
        for metric in METRICS:
            maps=[{x['case']:None if x['metrics'] is None else x['metrics'][metric] for x in groups[k]} for k in (left,right)]
            contrasts.append(dict(left=list(left),right=list(right),metric=metric,
                **paired_difference(*maps,seed=26102653+index)))
    return normalized({'tables':tables,'descriptive_paired_contrasts':contrasts})

def summarize_sp(rows,protocol):
    validate_rows(rows,'SP',protocol);curves=[]
    for phase,arms,steps,n in [('dev',SP_ARMS[1:],(0,1,3),2),('final',SP_ARMS,(None,),5)]:
        for arm in arms:
            for step in steps:
                items=[r for r in rows if r['phase']==phase and r['arm']==arm and (step is None or r['step']==step)]
                curves.append({'phase':phase,'arm':arm,'step':step if step is not None else (0 if arm==SP_ARMS[0] else 3),
                    'observed_n':len(items),'planned_n':n,'mean_character_F1':sum(r['metrics']['official_f1_char'] for r in items)/len(items) if items else None,
                    'mean_word_F1_diagnostic':sum(r['metrics']['diagnostic_f1_word'] for r in items)/len(items) if items else None,
                    'complete_denominator':len(items)==n})
    return {'descriptive_curves_and_means':curves}

def verify(folder):
    folder=Path(folder);protocol=read(folder/'protocol-public.json');data=read(folder/'anonymous-episodes.json')
    expected=read(folder/'results.json');actual={}
    require(protocol['schema_version']==data['schema_version']==expected['schema_version']==1,'Wrong schema')
    require(set(protocol['tasks'])==set(data['tasks'])==set(expected['tasks']),'Task sets differ')
    for task,p in protocol['tasks'].items():
        require(task in ('GN','SP'),'Unknown task')
        actual[task]=(summarize_gn if task=='GN' else summarize_sp)(data['tasks'][task],p)
        require(normalized(actual[task])==normalized(expected['tasks'][task]['numeric_results']),task+' arithmetic differs')
    manifest=read(folder/'numeric-export-manifest.json')
    for name,binding in manifest['files'].items():
        require(Path(name).name==name,'Unsafe manifest filename')
        raw=(folder/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==binding['sha256'] and len(raw)==binding['bytes'],'Public file hash differs')
    return {'status':'ARITHMETIC_AND_PUBLIC_FILE_HASHES_PASS','tasks':list(actual),
            'episode_slots':{task:len(rows) for task,rows in data['tasks'].items()},
            'private_provenance_reverified':False,'model_calls':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    print(json.dumps(verify(parser.parse_args().directory),sort_keys=True))
