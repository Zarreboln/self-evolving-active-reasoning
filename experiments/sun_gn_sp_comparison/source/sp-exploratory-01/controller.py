"""Fixed 33-episode unqualified SP pilot; injected transports only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'method-adapters'))
from method_adapters import dc_step, initial_memory, frozen_context
from public_feedback_sp import ace_public_feedback_sp_step

ARMS = ['ZeroShot', 'DC_Cumulative', 'ACE_Public_Feedback']
CONDITION = 'EXPLORATORY_UNQUALIFIED_SAME_FAMILY_REFEREE'

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def validate(scope, cases):
    fixed = {'condition':CONDITION, 'human_real_question_three_way_qualification':'UNMET',
             'production_qualification_claimed':False, 'model':'kimi-k2.6',
             'max_calls':1692, 'max_credit_usd':15, 'timeout_seconds':2700,
             'max_input_tokens':7168, 'max_output_tokens':1024,
             'memory_token_budget':1024, 'max_memory_estimator_calls':30,
             'arms':ARMS, 'dev_checkpoints':[0,1,3], 'final_memory_step':3}
    if any(scope.get(k) != v for k,v in fixed.items()):
        raise ValueError('Exploratory fixed scope mismatch')
    if set(cases) != {'train','dev','final'}:
        raise ValueError('Exactly train/dev/final case sets required')
    ids = []
    for split,n in [('train',3),('dev',2),('final',5)]:
        if len(cases[split]) != n:
            raise ValueError('Fixed split count mismatch')
        for row in cases[split]:
            if set(row) != {'case_id','surface','bottom','key_question'}:
                raise ValueError('Case schema mismatch')
            if any(not isinstance(row[k],str) or not row[k] for k in ('case_id','surface','bottom')):
                raise ValueError('Case identity/surface/bottom must be nonempty text')
            ids.append(row['case_id'])
    if len(set(ids)) != 10:
        raise ValueError('Cross-split case reuse')

def plan(cases):
    steps = []
    def dev(checkpoint):
        for arm in ARMS[1:]:
            for case in cases['dev']:
                steps.append({'kind':'episode','phase':'dev','arm':arm,'step':checkpoint,'case_id':case['case_id']})
    dev(0)
    for step,case in enumerate(cases['train'],1):
        for arm in ARMS[1:]:
            steps.append({'kind':'episode','phase':'train','arm':arm,'step':step,'case_id':case['case_id']})
            steps.append({'kind':'update','phase':'train','arm':arm,'step':step,'case_id':case['case_id']})
        if step in (1,3):dev(step)
    for index,case in enumerate(cases['final']):
        rotated = ARMS[index%3:]+ARMS[:index%3]
        for arm in rotated:
            steps.append({'kind':'episode','phase':'final','arm':arm,'step':0 if arm=='ZeroShot' else 3,'case_id':case['case_id']})
    return steps

def run_controller(*, cases, scope, chat, count_tokens, output,
                   episode_runner=None, projection=None,
                   evidence_kind='actual_injected_transport'):
    """No private target or outcome is passed into either memory updater.

    count_tokens is the declared provider single-user-message framing proxy;
    token_budget=1024 is upstream prompt guidance, never clipping or a hard cap.
    The caller enforces every request's real 7168-token input boundary.
    """
    validate(scope,cases)
    if episode_runner is None or projection is None:
        from episode import run_episode, public_projection
        episode_runner = episode_runner or run_episode
        projection = projection or public_projection
    output = Path(output);output.mkdir(exist_ok=False)
    schedule = plan(cases);save(output/'plan.json',schedule)
    case_map = {r['case_id']:r for group in cases.values() for r in group}
    memories = {'ZeroShot':'','DC_Cumulative':initial_memory('DC'),
                'ACE_Public_Feedback':initial_memory('ACE')}
    revisions = dict.fromkeys(ARMS,0)
    episodes, updates, completed = [], [], []
    attempts = 0; role_counts = Counter(); failure = None; latest_training = {}
    def record(event):
        with (output/'progress.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush()
    def snapshot(arm,step):
        memory=memories[arm]
        save(output/(arm+'-memory-'+str(step)+'.json'),
             {'arm':arm,'step':step,'memory':memory,'sha256':digest(memory),
              'provider_user_message_token_estimate_including_framing':count_tokens(memory),
              'count_is_exact_memory_tokens':False,'token_budget_is_prompt_guidance':True})
    def transport(item):
        def call(*,messages,max_tokens,seed,metadata):
            nonlocal attempts
            role=metadata.get('role');method=metadata.get('method')
            if item['kind']=='episode':
                if role=='referee':cap=64
                elif role=='actor' and metadata.get('actor_kind')=='question':cap=128
                elif role=='actor' and metadata.get('actor_kind')=='final':cap=512
                else:raise ValueError('Unexpected episode call role')
            else:
                allowed = ('curator',) if item['arm']=='DC_Cumulative' else ('reflector','curator')
                if role not in allowed:raise ValueError('Unexpected update call role')
                cap=1024
            if max_tokens!=cap or attempts>=1692:
                raise ValueError('Fixed call/output bound violated')
            requested = dict(metadata);effective = dict(metadata)
            requested_temp=metadata.get('temperature',.6)
            if requested_temp==0.0 and method=='ACE-Public-Feedback-SP' and item['kind']=='update':
                effective['sampling_adapter']='ACE_public_feedback_Kimi_instant_sampling_01'
                effective['upstream_requested_temperature']=0.0
            elif requested_temp!=.6:
                raise ValueError('Undeclared sampling adaptation')
            if metadata.get('top_p',.95)!=.95:
                raise ValueError('Undeclared top_p adaptation')
            effective.update(temperature=.6,top_p=.95,experiment_condition=CONDITION,
                             arm=item['arm'],phase=item['phase'],case_id=item['case_id'],step=item['step'])
            attempts+=1;role_counts[role]+=1
            record({'event':'sampling_admission','attempt':attempts,'requested_metadata':requested,
                    'effective_metadata':effective,'method_core_seed_placeholder':seed,
                    'provider_seed_sent':False,'max_tokens':max_tokens})
            response=chat(messages=messages,max_tokens=max_tokens,seed=None,metadata=effective)
            if not isinstance(response,dict) or not isinstance(response.get('text'),str):
                raise ValueError('Missing transport text')
            usage=response.get('usage',{})
            if any(type(usage.get(k)) is not int or not 0<=usage[k]<=lim
                   for k,lim in [('prompt_tokens',7168),('completion_tokens',max_tokens)]):
                raise ValueError('Invalid transport usage')
            return response
        return call
    try:
        for arm in ARMS:snapshot(arm,0)
        for index,item in enumerate(schedule):
            arm=item['arm'];before=digest(memories[arm]);record({'event':'begin','index':index,**item,'memory_sha256':before})
            if item['kind']=='episode':
                expected_step = item['step']-1 if item['phase']=='train' else item['step']
                if revisions[arm]!=expected_step:raise ValueError('Wrong frozen memory revision')
                row=case_map[item['case_id']]
                puzzle={k:row[k] for k in ('surface','bottom','key_question')}
                context='' if arm=='ZeroShot' else frozen_context('DC' if arm=='DC_Cumulative' else 'ACE',memories[arm])
                result=episode_runner(puzzle,chat=transport(item),seed_for=lambda *args:0,
                      scope=scope,phase=item['phase'],memory_context=context,evidence_kind=evidence_kind)
                save(output/f'episode-{len(episodes):02d}.json',{'schedule':item,'memory_sha256':before,'result':result})
                episodes.append({'schedule':item,'status':result.get('status'),'metrics':result.get('metrics'),
                                 'memory_sha256':before})
                if result.get('status')!='complete':raise RuntimeError('Episode failed; no retry or smaller replacement')
                if item['phase']=='train':latest_training[arm]=(item,projection(result))
                if digest(memories[arm])!=before:raise ValueError('Evaluation mutated memory')
            else:
                prior,public=latest_training.pop(arm)
                if prior['case_id']!=item['case_id'] or prior['step']!=item['step']:
                    raise ValueError('Update not paired with its completed training episode')
                if arm=='DC_Cumulative':
                    updated=dc_step(memories[arm],public,chat=transport(item),max_tokens=512,
                                    seed=0,phase='train',temperature=.6)
                else:
                    updated=ace_public_feedback_sp_step(memories[arm],public,
                        episode_completed=True,phase='train',chat=transport(item),count_tokens=count_tokens,
                        seed_for=lambda *args:0,reflector_max_tokens=1024,curator_max_tokens=1024,
                        token_budget=1024,step=item['step'],total_samples=3,use_json_mode=False)
                if not isinstance(updated.get('memory'),str):raise ValueError('Invalid updated memory')
                save(output/f'update-{len(updates):02d}.json',{'schedule':item,'update':updated,
                    'memory_measurement':'Provider token estimate for one user message, including framing; not exact standalone memory tokens.'})
                memories[arm]=updated['memory'];revisions[arm]=item['step'];updates.append(item)
                snapshot(arm,item['step'])
            completed.append(item);record({'event':'complete','index':index,**item})
        if len(episodes)!=33 or len(updates)!=6 or attempts!=1692:
            raise ValueError('Unexpected completed schedule/call count')
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
        record({'event':'halt','failure':failure,'transport_attempts':attempts})
    summary={'status':'EXPLORATORY_COMPLETE' if failure is None else 'EXPLORATORY_FAILED_OR_INCOMPLETE',
             'condition':CONDITION,'human_real_question_three_way_qualification':'UNMET',
             'production_qualification_claimed':False,'failure':failure,
             'completed_schedule_items':len(completed),'episodes':episodes,'updates':updates,
             'transport_attempts':attempts,'role_counts':dict(role_counts),'final_revisions':revisions,
             'final_memory_sha256':{a:digest(m) for a,m in memories.items()},
             'peak_selection':False,'fixed_final_step':3,'automatic_retries':0,
             'method_core_seed_placeholder':0,'provider_seed_sent':False,
             'sampling_adaptation':'Named ACE_public_feedback_Kimi_instant_sampling_01 changes upstream requested temperature0 to0.6/top_p0.95; preserved in admission records. DC temperature set explicitly0.6.',
             'memory_counting':'Provider single-user-message estimate including chat framing, cached; token_budget1024 is upstream guidance, no clipping.',
             'interpretation':'Unqualified same-family exploratory pilot only. F1 is descriptive official character overlap; no full referee qualification or causal method superiority.'}
    save(output/'summary.json',summary)
    return summary
