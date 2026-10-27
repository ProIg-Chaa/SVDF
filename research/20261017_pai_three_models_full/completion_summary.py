"""No-gold completion collector for the frozen PAI run. Does not score or submit jobs."""
import contextlib,datetime,gzip,hashlib,json,os,shutil,statistics,sys,time
from pathlib import Path
E=Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def key(s):return s['dataset']+':'+str(s['id'])
def stem(s):return hashlib.sha256(key(s).encode()).hexdigest()[:20]
def timing(values):
    return dict(n=len(values),sum_seconds=sum(values),mean_seconds=statistics.mean(values) if values else None,median_seconds=statistics.median(values) if values else None)
def collect():
    m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
    for fn,h in m['files'].items():assert sha(E/fn)==h,('frozen source changed',fn)
    for model,spec in m['models'].items():
        for fn,h in spec['checkpoint_files'].items():assert sha(Path(spec['path'])/fn)==h,(model,fn)
    approval=json.loads((E/'ROOT_APPROVAL.json').read_text());assert approval['approved'] and approval['manifest_sha256']==mh
    for lane in range(6):
        s=json.loads((E/'status'/('full_lane'+str(lane)+'.json')).read_text())
        assert s['state']=='COMPLETE' and s['completed']==s['expected']==3719 and s['manifest_sha256']==mh,('unfinished lane',lane)
    rows=lines(E/'public_all.jsonl');assert len(rows)==7438 and len({key(s) for s in rows})==7438
    final=E/'completion_exports'/mh[:16]
    if (final/'COMPLETE.json').exists():
        done=json.loads((final/'COMPLETE.json').read_text());assert done['manifest_sha256']==mh and done['total_answers']==22314
        for fn,h in done['export_files_sha256'].items():assert sha(final/fn)==h,fn
        print('ALREADY_COLLECTED',final,flush=True);return
    staging=final.with_name(final.name+'.staging.'+str(os.getpid())+'.'+str(time.time_ns()));staging.mkdir(parents=True)
    aliases={'r1':'model_A','openvl':'model_B','rethinker':'model_C'}
    aggregate={};raw_index=[]
    with contextlib.ExitStack() as stack:
        outputs={};blind={}
        for model in ('r1','openvl','rethinker'):
            pred=staging/'blind_predictions'/('predictions_'+aliases[model]+'.jsonl.gz');pred.parent.mkdir(parents=True,exist_ok=True)
            blind[model]=stack.enter_context(gzip.open(pred,'wt',encoding='utf-8'))
            aggregate[model]={}
            expected_names={stem(s)+'.json.gz' for s in rows}
            actual_names={p.name for p in (E/'results'/model).glob('*.json.gz')}
            assert actual_names==expected_names,('missing or extra keys',model,len(actual_names),len(expected_names))
            for dataset in m['datasets']:
                p=staging/'by_dataset'/model/(dataset+'.jsonl.gz');p.parent.mkdir(parents=True,exist_ok=True)
                outputs[model,dataset]=stack.enter_context(gzip.open(p,'wt',encoding='utf-8'))
                aggregate[model][dataset]=dict(n=0,reused=0,new=0,token_counts=[],truncated=0,reached_eos=0,new_prepare_plus_generation_diagnostic_seconds=[],historical_generation_only_diagnostic_seconds=[])
            for sample in rows:
                p=E/'results'/model/(stem(sample)+'.json.gz')
                with gzip.open(p,'rt',encoding='utf-8') as f:r=json.load(f)
                assert r['accepted'] and r['manifest_sha256']==mh and r['model']==model and r['sample']==sample and r['sample_key']==key(sample) and not r['labels_read']
                assert set(r['arms'])=={'pai_full'}
                a=r['arms']['pai_full'];ids=a['generated_token_ids'];n=len(ids)
                assert a['generated_tokens']==n and 0<n<=1024 and all(isinstance(t,int) for t in ids)
                assert a['forward_calls']==dict(main=n,weak=n,prefill_main=1,prefill_weak=1)
                meta=a['method_metadata'];att=a['attention_statistics']
                assert meta['alpha']==.2 and meta['gamma']==1.1 and meta['beta']==.1 and meta['maximum']==1024
                assert meta['cache_isolated'] and meta['branch_rope_isolated']
                assert att['positive_calls']==att['negative_calls']==att['modified_queries']==26*n
                reused=bool(r.get('reused',False));sec=a['wall_seconds'];assert isinstance(sec,(int,float)) and sec>=0
                ds=aggregate[model][sample['dataset']];ds['n']+=1;ds['reused' if reused else 'new']+=1;ds['token_counts'].append(n)
                ds['truncated']+=bool(a['truncated']);ds['reached_eos']+=bool(a['reached_eos'])
                scope='historical_generation_only_diagnostic_seconds' if reused else 'new_prepare_plus_generation_diagnostic_seconds'
                ds[scope].append(sec)
                record=dict(model=model,method='pai_full_Qwen25VL_matched_greedy',sample=sample,sample_key=key(sample),text=a['text'],generated_token_ids=ids,generated_tokens=n,
                    prompt_token_sha256=a['prompt_token_sha256'],reused=reused,reached_eos=a['reached_eos'],truncated=a['truncated'],wall_seconds=sec,timing_scope=a['timing_scope'],
                    forward_calls=a['forward_calls'],raw_result_sha256=sha(p),manifest_sha256=mh)
                outputs[model,sample['dataset']].write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                packet=dict(model_slot=aliases[model],dataset=sample['dataset'],id=str(sample['id']),sample_key=key(sample),question=sample['question'],
                    options=sample.get('options'),image=sample['image'],image_sha256=sample['image_sha256'],prediction=a['text'],generated_token_ids=ids,
                    truncated=a['truncated'],reached_eos=a['reached_eos'])
                assert not any(x in packet for x in ('gold','answer','correct','label','method'))
                blind[model].write(json.dumps(packet,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                raw_index.append(dict(model=model,sample_key=key(sample),path=str(p),sha256=record['raw_result_sha256'],reused=reused))
                if ds['n']%500==0:print('COLLECT',model,sample['dataset'],ds['n'],flush=True)
            for dataset,expected in m['datasets'].items():assert aggregate[model][dataset]['n']==expected,(model,dataset)
            assert sum(x['n'] for x in aggregate[model].values())==7438
    summary={}
    for model,groups in aggregate.items():
        summary[model]={}
        for dataset,stats in groups.items():
            summary[model][dataset]={k:v for k,v in stats.items() if not isinstance(v,list)}
            summary[model][dataset]['generated_tokens']=dict(sum=sum(stats['token_counts']),mean=statistics.mean(stats['token_counts']),median=statistics.median(stats['token_counts']))
            for scope in ['new_prepare_plus_generation_diagnostic_seconds','historical_generation_only_diagnostic_seconds']:
                summary[model][dataset][scope]=timing(stats[scope])
    assert sum(x['reused'] for groups in summary.values() for x in groups.values())==300
    save(staging/'DATASET_SUMMARY.json',dict(utc=utc(),manifest_sha256=mh,models=summary,accuracy_not_scored=True,
        timing_warning='All timings include canonical adapter diagnostic reductions and host synchronization. Historical300 exclude image preparation; never pool scopes or present intrinsic PAI overhead.'))
    save(staging/'RAW_RESULT_INDEX.json',raw_index)
    save(staging/'BLIND_KEY.json',dict(mapping=aliases,method='pai_full_Qwen25VL_matched_greedy',instruction='Keep separate from blind_predictions when assigning fact/text scoring; model aliases do not remove stylistic clues.'))
    save(staging/'SCORING_HANDOFF.json',dict(manifest_sha256=mh,answers=22314,questions_per_model=7438,labels_read=False,no_scoring_performed=True,
        packet='blind_predictions/*.jsonl.gz',raw_sources='RAW_RESULT_INDEX.json',eligibility='Bingo586 generated; scoring-side eligibility belongs to the scorer, not generator; POPE existing balanced500 subsets.',
        directive='r2/evaluator must use frozen existing scoring sidecars and parser; preserve predictions, record extraction failures, no generation-driven threshold search. No judge API called here.'))
    files={str(p.relative_to(staging)):sha(p) for p in staging.rglob('*') if p.is_file()}
    save(staging/'COMPLETE.json',dict(passed=True,utc=utc(),manifest_sha256=mh,total_answers=22314,models={x:7438 for x in aliases},reused=300,new=22014,
        no_gold_read=True,export_files_sha256=files))
    assert not final.exists();os.replace(staging,final)
    result=dict(passed=True,utc=utc(),manifest_sha256=mh,total_answers=22314,export_directory=str(final),complete_sha256=sha(final/'COMPLETE.json'))
    p=E/'COMPLETION_COLLECTION.json'
    if p.exists():
        old=json.loads(p.read_text());assert old['manifest_sha256']==mh and old['export_directory']==str(final)
    else:save(p,result)
    print('COLLECTION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':collect()
