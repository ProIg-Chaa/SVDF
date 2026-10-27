"""Public-only native faithful h2 for frozen checkpoints and public dataset shards."""
import argparse,hashlib,json,os
from pathlib import Path
from boundary import Boundary
from scale_contract import PUBLIC_KEYS,key,select_rows,validate_public
G=Path(__file__).resolve().parent
R=Path('/home/bingxing2/home/scx9ftv/gushuo');E=R/'experiments/20260914_h2_visulogic_completion'
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--model',choices=['r1','openvl','3B','32B'],required=True)
p.add_argument('--output',required=True);p.add_argument('--phase',choices=['smoke','full'],required=True)
p.add_argument('--profile',choices=['online','order_reverse','zero','default_off','fresh_reference'],default='online')
p.add_argument('--mode',choices=['h2','talr'],default='h2')
p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int)
p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID')
assert (a.mode=='h2' and (a.phase=='smoke' or a.profile=='online')) or (a.mode=='talr' and a.phase=='full' and a.profile=='default_off')
input_path=Path(a.inputs).resolve();assert any(str(input_path).startswith(str(E/n)+'/') for n in ('inputs','engineering_smoke')) and input_path.suffix=='.json'
folder=Path(a.output).resolve();allowed_output=R/'output/experiments/20260914_h2_visulogic_completion'
assert str(folder).startswith(str(allowed_output)+'/');folder.mkdir(parents=True,exist_ok=False);(folder/'vectors').mkdir()
raw=input_path.read_bytes();public=json.loads(raw);validate_public(public,require_pairs=a.phase!='smoke')
known=json.loads((G/'all_public.json').read_text());lookup={key(r):r for r in known}
assert len(lookup)==6419 and all(lookup.get(key(r))==r for r in public),'Input bytes must match frozen public rows exactly'
stop=len(public) if a.stop is None else a.stop;assert 0<=a.start<stop<=len(public)
entries=select_rows(public[a.start:stop],a.phase,a.shard_index,a.shard_count)
if a.phase=='full' and a.mode=='h2' and a.model in ('r1','openvl'):
    core=set(json.loads((G/'core_keys.json').read_text()));assert not core.intersection(key(r) for r in entries),'R1/OpenVL core already owned by original healthy lane'
guard=Boundary(str(R),str(G),str(folder),[x['image'] for x in entries],folder/'file_access.jsonl',public_files=[str(input_path)]);guard.install()
import time,torch,base,legacy_action,online_runtime,fresh_reference
from scale_model_loader import load_model
from datetime import datetime,timezone
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((G/'generation_manifest.json').read_text());assert all(sha(G/n)==h for n,h in freeze['files'].items())
deps=json.loads((G/'model_config_manifest.json').read_text());assert all(sha(Path(n))==h for n,h in deps.items())
spec=json.loads((G/'model_specs.json').read_text())[a.model];threshold=spec['threshold']
assert torch.cuda.is_available() and torch.cuda.device_count()==spec['gpus']
assert torch.get_num_threads()==torch.get_num_interop_threads()==128
config=dict(model=a.model,phase=a.phase,profile=a.profile,run_mode=a.mode,n=len(entries),input_path=str(input_path),input_sha256=hashlib.sha256(raw).hexdigest(),start=a.start,stop=stop,
    shard_index=a.shard_index,shard_count=a.shard_count,selected_public_ids=[dict(dataset=r['dataset'],id=r['id'],pair_id=r['pair_id'],image_sha256=r['image_sha256']) for r in entries],
    generation_manifest_sha256=sha(G/'generation_manifest.json'),controller=base.controller(threshold),seed=42,do_sample=False,logical_horizon=1024,
    geometry_capture='legacy_event_plus_1',probability='native_BF16_softmax_CPUFP32_thenFP16',vocabulary_size=spec['vocab_size'],hidden_size=spec['hidden_size'],primary_rule='faithful_h2_legacy',
    decision_step='actual_event+2_before_current_token',strengths=[0.,.5,1.,1.],duration=4,dtype='bfloat16',backend='sdpa',model_spec=spec,
    num_threads=128,num_interop_threads=128,slurm_job_id=os.environ['SLURM_JOB_ID'],no_old_event_prefix_probabilities_or_reference_outputs_loaded=True,
    full_HN_tails_allowed=False,fresh_BF_reference_is_smoke_only=True,peak_aggregate_semantics='sum_of_per_device_peaks_not_simultaneous_global_peak')
base.dump(folder/'config.json',config)
model,processor,tokenizer,math_ids=load_model(spec,folder)
calls=dict(generator=0,native1024=0,legacy_capture=0);original=base.generate_lead
def counted(*args,**kw):
    assert kw['lead_refinement_entropy_threshold']==threshold and kw['lead_refinement_window']==8 and kw['lead_refinement_soft_cap']==2
    assert not kw.get('trace_forced_answer_probe',False) and kw.get('trace_probe_gold_choice') is None
    calls['generator']+=1
    if kw.get('trace_soft_vector_collector') is not None:
        assert kw['max_new_tokens']==kw['trace_capture_soft_vector_step']+1;calls['legacy_capture']+=1
    else:assert kw['max_new_tokens']==1024;calls['native1024']+=1
    return original(*args,**kw)
base.generate_lead=counted;base.sweep.generate_lead=counted;legacy_action.generate_lead=counted
start=time.perf_counter()
for sample in entries:
    before=dict(calls)
    try:
        if a.profile=='fresh_reference':fresh_reference.run_reference(model,processor,tokenizer,math_ids,threshold,sample,folder)
        else:online_runtime.run_sample(model,processor,tokenizer,math_ids,threshold,sample,a.profile,folder)
    except BaseException as error:
        base.append(folder/'errors.jsonl',dict(dataset=sample['dataset'],id=sample['id'],runtime_error=type(error).__name__,message=str(error),replay_mismatch=False,incomplete_sample=True));raise
    base.append(folder/'generation_calls.jsonl',dict(dataset=sample['dataset'],id=sample['id'],calls={k:calls[k]-before[k] for k in calls}))
boundary=guard.snapshot();assert boundary['blocked_opens']==boundary['blocked_imports']==boundary['evaluator_calls']==0 and not boundary['evaluator_modules_loaded']
base.dump(folder/'GENERATION_COMPLETE.json',dict(actual_finish_utc=datetime.now(timezone.utc).isoformat(),model=a.model,phase=a.phase,profile=a.profile,samples=len(entries),
    calls=calls,wall_seconds=time.perf_counter()-start,boundary=boundary,generation_manifest_sha256=sha(G/'generation_manifest.json'),
    acceptance='pending independent file trace/source/state/fresh-reference engineering checks and separate evaluation',evaluator_calls=0))
