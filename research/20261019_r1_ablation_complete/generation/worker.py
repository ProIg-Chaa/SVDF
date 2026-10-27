"""Complete frozen R1 core-four ablations; generation is label-free."""
import argparse,gc,gzip,hashlib,json,os,time,traceback
from pathlib import Path
import frozen_engine as eng
from provider import OnlineProvider
E=Path(__file__).resolve().parent
R=Path('/home/bingxing2/home/scx9ftv/gushuo')
O=R/'output/experiments/20261019_r1_ablation_complete'
DIRECTIONS=('true_only','weak_only','neg_original_soft','reverse','random_104729')
CONFIG={n:dict(kind=n,start=2,dynamic=False) for n in DIRECTIONS}
CONFIG.update(start16=dict(kind='difference',start=16,dynamic=False),start32=dict(kind='difference',start=32,dynamic=False),
 dynamic_f4=dict(kind='difference',start=2,dynamic=True),reference_f4=dict(kind='difference',start=2,dynamic=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def key(s):return hashlib.sha256((s['dataset']+':'+str(s['id'])).encode()).hexdigest()[:20]
def run_arm(s,name,run,zero=False):
 eng.SOFT_START=True;torch=eng.torch;eng.isolation.restore_checked(eng.initial);gc.collect();torch.cuda.empty_cache()
 p=OnlineProvider(s,run/'masks'/(key(s)+'.png'),zero=zero,**CONFIG[name])
 calls={'main':0,'weak':0}
 def counter(module,args,kws):calls['weak' if p.in_weak else 'main']+=1
 hook=eng.model.register_forward_pre_hook(counter,with_kwargs=True)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 try:tokens,trace,meta=eng.generate(s,cap=0,lam=.95,provider=p,feedback_count=4)
 finally:hook.remove()
 torch.cuda.synchronize();wall=time.perf_counter()-t;event=-1 if p.event is None else p.event
 assert calls['main']==len(tokens) and calls['weak']==(event+1 if event>=0 else 0)
 assert not any(x.get('lead_refinement_active') for x in trace)
 if event>=0:
  assert p.audit['main_cache_unchanged'] and p.audit['independent_weak_cache']
  if not CONFIG[name]['dynamic']:assert CONFIG[name]['start']<=event<CONFIG[name]['start']+8
 return dict(tokens=tokens,trace=trace,meta=meta,event=event,provider=p,calls=calls,seconds=wall,peak=int(torch.cuda.max_memory_allocated()))
def get_reference(s):
 m=json.loads((E/'manifest.json').read_text())
 ref=Path(m['fixed_reference_core' if s['dataset']!='visulogic' else 'fixed_reference_visulogic'])/key(s)
 ac=json.loads((ref/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['results_sha256']==sha(ref/'results.jsonl')
 row=next(x for x in read(ref/'results.jsonl') if x['method']=='plain_residual_only')
 return row,ref
def check_reference(s,v):
 r,folder=get_reference(s)
 assert r['generated_token_ids']==v['tokens'] and r['residual_event']==v['event'],'old F4 tokens or event differs'
 if v['event']>=0:
  for k in ('raw_true_sha256','raw_weak_sha256','actual_action_sha256','cast_action_sha256'):
   assert r['online_audit'][k]==v['provider'].audit[k],k
  assert r['online_audit']['rho']==v['provider'].audit['rho']
 ip=folder/'plain_residual_only_inputs.json'
 if ip.exists():assert json.loads(ip.read_text())['hashes']==v['meta']['hashes']
 return dict(old_F4_tokens_exact=True,old_event_exact=True,old_action_readouts_exact=v['event']>=0)
def save_result(s,name,v,run,mh):
 q=run/name/key(s);q.mkdir(parents=True);p=v['provider'];tok=v['tokens'];meta=v['meta']
 text=eng.tokenizer.decode(tok,skip_special_tokens=True)
 row=dict(model='r1',dataset=s['dataset'],id=str(s['id']),pair_id=s.get('pair_id'),method=name,text=text,
  text_sha256=hashlib.sha256(text.encode()).hexdigest(),generated_token_ids=tok,image_sha256=s['image_sha256'],
  runtime_error=None,residual_event=v['event'],initial_soft=True,feedback_count=4,
  injection_prepared_steps=[x['step'] for x in meta['residual_geometry']],
  effective_consumed_steps=meta['effective_consumed'],actual_perstep_geometry=meta['residual_geometry'],
  initial_geometry=meta['initial_geometry'],refinement_steps=[],seconds=v['seconds'],physical_seconds=v['seconds'],
  peak_cuda_allocated_bytes=v['peak'],forward_calls_main=v['calls']['main'],forward_calls_weak=v['calls']['weak'],
  threshold=1.25,online_audit=p.audit,dynamic_decisions=p.decisions,
  eos_terminated=tok[-1]==eng.tokenizer.eos_token_id,maxed1024=len(tok)==1024 and tok[-1]!=eng.tokenizer.eos_token_id,
  inputs_sha256=hashlib.sha256(json.dumps(meta['hashes']).encode()).hexdigest())
 with (q/'results.jsonl').open('x') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 with gzip.open(q/'trace.json.gz','wt',encoding='utf-8') as f:json.dump(v['trace'],f,separators=(',',':'))
 save(q/'inputs.json',dict(hashes=meta['hashes']))
 save(q/'ACCEPTANCE.json',dict(passed=True,manifest_sha256=mh,results_sha256=sha(q/'results.jsonl'),labels_read=False))
 dst=O/'completed'/name/key(s);dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(q,target_is_directory=True)
 return row
def smoke(samples,run,mh,lane):
 checks=[]
 for s in samples:
  assert sha(s['image'])==s['image_sha256']
  ref=run_arm(s,'reference_f4',run);ck=check_reference(s,ref)
  z=run_arm(s,'reference_f4',run,zero=True);eng.isolation.restore_checked(eng.initial)
  nt,nh=eng.native_plain_once(s);assert z['tokens']==nt and z['meta']['hashes']==nh,'zero != native Once'
  ck.update(zero_equals_native_Once=True,sample=[s['dataset'],str(s['id'])],directions=[],same_hard_actual_delta_checks=[])
  # One active public smoke covers every signed/random direction at equal FP32 norm.
  if s['dataset']=='mmvp':
   assert ref['event']>=0
   for name in DIRECTIONS:
    v=run_arm(s,name,run);assert v['event']==ref['event']
    t=ref['event'];assert v['tokens'][:t+1]==ref['tokens'][:t+1] and v['meta']['hashes'][:t+1]==ref['meta']['hashes'][:t+1]
    a=v['provider'].audit;b=ref['provider'].audit
    assert a['raw_true_sha256']==b['raw_true_sha256'] and a['raw_weak_sha256']==b['raw_weak_sha256']
    assert abs(a['action_norm']-b['action_norm'])<=max(1e-7,b['rho']*2e-6)
    assert abs(a['cast_action_norm']-b['cast_action_norm'])<=max(1e-7,b['rho']*.01)
    hard=eng.weight[ref['tokens'][t]].detach()
    real_action=ref['provider'].actual_vector.to(hard.device,dtype=hard.dtype)
    control_action=v['provider'].actual_vector.to(hard.device,dtype=hard.dtype)
    real_delta=float(((hard+real_action).float()-hard.float()).norm())
    control_delta=float(((hard+control_action).float()-hard.float()).norm())
    relative=abs(control_delta-real_delta)/max(real_delta,1e-12)
    assert abs(control_delta-v['meta']['residual_geometry'][0]['actual_delta_norm'])<=max(1e-7,control_delta*2e-6)
    assert relative<=.02,('same-hard actual BF16 delta norm mismatch',name,relative)
    ck['same_hard_actual_delta_checks'].append(dict(direction=name,step=t,real_actual_delta_norm=real_delta,control_actual_delta_norm=control_delta,relative_error=relative,relative_tolerance=.02))
    ck['directions'].append(name)
  for name in ('start16','start32'):
   v=run_arm(s,name,run);assert v['event']<0 or CONFIG[name]['start']<=v['event']<CONFIG[name]['start']+8
   ck[name]=dict(event=v['event'],weak_calls=v['calls']['weak'],no_future_length_rule=True)
  v=run_arm(s,'dynamic_f4',run)
  # Original frozen dynamic numerical implementation is retained separately for two representative checks.
  if s['dataset']=='mmvp' or (s['dataset']=='visulogic' and str(s['id'])=='26'):
   import old_dynamic_worker as old
   old.eng=eng
   old.reference.bind()
   eng.isolation.restore_checked(eng.initial)
   oldv=old.run_arm(s,'online',run)
   assert oldv['tokens']==v['tokens'],'frozen dynamic full tokens differ'
   assert oldv['meta']['hashes']==v['meta']['hashes'],'frozen dynamic inputs differ'
   assert oldv['event']==v['event']
   ck['dynamic_frozen_tokens_and_inputs_exact']=True
  ck['dynamic_event']=v['event'];checks.append(ck)
  print('SMOKE_SAMPLE_PASS',lane,s['dataset'],s['id'],flush=True)
 save(O/('SMOKE_PASS_'+str(lane)+'.json'),dict(passed=True,manifest_sha256=mh,checks=checks,labels_read=False,lane=lane))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--lane',type=int,required=True);a=ap.parse_args();assert 0<=a.lane<3
 m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
 for n,h in m['files'].items():assert sha(E/n)==h,n
 for n,h in m['checkpoint_files'].items():assert sha(Path(m['checkpoint']['path'])/n)==h,n
 samples=read(E/'public.jsonl');bykey={(s['dataset'],str(s['id'])):s for s in samples}
 if a.phase=='smoke':samples=[bykey[tuple(v)] for i,v in enumerate(m['smoke_keys']) if i%3==a.lane]
 else:
  for lane in range(3):
   sm=json.loads((O/('SMOKE_PASS_'+str(lane)+'.json')).read_text());assert sm['passed'] and sm['manifest_sha256'] in (mh,m.get('engineering_smoke_manifest_sha256'))
  approval=json.loads((E/'FULL_APPROVAL.json').read_text());assert approval['approved'] and approval['manifest_sha256']==mh
  samples=[s for i,s in enumerate(samples) if i%3==a.lane]
 run=O/a.phase/(os.environ.get('SLURM_JOB_ID','manual')+'_'+str(a.lane)+'_'+str(time.time_ns()));run.mkdir(parents=True);(run/'masks').mkdir()
 t=time.perf_counter();eng.init(m['checkpoint'],run,1.25);eng.torch.cuda.synchronize()
 save(run/'LOAD.json',dict(seconds=time.perf_counter()-t,manifest_sha256=mh,labels_read=False))
 if a.phase=='smoke':smoke(samples,run,mh,a.lane);return
 dynamic_reuse={(x['dataset'],str(x['id'])):x for x in read(E/'dynamic_reuse.jsonl')}
 direction_reuse={(x['dataset'],str(x['id'])):x for x in read(E/'direction_noevent_reuse.jsonl')}
 accepted=0
 for s in samples:
  assert sha(s['image'])==s['image_sha256'];names=['start16','start32']
  if s['dataset']=='visulogic':
   if (s['dataset'],str(s['id'])) not in direction_reuse:names+=list(DIRECTIONS)
   if (s['dataset'],str(s['id'])) not in dynamic_reuse:names.append('dynamic_f4')
  order=sorted(names,key=lambda n:hashlib.sha256(('r1-ablation-completion-v1:'+s['dataset']+':'+str(s['id'])+':'+n).encode()).hexdigest())
  for name in order:
   done=O/'completed'/name/key(s)
   if done.exists():
    ac=json.loads((done/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh and ac['results_sha256']==sha(done/'results.jsonl');continue
   try:
    v=run_arm(s,name,run);save_result(s,name,v,run,mh);accepted+=1
    print('ACCEPTED',a.lane,accepted,s['dataset'],s['id'],name,v['event'],len(v['tokens']),flush=True)
    del v;gc.collect();eng.torch.cuda.empty_cache()
   except BaseException as exc:
    save(run/('ERROR_'+name+'_'+key(s)+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),sample=s,method=name));raise
 save(run/'LANE_COMPLETE.json',dict(passed=True,lane=a.lane,new_acceptances=accepted,manifest_sha256=mh,labels_read=False))
if __name__=='__main__':main()
