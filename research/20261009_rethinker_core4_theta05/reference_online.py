"""Fresh R1 finalclean extension: same online numerical path, independent public questions."""
import argparse,gc,gzip,hashlib,json,os,sys,time,traceback
from pathlib import Path
import frozen_engine as eng
E=Path(__file__).resolve().parent;R=E.parent.parent;O=R/'output/experiments'/E.name/'r1'
METHODS=('plain_soft_once','plain_residual_only')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def stem(s):return hashlib.sha256((s['dataset']+':'+str(s['id'])).encode()).hexdigest()[:20]
class OnlineProvider:
 def __init__(self,s,mask_path,zero=False):
  self.sample=s;self.mask_path=mask_path;self.zero=zero;self.event=None;self.vector=None;self.actual_vector=None;self.in_weak=False;self.audit={};self.raw_true=None;self.raw_weak=None;self.hard=None;self.signatures=[]
 def __call__(self,step,selected_token_id,raw_soft,hard,transition_step,entropy_proposed,locked_normal,controller_mode):
  if self.event is not None:return dict(event=self.event,vector=self.vector) if self.event<=step<self.event+4 else None
  allowed=transition_step>=0 and 0<step-transition_step<=8 and entropy_proposed and not locked_normal
  if not allowed:return None
  assert 2<=step<=9 and controller_mode==1
  self.event=step;frame=sys._getframe(1);live=frame.f_locals;prefix=live['all_generated'][0][live['prompt_lens'][0]:]+[selected_token_id];assert len(prefix)==step+1
  self.raw_true=raw_soft.detach().float().cpu();self.hard=hard.detach().float().cpu();main_state=eng.isolation.snapshot()
  from state_isolation import cache_signature,assert_disjoint
  before=cache_signature(frame);self.signatures=[before];weak=[]
  def observe(wframe):
   if int(wframe.f_locals['step']) in (0,step):
    sig=cache_signature(wframe);assert_disjoint([before,sig]);weak.append(sig)
  assert eng.weak_observer is None;eng.weak_observer=observe;self.in_weak=True;t0=time.perf_counter()
  try:
   mask=eng.base.sweep.create_mean_mask(self.sample['image'],self.mask_path)
   wi,wa=eng.capture(self.sample,mask,prefix,step,lam=.95)
   assert wa['refinement_steps']==wa['consumed_refinement_steps']==[] and wi['selected_token_id']==selected_token_id
   self.raw_weak=wi['soft_embedding'];assert eng.torch.equal(wi['hard_embedding'],self.hard)
  finally:
   eng.isolation.restore_checked(main_state);eng.weak_observer=None;self.in_weak=False
  after=cache_signature(frame);assert before==after,'main KV changed during independent weak replay';assert len(weak)==2;self.signatures.extend(weak);self.signatures.append(after)
  ui=self.raw_true-self.hard;uw=self.raw_weak-self.hard;rho=float(ui.norm());difference=ui-uw;dn=float(difference.norm());self.actual_vector=difference*(rho/dn) if dn>1e-12 else eng.torch.zeros_like(difference);self.vector=eng.torch.zeros_like(self.actual_vector) if self.zero else self.actual_vector
  assert eng.torch.isfinite(self.vector).all()
  from state_isolation import tensor_sha
  self.audit=dict(event=step,rho=rho,difference_norm=dn,weak_capture=wa,weak_replay_and_state_restore_seconds=time.perf_counter()-t0,main_cache_unchanged=True,independent_weak_cache=True,state_RNG_restored=True,raw_true_sha256=tensor_sha(self.raw_true),raw_weak_sha256=tensor_sha(self.raw_weak),actual_action_sha256=tensor_sha(self.actual_vector),cast_action_sha256=tensor_sha(self.actual_vector.to(eng.model.dtype)),zero=self.zero)
  return dict(event=step,vector=self.vector)
def run_arm(s,name,run):
 torch=eng.torch;eng.isolation.restore_checked(eng.initial);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();counter={'main':0,'weak':0};state={'provider':None};provider=None;geometry=None
 def count(module,args,kws):counter['weak' if state['provider'] is not None and state['provider'].in_weak else 'main']+=1
 hook=eng.model.register_forward_pre_hook(count,with_kwargs=True);t0=time.perf_counter()
 try:
  if name=='naive':
   prefix,dt,dm=eng.generate(s,cap=0,stop_first=True,lam=.95);event=len(prefix)-1 if dm['stopped'] else -1
   if event>=0:
    mask=eng.base.sweep.create_mean_mask(s['image'],run/'masks'/(stem(s)+'_naive.png'));ti,ta=eng.capture(s,s['image'],prefix,event,lam=.95);wi,wa=eng.capture(s,mask,prefix,event,lam=.95);assert ta['consumed_refinement_steps']==wa['consumed_refinement_steps']==[]
    ui=ti['soft_embedding'].float()-ti['hard_embedding'].float();uw=wi['soft_embedding'].float()-wi['hard_embedding'].float();rho=float(ui.norm());d=ui-uw;dn=float(d.norm());r=d*(rho/dn) if dn>1e-12 else torch.zeros_like(d)
    tokens,trace,meta=eng.generate(s,cap=0,prefix=prefix,event=event,action=r,lam=.95);geometry=dict(true=ti,weak=wi,action=r)
   else:tokens,trace,meta=prefix,dt,dm
  elif name in ('native_once','plain_soft_once'):
   tokens,hashes=eng.native_plain_once(s);trace=[];meta=dict(hashes=hashes,consumed=list(range(len(tokens)-1)),effective_consumed=[],initial_geometry=[],residual_geometry=[]);event=-1
  else:
   provider=OnlineProvider(s,run/'masks'/(stem(s)+'_'+name+'.png'),zero=name=='zero');state['provider']=provider;tokens,trace,meta=eng.generate(s,cap=0,trace_on=name!='traceoff',lam=.95,provider=provider);event=provider.event if provider.event is not None else -1
 finally:hook.remove()
 torch.cuda.synchronize();wall=time.perf_counter()-t0;peak=int(torch.cuda.max_memory_allocated());n=len(tokens)
 if name=='naive':assert counter['main']==n+(3*(event+1) if event>=0 else 0) and counter['weak']==0
 elif provider is not None:assert counter['main']==n and counter['weak']==(event+1 if event>=0 else 0)
 else:assert counter['main']==n and counter['weak']==0
 return dict(tokens=tokens,trace=trace,meta=meta,event=event,provider=provider,geometry=geometry,wall=wall,peak=peak,counter=counter)
def smoke(samples,run,mh,lane):
 torch=eng.torch;checks=[]
 for s in samples:
  assert sha(s['image'])==s['image_sha256'];values={}
  for name in ('naive','online','zero','traceoff','native_once'):
   v=run_arm(s,name,run);values[name]=v
  anchor=values['naive'];online=values['online'];zero=values['zero'];off=values['traceoff'];native=values['native_once'];event=anchor['event']
  assert anchor['tokens']==online['tokens']==off['tokens'] and anchor['meta']['hashes']==online['meta']['hashes']==off['meta']['hashes'];assert zero['tokens']==native['tokens'] and zero['meta']['hashes']==native['meta']['hashes'];assert online['event']==zero['event']==off['event']==event
  if event>=0:
   p=online['provider'];g=anchor['geometry'];assert torch.equal(p.raw_true,g['true']['soft_embedding']) and torch.equal(p.raw_weak,g['weak']['soft_embedding']) and torch.equal(p.hard,g['true']['hard_embedding']) and torch.equal(p.actual_vector,g['action']);assert torch.equal(p.actual_vector.to(eng.model.dtype),g['action'].to(eng.model.dtype));expected=[g['step'] for g in online['meta']['residual_geometry'] if g['step']<len(online['tokens'])-1 and not g['actual_noop']];assert online['meta']['effective_consumed']==anchor['meta']['effective_consumed']==expected
  else:assert online['tokens']==native['tokens'] and online['meta']['hashes']==native['meta']['hashes']
  small=dict(dataset=s['dataset'],id=str(s['id']),event=event,passed=True,full_tokens_actual_inputs_exact=True,zero_equals_native=True,traceoff_equals_naive=True,readouts_action_equal=True if event>=0 else None,branches={name:dict(tokens=v['tokens'],hashes=v['meta']['hashes'],seconds=v['wall'],forward_calls=v['counter'],peak_cuda_bytes=v['peak']) for name,v in values.items()},labels_read=False)
  save(run/(stem(s)+'_SMOKE_CASE.json'),small);checks.append(dict(dataset=s['dataset'],id=str(s['id']),event=event,passed=True));print('SMOKE_CASE_PASS',lane,s['dataset'],s['id'],event,flush=True);del values;gc.collect();torch.cuda.empty_cache()
 save(run/'SMOKE_LANE_PASS.json',dict(passed=True,lane=lane,n=len(samples),checks=checks,manifest_sha256=mh,labels_read=False))
def full(samples,run,mh,lane):
 torch=eng.torch;accepted=0
 for ix,s in enumerate(samples):
  d=O/'completed'/stem(s)
  if d.exists():
   ac=json.loads((d/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh and ac['results_sha256']==sha(d/'results.jsonl');accepted+=1;continue
  assert sha(s['image'])==s['image_sha256'];order_key=hashlib.sha256(('extension-full-order-v1:'+s['dataset']+':'+str(s['id'])).encode()).hexdigest();order=list(METHODS) if int(order_key,16)%2==0 else list(reversed(METHODS));q=run/stem(s);q.mkdir();values={}
  try:
   for name in order:values[name]=run_arm(s,name,run)
   native=values['plain_soft_once'];visual=values['plain_residual_only'];event=visual['event']
   if event<0:assert native['tokens']==visual['tokens'] and native['meta']['hashes']==visual['meta']['hashes']
   else:
    assert native['tokens'][:event+1]==visual['tokens'][:event+1] and native['meta']['hashes'][:event+1]==visual['meta']['hashes'][:event+1];assert visual['meta']['effective_consumed']==[g['step'] for g in visual['meta']['residual_geometry'] if g['step']<len(visual['tokens'])-1 and not g['actual_noop']]
   rows=[]
   for name in METHODS:
    v=values[name];tok=v['tokens'];me=v['meta'];p=v['provider'];rows.append(dict(model='r1',dataset=s['dataset'],id=str(s['id']),pair_id=s.get('pair_id'),method=name,text=eng.tokenizer.decode(tok,skip_special_tokens=True),generated_token_ids=tok,image_sha256=s['image_sha256'],runtime_error=None,residual_event=v['event'],effective_consumed_steps=me.get('effective_consumed',[]),refinement_steps=[],maxed1024=len(tok)==1024 and tok[-1]!=eng.tokenizer.eos_token_id,eos_terminated=tok[-1]==eng.tokenizer.eos_token_id,seconds=v['wall'],physical_seconds=v['wall'],physical_forward_calls=sum(v['counter'].values()),forward_calls_main=v['counter']['main'],forward_calls_weak=v['counter']['weak'],peak_cuda_allocated_bytes=v['peak'],order_key=order_key,actual_order=order,first_case_in_lane=ix==0,threshold=1.25,online_audit=p.audit if p is not None else None,timing_scope='Includes image prepare, generation, weak replay, StateIsolation and audit hooks; model load and smoke excluded; both fresh arms executed, no duplicate-work reuse'))
    save(q/(name+'_inputs.json'),dict(hashes=me['hashes'],consumed=me.get('consumed'),initial_geometry=me.get('initial_geometry'),residual_geometry=me.get('residual_geometry')))
    if name=='plain_residual_only':
     with gzip.open(str(q/(name+'_trace.json.gz')),'wt',encoding='utf-8') as f:json.dump(v['trace'],f,separators=(',',':'))
   (q/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));save(q/'AUDIT.json',dict(passed=True,labels_read=False,manifest_sha256=mh,fresh_arms=True,common_prefix_exact=True,native_noevent_equal=event<0,online_state_isolated=event>=0));save(q/'ACCEPTANCE.json',dict(passed=True,n=2,manifest_sha256=mh,results_sha256=sha(q/'results.jsonl'),audit_sha256=sha(q/'AUDIT.json'),labels_read=False));(O/'completed').mkdir(parents=True,exist_ok=True);os.rename(str(q),str(d));accepted+=1;print('FULL_ACCEPTED',lane,accepted,len(samples),s['dataset'],s['id'],event,flush=True);del values;gc.collect();torch.cuda.empty_cache()
  except Exception as exc:save(q/'ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),labels_read=False));raise
 save(run/'LANE_COMPLETE.json',dict(passed=True,lane=lane,n=accepted,expected=len(samples),manifest_sha256=mh,labels_read=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--lane',type=int,required=True);args=ap.parse_args();manifest=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
 for p,h in manifest['files'].items():assert sha(E/p)==h,p
 for p,h in manifest['checkpoint_files'].items():assert sha(Path(manifest['checkpoint']['path'])/p)==h,p
 if args.phase=='smoke':assert args.lane in (0,1);samples=[x['sample'] for x in json.loads((E/'smoke16.json').read_text()) if x['rank']==args.lane]
 else:
  assert args.lane in (0,1,2,3);permit=json.loads((O/'SMOKE_PASS.json').read_text());assert permit['passed'] and permit['manifest_sha256']==mh;samples=[s for u in read(E/'units.jsonl') if u['unit']%4==args.lane for s in u['samples']];assert len(samples)==1125
 run=O/args.phase/(os.environ.get('SLURM_JOB_ID','manual')+'_'+str(args.lane));run.mkdir(parents=True);(run/'masks').mkdir();t0=time.perf_counter();eng.init(manifest['checkpoint'],run,1.25);eng.torch.cuda.synchronize();save(run/'LOAD.json',dict(model_load_seconds=time.perf_counter()-t0,lane=args.lane,manifest_sha256=mh,labels_read=False))
 if args.phase=='smoke':smoke(samples,run,mh,args.lane)
 else:full(samples,run,mh,args.lane)
if __name__=='__main__':
 try:main()
 except Exception:print('FAILED',traceback.format_exc(),flush=True);raise
