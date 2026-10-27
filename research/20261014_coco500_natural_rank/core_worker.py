"""Current soft-initialized SVDF F4, Rethinker core four datasets; pinned prior results, no gold or random."""
import argparse,functools,gc,hashlib,json,os,time,traceback
from pathlib import Path
import frozen_engine as eng
import reference_online as reference
from lean_generation import generate_lean_fixed
from common_decode import generate_baseline
E=Path(__file__).resolve().parent;R=E.parent.parent;O=R/'output/experiments'/E.name
CFG=dict(temperature=.6,top_p=.95,top_k=20,min_p=0.,do_sample=False,max_new_tokens=512)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def stamp():
 import datetime
 return datetime.datetime.now(datetime.timezone.utc).isoformat()
def key(s):return hashlib.sha256((s['dataset']+':'+str(s['id'])).encode()).hexdigest()[:20]
def run_arm(s,name,run):
 torch=eng.torch;eng.isolation.restore_checked(eng.initial);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 counter={'main':0,'weak':0};state={'provider':None};provider=None;audit=[]
 def count(module,args,kws):counter['weak' if state['provider'] is not None and state['provider'].in_weak else 'main']+=1
 hook=eng.model.register_forward_pre_hook(count,with_kwargs=True);t0=time.perf_counter()
 try:
  if name=='cot':
   inputs,_=eng.base.prepare(eng.processor,s,s['image'],eng.model.device)
   out=generate_baseline(eng.model,eng.tokenizer,inputs,config=CFG,seed=42);tokens=out['token_ids'];trace=out['trace'];meta=out['metadata'];event=-1
  elif name=='legacy_cot':
   inputs,length=eng.base.prepare(eng.processor,s,s['image'],eng.model.device)
   with torch.no_grad():out=eng.generate_cot(eng.model,eng.tokenizer,**inputs,**CFG)[0]
   tokens=[int(x) for x in out[length:].tolist()];trace=[];meta={};event=-1
  else:
   if name in ('plain_residual_only','zero'):
    provider=reference.OnlineProvider(s,run/'masks'/(key(s)+'_'+name+'.png'),zero=name=='zero');state['provider']=provider
   tokens,trace,meta=eng.generate(s,cap=0,lam=.95,provider=provider,generator=functools.partial(generate_lean_fixed,lean_compute_audit_sink=audit))
   event=provider.event if provider is not None and provider.event is not None else -1
 finally:hook.remove()
 torch.cuda.synchronize();seconds=time.perf_counter()-t0
 assert counter['main']==len(tokens)
 assert counter['weak']==(event+1 if event>=0 else 0)
 if audit:
  assert len(audit)==1 and audit[0]['soft_matmul_steps']==([] if eng.hard_initialization else [0])+([event] if event>=0 else [])
  assert all(i<=9 for i in audit[0]['entropy_steps']) and audit[0]['ordinary_phase_soft_matmul_count']==0
 return dict(tokens=tokens,trace=trace,meta=meta,event=event,provider=provider,counter=counter,seconds=seconds,peak=int(torch.cuda.max_memory_allocated()),compute=audit)

def smoke(model,run,mh):
 torch=eng.torch;checks=[]
 for s in json.loads((E/(model+'_smoke.json')).read_text()):
  assert sha(s['image'])==s['image_sha256']
  zero=run_arm(s,'zero',run);real=run_arm(s,'plain_residual_only',run)
  if eng.hard_initialization:
   expected=BASE_REFS[s['dataset']+':'+str(s['id'])]['generated_token_ids']
   assert zero['tokens']==expected,'zero hard feedback differs pinned native CoT'
   assert real['meta']['initial_geometry'][0]['exact_hard'] and zero['meta']['initial_geometry'][0]['exact_hard']
  else:
   expected,expected_hashes=eng.native_plain_once(s)
   assert zero['tokens']==expected and zero['meta']['hashes']==expected_hashes,'zero soft-initialized feedback differs native oneSoft'
  assert zero['event']==real['event']
  event=real['event']
  if event>=0:
   assert real['tokens'][:event+1]==zero['tokens'][:event+1]
   p=real['provider'];prefix=real['tokens'][:event+1]
   ti,ta=eng.capture(s,s['image'],prefix,event,lam=.95)
   wi,wa=eng.capture(s,str(p.mask_path),prefix,event,lam=.95)
   assert torch.equal(ti['soft_embedding'],p.raw_true) and torch.equal(wi['soft_embedding'],p.raw_weak)
   init='hard' if eng.hard_initialization else 'plain'
   assert ta['initialization']==wa['initialization']==init
  else:assert real['tokens']==zero['tokens']
  c=dict(passed=True,model=model,dataset=s['dataset'],id=str(s['id']),event=event,threshold=eng.threshold,initialization='hard' if eng.hard_initialization else 'onesoft',zero_full_tokens_inputs_equivalent=True,active_two_condition_readout_exact_reference=(event>=0),labels_read=False)
  checks.append(c);save(run/(key(s)+'_SMOKE.json'),c);print('SMOKE_PASS',model,s['dataset'],s['id'],event,flush=True)
  del real,zero;gc.collect();torch.cuda.empty_cache()
 assert len(checks)==3 and sum(v['event']>=0 for v in checks)>=2
 save(run/'SMOKE_PASS.json',dict(passed=True,model=model,n=3,active=sum(v['event']>=0 for v in checks),checks=checks,manifest_sha256=mh,labels_read=False))
def raw_row(s,model,name,v,mh):
 tok=v['tokens'];meta=v['meta'];p=v['provider'];n=len(tok);assert 0<n<=512
 if name=='plain_residual_only':
  if eng.hard_initialization:
   assert meta['initial_geometry'][0]['exact_hard']
   cot=BASE_REFS.get(s['dataset']+':'+str(s['id']))
   assert cot is not None
   expected=cot['generated_token_ids']
   if v['event']<0:assert tok==expected,'untriggered hard method differs CoT'
   else:assert tok[:v['event']+1]==expected[:v['event']+1],'hard method differs prefeedback CoT'
  assert all(not r.get('lead_refinement_active',False) for r in v['trace'])
 return dict(model=model,dataset=s['dataset'],id=str(s['id']),pair_id=s.get('pair_id'),method=('hard_residual_only' if eng.hard_initialization else 'plain_residual_only') if name=='plain_residual_only' else name,comparison_arm='svdf_adjusted' if name=='plain_residual_only' else 'cot',initialization='hard' if name=='cot' or eng.hard_initialization else 'onesoft',text=eng.tokenizer.decode(tok,skip_special_tokens=True),generated_token_ids=tok,image_sha256=s['image_sha256'],runtime_error=None,residual_event=v['event'],effective_consumed_steps=meta.get('effective_consumed',[]),refinement_steps=[],maxed1024=n==512 and tok[-1]!=eng.tokenizer.eos_token_id,eos_terminated=tok[-1]==eng.tokenizer.eos_token_id,seconds=v['seconds'],physical_seconds=v['seconds'],physical_forward_calls=sum(v['counter'].values()),forward_calls_main=v['counter']['main'],forward_calls_weak=v['counter']['weak'],peak_cuda_allocated_bytes=v['peak'],threshold=eng.threshold,online_audit=p.audit if p is not None else None,lean_compute_audit=v['compute'],trace_first10=v['trace'][:10],initial_geometry=meta.get('initial_geometry',[]),residual_geometry=meta.get('residual_geometry',[]),full_input_hashes_sha256=hashlib.sha256(json.dumps(meta.get('hashes',[])).encode()).hexdigest(),manifest_sha256=mh,labels_read=False,reused_from_prior=False,timing_scope='fresh resident synchronized arm, same processor; reused CoT costs retain historical scope, not matched fresh cost benchmark')
def full(model,run,mh,chunks):
 global BASE_REFS
 torch=eng.torch;queue=O/model/'chunks';claims=O/model/'claims';donebase=O/model/'completed';accepted=0
 for d in [queue,claims,donebase]:d.mkdir(parents=True,exist_ok=True)
 for chunk in chunks:
  cid='%04d'%chunk['chunk'];sealed=queue/(cid+'.json')
  if sealed.exists():continue
  claim=claims/cid
  try:claim.mkdir()
  except FileExistsError:continue
  save(claim/'OWNER.json',dict(utc=stamp(),pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),model=model,manifest_sha256=mh,chunk=chunk['chunk'],no_expiry=True))
  for s in chunk['samples']:
   k=s['dataset']+':'+str(s['id']);done=donebase/key(s)
   if done.exists():
    ac=json.loads((done/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh and ac['results_sha256']==sha(done/'results.jsonl');continue
   assert sha(s['image'])==s['image_sha256'];q=run/key(s);q.mkdir();rows=[]
   try:
    cot=BASE_REFS.get(k)
    if cot is None:
     assert model=='rethinker','CoT should already be available'
     cv=run_arm(s,'cot',run);cot=raw_row(s,model,'cot',cv,mh);del cv
     BASE_REFS[k]=cot;print('ARM_DONE',model,chunk['chunk'],s['dataset'],s['id'],'cot',len(cot['generated_token_ids']),-1,round(cot['seconds'],3),flush=True)
    else:
     cot=dict(cot);cot.update(comparison_arm='cot',reused_from_prior=True,new_experiment_manifest_sha256=mh);assert cot['image_sha256']==s['image_sha256']
    rows.append(cot)
    reused=METHOD_REFS.get(k)
    if reused is not None:
     assert model=='rethinker' and not eng.hard_initialization and reused['threshold']==eng.threshold
     row=dict(reused);row.update(comparison_arm='svdf_adjusted',initialization='onesoft',reused_from_prior=True,new_experiment_manifest_sha256=mh)
    else:
     mv=run_arm(s,'plain_residual_only',run);row=raw_row(s,model,'plain_residual_only',mv,mh);del mv
    rows.append(row);assert row['image_sha256']==s['image_sha256'] and len(row['generated_token_ids'])>0
    (q/'results.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows))
    save(q/'ACCEPTANCE.json',dict(passed=True,n=2,model=model,manifest_sha256=mh,results_sha256=sha(q/'results.jsonl'),labels_read=False))
    os.rename(str(q),str(done));accepted+=1
    print('ARM_DONE',model,chunk['chunk'],s['dataset'],s['id'],'svdf_adjusted',len(row['generated_token_ids']),row['residual_event'],round(row['seconds'],3),'reused',row.get('reused_from_prior',False),flush=True)
    del rows,cot,row;gc.collect();torch.cuda.empty_cache()
   except Exception as exc:
    save(q/'ERROR.json',dict(utc=stamp(),error=repr(exc),traceback=traceback.format_exc(),labels_read=False));raise
  save(sealed,dict(passed=True,utc=stamp(),chunk=chunk['chunk'],model=model,n=len(chunk['samples']),manifest_sha256=mh))
  print('CHUNK_COMPLETE',model,chunk['dataset'],chunk['chunk'],len(chunk['samples']),flush=True)
 save(run/'WORKER_COMPLETE.json',dict(passed=True,utc=stamp(),model=model,accepted_fresh=accepted,manifest_sha256=mh))
def main():
 global BASE_REFS,METHOD_REFS
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['qwen3b','openvl','rethinker'],required=True);a=ap.parse_args()
 m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json');approved=json.loads((E/'ROOT_APPROVAL.json').read_text());assert approved['passed'] and approved['manifest_sha256']==mh
 for f,h in m['files'].items():assert sha(E/f)==h,f
 chunks=json.loads((E/'half_chunks.json').read_text())
 available=[c for c in chunks if not (O/a.model/'chunks'/('%04d.json'%c['chunk'])).exists() and not (O/a.model/'claims'/('%04d'%c['chunk'])).exists()]
 if not available:print('NO_UNCLAIMED_WORK',a.model,flush=True);return
 BASE_REFS=json.loads((E/(a.model+'_cot_reuse.json')).read_text());METHOD_REFS=json.loads((E/'rethinker_method_reuse.json').read_text())
 spec=json.loads((E/'models.json').read_text())[a.model]
 for f,h in spec['checkpoint_files'].items():assert sha(Path(spec['path'])/f)==h
 run=O/a.model/'runtime'/(os.environ.get('SLURM_JOB_ID','manual')+'_'+os.environ.get('SLURM_ARRAY_TASK_ID','n')+'_'+str(os.getpid()));run.mkdir(parents=True);(run/'masks').mkdir();t0=time.perf_counter()
 eng.init(spec,run,spec['threshold']);eng.torch.cuda.synchronize();save(run/'LOAD.json',dict(utc=stamp(),model=a.model,model_load_seconds=time.perf_counter()-t0,manifest_sha256=mh,initialization=spec['initialization'],threshold=spec['threshold'],labels_read=False))
 smoke(a.model,run,mh);full(a.model,run,mh,chunks)
if __name__=='__main__':
 try:main()
 except Exception:print('FAILED',traceback.format_exc(),flush=True);raise
