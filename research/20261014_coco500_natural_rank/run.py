"""COCO natural generation/replay. No oracle is read by generation."""
import argparse,contextlib,datetime,gc,gzip,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent.parent
sys.path.insert(0,str(E/'source'))
import frozen_engine as eng
PROMPT='Please help me describe the image in detail.'
ARMS=('cot','once','svdf','pai')
CFG=dict(temperature=.6,top_p=.95,top_k=20,min_p=0.,do_sample=False,max_new_tokens=512)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with(gzip.open(p,'rt')if str(p).endswith('.gz')else Path(p).open())as f:return json.load(f)
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as raw:
  if str(p).endswith('.gz'):
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as f:f.write(json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode())
  else:raw.write(json.dumps(obj,ensure_ascii=False,indent=2).encode())
def caption_prepare(processor,sample,image_path,device):
 from lead.inference import prepare_inputs
 assert sample['question']==PROMPT and sample.get('options',[])==[]
 messages=[{'role':'user','content':[{'type':'image','image':str(image_path)},{'type':'text','text':PROMPT}]}]
 inputs=prepare_inputs(processor,messages,device)
 assert 'pixel_values'in inputs and inputs['pixel_values'].numel()>0 and inputs['image_grid_thw'].numel()>0
 return inputs,int(inputs['input_ids'].shape[1])
def points(n,ids,eos):
 result=[];edges=[0,n//3,2*n//3,n]
 for stage,lo,hi in zip(('early','middle','late'),edges[:-1],edges[1:]):
  count=min(8,hi-lo)
  for i in range(count):
   step=lo if count==1 else lo+((hi-1-lo)*i)//(count-1)
   result.append(dict(step=step,stage=stage,relative_position=step/max(n-1,1),eos_prediction=ids[step]==eos))
 return result
def run_arm(sample,arm,run,candidates=None,point_plan=None,dual=False,prepare_fn=None):
 import torch,core_worker as utility,reference_online as reference,pai_adapter
 from route_tape import tensor_hash
 from rank_observer import RankObserver
 eng.isolation.restore_checked(eng.initial)
 inputs,_=(prepare_fn or caption_prepare)(eng.processor,sample,sample['image'],eng.model.device)
 prompt_ids=inputs['input_ids'].detach().cpu().tolist();prompt_hash=hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
 original_provider=reference.OnlineProvider;original_attention=pai_adapter.PAIAttention;original_select=pai_adapter.select_token
 holder={'provider':None,'pai':None};hashes={'main':[],'weak':[],'pai_negative':[]};processed=[];select_calls=[0]
 class TrackedProvider(original_provider):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);holder['provider']=self
 class TrackedPAI(original_attention):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);holder['pai']=self
 def branch():
  if holder['provider']is not None and holder['provider'].in_weak:return'weak'
  if holder['pai']is not None and holder['pai'].negative:return'pai_negative'
  return'main'
 def observe_input(module,args,kw):
  key='inputs_embeds'if kw.get('inputs_embeds')is not None else'input_ids';value=kw[key]
  hashes[branch()].append(dict(kind=key,sha256=tensor_hash(value),shape=list(value.shape),dtype=str(value.dtype)))
 obs=RankObserver(eng.model,candidates,point_plan,positive_branch=lambda:branch()=='main',dual_check=dual)if candidates is not None else None
 def observe_select(scores,config):
  result=original_select(scores,config);call=select_calls[0];select_calls[0]+=1
  if obs is not None and call%2==1 and obs.step in obs.points:
   raw=scores[0].detach().double();lp=torch.log_softmax(raw,dim=-1);cs=[]
   for q in candidates:
    tid=q['token_ids'][0];v=float(raw[tid]);ok=math.isfinite(v)
    cs.append(dict(role=q['role'],text=q['text'],first_token_id=tid,rank=1+int((raw>raw[tid]).sum()),ties=int((raw==raw[tid]).sum()),score=v if ok else None,allowed=ok,normalized_log_probability=float(lp[tid])if ok else None,probability=float(lp[tid].exp())if ok else 0.,actual_filtered_probability=float(result[2][0,tid])))
   processed.append(dict(step=obs.step,stage=obs.points[obs.step]['stage'],score_definition='gamma*(lp_image-lp_noimage)+lp_noimage; beta plausibility mask; before sampling filters',candidates=cs))
  return result
 reference.OnlineProvider=TrackedProvider;pai_adapter.PAIAttention=TrackedPAI;pai_adapter.select_token=observe_select
 hook=eng.model.register_forward_pre_hook(observe_input,with_kwargs=True);torch.cuda.synchronize();start=time.perf_counter()
 try:
  with torch.no_grad(),(obs if obs is not None else contextlib.nullcontext()):
   if arm=='pai':
    p=pai_adapter.generate(eng.model,eng.tokenizer,inputs,alpha=.2,gamma=1.1,beta=.1,maximum=512,config=CFG)
    tokens=p['token_ids'];meta=dict(metadata=p['metadata'],attention=p['attention']);trace=p['metadata']['trace_first16'];event=-1;provider=None;compute=[]
   else:
    name={'cot':'cot','once':'plain_soft_once','svdf':'plain_residual_only','zero':'zero','legacy_cot':'legacy_cot'}[arm]
    v=utility.run_arm(sample,name,run);tokens=v['tokens'];meta=v['meta'];trace=v['trace'];event=v['event'];provider=v['provider'];compute=v['compute']
 finally:
  hook.remove();reference.OnlineProvider=original_provider;pai_adapter.PAIAttention=original_attention;pai_adapter.select_token=original_select
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 assert 0<len(tokens)<=512 and len(hashes['main'])==len(tokens)
 if arm=='pai':assert len(hashes['pai_negative'])==len(tokens)and not hashes['weak']
 else:assert not hashes['pai_negative']and len(hashes['weak'])==(event+1 if event>=0 else 0)
 record=dict(case_id=sample['case_id'],arm=arm,generated_token_ids=tokens,text=eng.tokenizer.decode(tokens,skip_special_tokens=True),text_with_special=eng.tokenizer.decode(tokens,skip_special_tokens=False),n_tokens=len(tokens),prompt=PROMPT,prompt_input_ids_sha256=prompt_hash,actual_input_hashes=hashes,event=event,effective_consumed_steps=meta.get('effective_consumed',[]),method_metadata=meta,trace_first16=trace[:16],lean_compute_audit=compute,image_sha256=sample['image_sha256'],seconds=elapsed,eos_terminated=tokens[-1]==eng.tokenizer.eos_token_id,truncated=len(tokens)==512 and tokens[-1]!=eng.tokenizer.eos_token_id,online_audit=provider.audit if provider is not None else None,oracle_read=False)
 capture=None
 if provider is not None and event>=0:
  capture=dict(raw_true=provider.raw_true,raw_weak=provider.raw_weak,hard=provider.hard,actual_vector=provider.actual_vector,vector=provider.vector,event=event)
 if obs is not None:
  record.update(points=point_plan,candidates=candidates,records=obs.records,processed_readouts=processed,engineering=obs.validate(len(tokens)),own_natural_trajectory=True,shared_prefix=False)
  if dual:record['_raw_dumps']=obs.raw_dumps
 return record,capture

def equal_generation(a,b):
 for k in ('generated_token_ids','prompt_input_ids_sha256','actual_input_hashes','event','effective_consumed_steps'):assert a[k]==b[k],k
 if a['arm']in('once','svdf'):assert a['method_metadata']['hashes']==b['method_metadata']['hashes']
def verify_manifest(m):
 for f,h in m['files'].items():assert sha(E/f)==h,f
 for f,h in m['checkpoint_files'].items():assert sha(Path(m['checkpoint']['path'])/f)==h,f
 assert sha(E/'input/public.jsonl')==m['public_sha256']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['generation_smoke','generation_full','replay_smoke','replay_full'],required=True);ap.add_argument('--lane',type=int,default=0);a=ap.parse_args()
 m=load(E/'GENERATION_MANIFEST.json');mh=sha(E/'GENERATION_MANIFEST.json');gate=load(E/'ROOT_GENERATION_APPROVAL.json');assert gate['passed']and gate['manifest_sha256']==mh;verify_manifest(m)
 samples=[json.loads(l)for l in(E/'input/public.jsonl').read_text().splitlines()];assert len(samples)==500
 generation=a.phase.startswith('generation');smoke=a.phase.endswith('smoke');out=E/'output'/('generation'if generation else'replay');run=out/'runs'/(a.phase+'_'+os.environ.get('SLURM_JOB_ID','manual')+'_'+str(a.lane)+'_'+str(time.time_ns()));run.mkdir(parents=True);(run/'masks').mkdir()
 if smoke:assert a.lane==0;samples=[s for s in samples if s['case_id']in m['smoke_case_ids']];assert len(samples)==2
 else:
  assert 0<=a.lane<4;passed=load(out/'SMOKE_PASS.json');assert passed['passed']and passed['generation_manifest_sha256']==mh;samples=[s for i,s in enumerate(samples)if i%4==a.lane]
 oracle=None;rh=None
 if not generation:
  rm=load(E/'REPLAY_MANIFEST.json');rh=sha(E/'REPLAY_MANIFEST.json');rp=load(E/'ROOT_REPLAY_APPROVAL.json');assert rp['passed']and rp['manifest_sha256']==rh
  assert rm['generation_manifest_sha256']==mh and sha(E/rm['oracle_path'])==rm['oracle_sha256'];oracle={x['case_id']:x for x in map(json.loads,(E/rm['oracle_path']).read_text().splitlines())}
 eng.init(m['checkpoint'],run,1.25);historical_prepare=eng.base.prepare;eng.base.prepare=caption_prepare;checks=[];active_smoke=[]
 import torch
 for sample in samples:
  assert sha(sample['image'])==sample['image_sha256'];cid=sample['case_id']
  for arm in ARMS:
   final=out/'completed'/cid/(arm+'.json.gz')
   if final.exists():
    old=load(final);assert old['generation_manifest_sha256']==mh
    if not generation:assert old['replay_manifest_sha256']==rh and(old.get('replay_verified')is True or old.get('replay_status')=='no_eligible_candidate')
    continue
   ref=None;cs=None;plan=None
   if not generation:
    refpath=E/'output/generation/completed'/cid/(arm+'.json.gz');ref=load(refpath);assert ref['generation_manifest_sha256']==mh
    # Oracle schema frozen separately; never derive labels from readout ranks.
    cs=oracle[cid]['candidates'];assert all(x['token_ids']for x in cs)
    if not cs:
     record=dict(ref,replay_manifest_sha256=rh,replay_verified=None,replay_status='no_eligible_candidate',source_generation=str(refpath),source_generation_sha256=sha(refpath),own_natural_trajectory=True,shared_prefix=False,oracle_read=True,records=[],points=[],candidates=[],processed_readouts=[],engineering={'replay_skipped':True,'reason':'no_eligible_candidate'})
     save(final,record);checks.append(dict(case_id=cid,arm=arm,sha256=sha(final),status='no_eligible_candidate'));print('COCO_SKIPPED_RANK',cid,arm,'no_eligible_candidate',flush=True);continue
    plan=points(ref['n_tokens'],ref['generated_token_ids'],eng.tokenizer.eos_token_id)
   rec,cap=run_arm(sample,arm,run,cs,plan,dual=smoke and not generation);rec['generation_manifest_sha256']=mh
   if generation and smoke:
    repeat,cap2=run_arm(sample,arm,run);equal_generation(rec,repeat)
    if cap is not None:
     assert cap2 is not None
     for k in ('raw_true','raw_weak','hard','actual_vector','vector'):assert torch.equal(cap[k],cap2[k]),k
    rec['engineering_repeat_exact']=True
    if arm=='svdf' and cap is not None:
     event=cap['event'];prefix=rec['generated_token_ids'][:event+1]
     mask=eng.base.sweep.create_mean_mask(sample['image'],run/'masks'/(cid+'_check_mean.png'))
     ti,ta=eng.capture(sample,sample['image'],prefix,event,lam=.95)
     wi,wa=eng.capture(sample,mask,prefix,event,lam=.95)
     assert torch.equal(ti['soft_embedding'],cap['raw_true'])and torch.equal(wi['soft_embedding'],cap['raw_weak'])
     assert torch.equal(ti['hard_embedding'],cap['hard'])and ta['refinement_steps']==wa['refinement_steps']==[]
     rec['engineering_capture_reference_exact']=True
   if not generation:
    equal_generation(rec,ref);rec['engineering'].update(full_historical_tokens_exact=True,actual_inputs_exact=True,no_forced_tokens=True);rec.update(replay_verified=True,replay_manifest_sha256=rh,source_generation=str(refpath),source_generation_sha256=sha(refpath),oracle_read=True)
    if cap is not None:
     oldcap=torch.load(ref['capture_path'],map_location='cpu',weights_only=False);assert sha(ref['capture_path'])==ref['capture_sha256']
     for k in ('raw_true','raw_weak','hard','actual_vector','vector'):assert torch.equal(cap[k],oldcap[k]),k
     rec['capture_recomputed_exact']=True
    if '_raw_dumps'in rec:
     rp=run/(cid+'_'+arm+'_raw.pt.gz')
     with gzip.open(rp,'wb')as f:torch.save(rec.pop('_raw_dumps'),f)
     rec.update(raw_smoke_path=str(rp),raw_smoke_sha256=sha(rp))
   elif cap is not None:
    cp=out/'captures'/(cid+'_'+arm+'.pt');cp.parent.mkdir(parents=True,exist_ok=True)
    with cp.open('xb')as f:torch.save(cap,f)
    rec.update(capture_path=str(cp),capture_sha256=sha(cp))
   save(final,rec);checks.append(dict(case_id=cid,arm=arm,sha256=sha(final),seconds=rec['seconds'],event=rec['event']));print('COCO_ACCEPTED',a.phase,cid,arm,rec['n_tokens'],rec['event'],round(rec['seconds'],3),flush=True)
   del rec,cap;gc.collect();torch.cuda.empty_cache()
  if smoke and generation:
   once=load(out/'completed'/cid/'once.json.gz');svdf=load(out/'completed'/cid/'svdf.json.gz');zero,_=run_arm(sample,'zero',run)
   assert zero['generated_token_ids']==once['generated_token_ids']and zero['actual_input_hashes']['main']==once['actual_input_hashes']['main']
   assert zero['event']==svdf['event'];ev=svdf['event']
   if ev>=0:
    assert svdf['generated_token_ids'][:ev+1]==once['generated_token_ids'][:ev+1]
    if len(svdf['effective_consumed_steps'])==4:active_smoke.append(cid)
   else:assert svdf['generated_token_ids']==once['generated_token_ids']and svdf['actual_input_hashes']['main']==once['actual_input_hashes']['main']
   cot=load(out/'completed'/cid/'cot.json.gz');legacy,_=run_arm(sample,'legacy_cot',run);assert cot['generated_token_ids']==legacy['generated_token_ids']
 if smoke and generation and not active_smoke:
  fixture=load(E/'engineering_fixture.json');s=fixture['sample'];assert sha(s['image'])==s['image_sha256'];eng.base.prepare=historical_prepare
  try:
   once,_=run_arm(s,'once',run,prepare_fn=historical_prepare)
   zero,_=run_arm(s,'zero',run,prepare_fn=historical_prepare)
   real,cap=run_arm(s,'svdf',run,prepare_fn=historical_prepare)
   assert zero['generated_token_ids']==once['generated_token_ids']and zero['actual_input_hashes']['main']==once['actual_input_hashes']['main']
   assert real['event']==zero['event']==fixture['expected_event']and real['effective_consumed_steps']==fixture['expected_effective_consumed_steps']
   ev=real['event'];prefix=real['generated_token_ids'][:ev+1];assert real['generated_token_ids'][:ev+1]==once['generated_token_ids'][:ev+1]
   mask=eng.base.sweep.create_mean_mask(s['image'],run/'masks'/'engineering_M002_mean.png')
   ti,ta=eng.capture(s,s['image'],prefix,ev,lam=.95);wi,wa=eng.capture(s,mask,prefix,ev,lam=.95)
   assert torch.equal(ti['soft_embedding'],cap['raw_true'])and torch.equal(wi['soft_embedding'],cap['raw_weak'])
   save(run/'ACTIVE_ENGINEERING_FIXTURE_PASS.json',dict(passed=True,case_id=s['case_id'],engineering_only=True,not_in_coco500=True,event=ev,effective_consumed_steps=real['effective_consumed_steps'],zero_equals_once=True,capture_reference_exact=True,generation_manifest_sha256=mh))
   active_smoke.append(s['case_id'])
  finally:eng.base.prepare=caption_prepare
 if smoke and generation:assert active_smoke
 save(run/'COMPLETE.json',dict(passed=True,phase=a.phase,lane=a.lane,n_cases=len(samples),new_arms=len(checks),generation_manifest_sha256=mh,replay_manifest_sha256=rh,checks=checks))
 if smoke:save(out/'SMOKE_PASS.json',dict(passed=True,generation_manifest_sha256=mh,replay_manifest_sha256=rh,case_ids=m['smoke_case_ids'],active_generation_smoke=active_smoke if generation else None,checks=checks))
if __name__=='__main__':
 try:main()
 except Exception:print('COCO_FAILED',traceback.format_exc(),flush=True);raise
