"""Observe original generators; require exact historical natural-token reproduction."""
import os,sys,json,gzip,hashlib,time,math,argparse,gc,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent.parent;O=E/'output';sys.path.insert(0,str(E/'source'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 if str(p).endswith('.gz'):
  with gzip.open(p,'rt') as f:return json.load(f)
 return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.gz':
  with p.open('xb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(json.dumps(x,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
 else:
  with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def points(n,tokens,eos):
 assert 1<=n<=1024 and len(tokens)==n
 out=[]
 for stage,lo,hi in [('early',0,n//3),('middle',n//3,2*n//3),('late',2*n//3,n)]:
  if hi<=lo:continue
  count=min(8,hi-lo);ids=[lo] if count==1 else [lo+(hi-1-lo)*i//(count-1) for i in range(count)]
  assert len(set(ids))==count
  for step in ids:out.append(dict(step=step,stage=stage,relative_position=step/n,eos_prediction=tokens[step]==eos))
 assert len({x['step'] for x in out})==len(out)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--lane',type=int,required=True);args=ap.parse_args();m=load(E/'manifest.json');mh=sha(E/'manifest.json')
 for n,h in m['code_sha256'].items():assert sha(E/n)==h,n
 for n,h in m['source_files'].items():assert sha(E/'source'/n)==h,n
 for n,h in m['checkpoint_files'].items():assert sha(Path(m['checkpoint']['path'])/n)==h,n
 assert sha(E/'input/facts_and_trajectories.jsonl')==m['input_sha256'] and sha(E/'input/INPUT_FREEZE.json')==m['input_freeze_sha256']
 allcases=[json.loads(l)for l in(E/'input/facts_and_trajectories.jsonl').read_text().splitlines()];assert len(allcases)==208
 assert sum(x['fact']['rank_primary_eligible'] for x in allcases)==201
 if args.phase=='smoke':
  assert args.lane==0;cases=[x for x in allcases if x['case_id'] in m['smoke_case_ids']];assert len(cases)==2
 else:
  assert 0<=args.lane<4
  for lane in range(1):
   passed=load(O/('SMOKE_PASS_'+str(lane)+'.json'));assert passed['passed'] and passed['manifest_sha256']==mh
  cases=[x for i,x in enumerate(allcases)if i%4==args.lane]
 import torch,base,pai_adapter
 from scale_model_loader import load_model
 from state_isolation import StateIsolation
 from common_decode import generate_baseline
 from engine import make_engine
 from rank_observer import RankObserver
 torch.set_num_threads(4);run=O/'runs'/(args.phase+'_'+os.environ.get('SLURM_JOB_ID','manual')+'_'+str(args.lane)+'_'+str(time.time_ns()));run.mkdir(parents=True)
 model,processor,tok,mi=load_model(m['checkpoint'],run);iso=StateIsolation(model);initial=iso.snapshot();engine=make_engine(model,processor,tok,mi,threshold=1.25);cfg=m['generation'];checks=[]
 for case in cases:
  sample=case['sample'];assets=case['replay_assets'];candidates=[dict(role='correct',text=case['fact']['true_text'],token_ids=case['fact']['true_token_ids']),dict(role='wrong',text=case['fact']['false_text'],token_ids=case['fact']['false_token_ids'])]
  assert sha(sample['image'])==sample['image_sha256']
  for arm in m['arms']:
   ref=case['trajectories'][arm];assert sha(ref['source'])==ref['source_sha256']
   final=O/'completed'/case['case_id']/(arm+'.json.gz')
   if final.exists():
    old=load(final);assert old['manifest_sha256']==mh and old['replay_verified'] and old['n_tokens']==len(ref['token_ids']);continue
   iso.restore_checked(initial);inp,_=base.prepare(processor,sample,sample['image'],model.device);ids=inp['input_ids'].detach().cpu();jh=hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest();bh=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
   assert jh==case['trajectories']['once']['metadata']['input_ids_sha256']
   assert jh==case['trajectories']['svdf']['metadata']['input_ids_sha256']
   assert bh==case['trajectories']['pai']['prompt_input_ids_sha256']
   pts=points(len(ref['token_ids']),ref['token_ids'],tok.eos_token_id);processed=[];holder=[None];original_attention=pai_adapter.PAIAttention;original_select=pai_adapter.select_token;select_calls=[0]
   class TrackedPAIAttention(original_attention):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);holder[0]=self
   obs=RankObserver(model,candidates,pts,positive_branch=lambda:holder[0] is None or not holder[0].negative,dual_check=args.phase=='smoke')
   def observe_select(scores,config):
    result=original_select(scores,config);call=select_calls[0];select_calls[0]+=1
    if call%2==1 and obs.step in obs.points:
     raw=scores[0].detach().double();lp=torch.log_softmax(raw,dim=-1);cs=[]
     for candidate in candidates:
      tid=candidate['token_ids'][0];val=float(raw[tid]);ok=math.isfinite(val)
      cs.append(dict(role=candidate['role'],first_token_id=tid,rank=int((raw>raw[tid]).sum())+1,ties=int((raw==raw[tid]).sum()),score=val if ok else None,allowed=ok,normalized_log_probability=float(lp[tid]) if ok else None,probability=float(lp[tid].exp()) if ok else 0.,actual_filtered_probability=float(result[2][0,tid])))
     processed.append(dict(step=obs.step,stage=obs.points[obs.step]['stage'],score_definition='gamma*(lp_image-lp_noimage)+lp_noimage; beta plausibility mask; before sampling filters',candidates=cs))
    return result
   torch.cuda.synchronize();begin=time.monotonic()
   try:
    with torch.inference_mode(),obs:
     if arm=='cot':
      result=generate_baseline(model,tok,inp,config=cfg);tokens=result['token_ids'];extra=result['metadata']
     elif arm in ('once','svdf'):
      action=None;event=-1
      if arm=='svdf' and assets['event'] is not None and assets['event']>=2:
       assert sha(assets['vectors_path'])==assets['vectors_sha256'];pv=torch.load(assets['vectors_path'],map_location='cpu',weights_only=False);action=pv['b']['simple'];event=int(assets['event']);assert 2<=event<=9
      tokens,trace,extra=engine['generate'](sample,cap=0,lam=1.,event=event,action=action)
      inputpath=assets[arm+'_inputs_path'];assert sha(inputpath)==assets[arm+'_inputs_sha256'];historical=load(inputpath)['hashes'];assert extra['hashes']==historical,('historical actual inputs differ',case['case_id'],arm)
      assert [r['step'] for r in trace if r.get('route_override_active')]==ref['metadata']['injection_prepared_steps']
      assert extra['effective_consumed']==ref['metadata']['effective_consumed_steps']
     else:
      pai_adapter.PAIAttention=TrackedPAIAttention;pai_adapter.select_token=observe_select
      result=pai_adapter.generate(model,tok,inp,alpha=.2,gamma=1.1,beta=.1,maximum=1024,config=cfg);tokens=result['token_ids'];extra=dict(metadata=result['metadata'],attention=result['attention'])
   except Exception as ex:
    save(run/(case['case_id']+'_'+arm+'_FAILED.json'),dict(case_id=case['case_id'],arm=arm,error=repr(ex),traceback=traceback.format_exc(),manifest_sha256=mh));raise
   finally:pai_adapter.PAIAttention=original_attention;pai_adapter.select_token=original_select
   torch.cuda.synchronize();seconds=time.monotonic()-begin
   if tokens!=ref['token_ids']:
    mismatch=next((i for i,(a,b)in enumerate(zip(tokens,ref['token_ids']))if a!=b),min(len(tokens),len(ref['token_ids'])))
    save(run/(case['case_id']+'_'+arm+'_TOKEN_MISMATCH.json'),dict(case_id=case['case_id'],arm=arm,first_mismatch=mismatch,actual=tokens,expected=ref['token_ids'],manifest_sha256=mh));raise AssertionError(('historical natural tokens differ',case['case_id'],arm,mismatch))
   validation=obs.validate(len(tokens));assert arm!='pai' or (len(processed)==len(pts) and select_calls[0]==2*len(tokens) and validation['negative_forward_calls']==len(tokens))
   validation.update(full_historical_tokens_exact=True,prompt_json_sha256=jh,prompt_array_bytes_sha256=bh,source_sha256=ref['source_sha256'],actual_inputs_exact=arm in ('once','svdf'),no_forced_tokens=True,seconds=seconds)
   if args.phase=='smoke':
    dumps=run/(case['case_id']+'_'+arm+'_raw_smoke.pt.gz')
    with dumps.open('xb')as rawfile:
     with gzip.GzipFile(fileobj=rawfile,mode='wb',mtime=0)as z:torch.save(obs.raw_dumps,z)
    for item in obs.raw_dumps:
     raw=item['raw_logits'].double();record=next(x for x in obs.records if x['step']==item['step']and x['layer_index']==item['layer_index'])
     for q in record['candidates']:
      tid=q['first_token_id'];assert q['rank']==int((raw>raw[tid]).sum())+1 and q['ties']==int((raw==raw[tid]).sum()) and abs(q['log_probability']-float(torch.log_softmax(raw,dim=-1)[tid]))<1e-10
    validation.update(raw_smoke_path=str(dumps),raw_smoke_sha256=sha(dumps),raw_smoke_cpu_recount_passed=True)
   record=dict(case_id=case['case_id'],arm=arm,n_tokens=len(tokens),token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),points=pts,candidates=candidates,records=obs.records,processed_readouts=processed,replay_verified=True,engineering=validation,method_metadata=extra,manifest_sha256=mh,own_natural_trajectory=True,shared_prefix=False)
   save(final,record);checks.append(dict(case_id=case['case_id'],arm=arm,seconds=seconds,n_tokens=len(tokens),output_sha256=sha(final),engineering=validation));print('ACCEPTED_NATURAL',case['case_id'],arm,len(tokens),seconds,flush=True);del obs,inp;gc.collect();torch.cuda.empty_cache()
 if args.phase=='smoke':save(O/('SMOKE_PASS_'+str(args.lane)+'.json'),dict(passed=True,manifest_sha256=mh,checks=checks,case_ids=m['smoke_case_ids']))
 else:save(run/'COMPLETE.json',dict(passed=True,manifest_sha256=mh,cases=[x['case_id']for x in cases],new_checks=checks))
if __name__=='__main__':main()
