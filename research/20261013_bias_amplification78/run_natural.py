from common import *
import argparse,gc

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--lane',type=int,required=True);a=ap.parse_args();assert a.lane in (0,1)
 m,mh=check_manifest();samples=rows(E/'public.jsonl');reg={str(x['id']):x for x in rows(E/'registry.jsonl')};out=O/'natural'
 if a.phase=='smoke':samples=[x for x in samples if str(x['id'])==m['smoke_ids'][a.lane]]
 else:
  for k in range(2):assert json.loads((out/('SMOKE_PASS_'+str(k)+'.json')).read_text())['manifest_sha256']==mh
  samples=[x for i,x in enumerate(samples) if i%2==a.lane]
 run=out/a.phase/(os.environ.get('SLURM_JOB_ID','manual')+'_'+str(a.lane)+'_'+str(time.time_ns()));run.mkdir(parents=True);(run/'masks').mkdir()
 import torch,base
 from scale_model_loader import load_model
 from engine import make_engine,eligible
 from attention_boost import VisualAttentionBoost
 torch.set_num_threads(4);model,processor,tok,mi=load_model(m['checkpoint'],run);engine=make_engine(model,processor,tok,mi);generate=engine['generate'];checks=[]
 for s in samples:
  done=out/'completed'/stem(s)
  if done.exists():
   ac=json.loads((done/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh and ac['results_sha256']==sha(done/'results.jsonl');continue
  folder=run/stem(s);folder.mkdir();ref=reg[str(s['id'])];d=sample_inputs(s,ref);event=d['event'];check=dict(id=s['id'],event=event)
  try:
   dt,tr,dm=generate(s,cap=0,lam=.95,stop_first=True);assert dt==d['discovery'] and dm['hashes']==d['bh'][:event+1] and eligible(tr[-1]);check['discovery_exact']=True
   if a.phase=='smoke':
    nt,nh=engine['native_plain_once'](s);assert nt==d['bt'] and nh==d['bh']
    with VisualAttentionBoost(model,event,1.) as obs:zt,_,zm=generate(s,cap=0,lam=.95)
    assert zt==nt and zm['hashes']==nh;obs.validate(len(zt))
    rt,_,rm=generate(s,cap=0,lam=.95,prefix=d['discovery'],event=event,action=d['action']);assert rt==d['real']['generated_token_ids'] and rm['hashes']==json.loads((d['parent']/'plain_residual_only_inputs.json').read_text())['hashes']
    weak=base.sweep.create_mean_mask(s['image'],run/'masks'/(s['id']+'.png'))
    for cond,img in [('true',s['image']),('mask',weak)]:
     v,vm=engine['capture'](s,img,d['discovery'],event,lam=.95);assert torch.equal(v['soft_embedding'],d['pv'][cond]['soft_embedding']) and torch.equal(v['hard_embedding'],d['pv'][cond]['hard_embedding'])
    check.update(native_baseline_exact=True,zero_attention_exact=True,real_replay_exact=True,paired_capture_exact=True)
   tt,tr,mm=generate(s,cap=0,lam=.95,prefix=d['discovery'],event=event,action=d['raw_si']);assert tt[:event+1]==d['discovery'] and mm['hashes'][:event+1]==d['bh'][:event+1];assert not any(x.get('lead_refinement_active') for x in tr)
   for g in mm['residual_geometry']:g['consumed']=g['step']<len(tt)-1
   match_geometry([g for g in mm['residual_geometry'] if g['consumed']],d['real']['residual_feedback_geometry'],d['rho'])
   assert mm['effective_consumed']==list(range(event,min(event+4,len(tt)-1))),('feedback schedule until natural EOS',s['id'],mm['effective_consumed'])
   row=dict(dataset=s['dataset'],id=s['id'],method='raw_sI',event=event,image_sha256=s['image_sha256'],generated_token_ids=tt,text=tok.decode(tt,skip_special_tokens=False),runtime_error=None,seconds=mm['seconds'],residual_geometry=mm['residual_geometry'],source_geometry=dict(rho=d['rho'],sI_norm=float(d['pv']['true']['soft_embedding'].float().norm())),maxed1024=len(tt)==1024,feedback_consumed_count=len(mm['effective_consumed']),early_eos_before_four=len(tt)-1<event+4)
   with (folder/'results.jsonl').open('x') as f:f.write(json.dumps(row)+'\n')
   save(folder/'raw_sI_inputs.json',dict(hashes=mm['hashes']));save(folder/'raw_sI_trace.json',tr);save(folder/'engineering.json',check)
   save(folder/'ACCEPTANCE.json',dict(passed=True,manifest_sha256=mh,results_sha256=sha(folder/'results.jsonl'),engineering_sha256=sha(folder/'engineering.json'),reused_natural_results=ref['reuse_folder'],reused_sha256=ref['reuse_results_sha256']))
   done.parent.mkdir(exist_ok=True);done.symlink_to(folder,target_is_directory=True);checks.append(check);print('ACCEPTED_NATURAL',s['id'],event,len(tt),flush=True)
  except BaseException as ex:
   save(folder/'ERROR.json',dict(type=type(ex).__name__,error=str(ex)));raise
  finally:gc.collect();torch.cuda.empty_cache()
 if a.phase=='smoke':save(out/('SMOKE_PASS_'+str(a.lane)+'.json'),dict(passed=True,manifest_sha256=mh,checks=checks))
 else:save(run/'LANE_COMPLETE.json',dict(passed=True,manifest_sha256=mh,n=len(samples)))
if __name__=='__main__':main()
