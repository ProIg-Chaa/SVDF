from common import *
import argparse,gc

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--lane',type=int,required=True);a=ap.parse_args();assert a.lane in (0,1)
 m,mh=check_manifest();pm=json.loads((E/'probe_manifest.json').read_text());ph=sha(E/'probe_manifest.json');assert pm['base_manifest_sha256']==mh and sha(E/'probe_registry.jsonl')==pm['probe_registry_sha256'] and sha(E/'annotation/facts_frozen.jsonl')==pm['annotation_sha256']
 for n,h in pm['code_sha256'].items():assert sha(E/n)==h,n
 assert all(pm['smoke_ids']);eligible=[x for x in rows(E/'probe_registry.jsonl') if x['eligible']];public={str(x['id']):x for x in rows(E/'public.jsonl')};reg={str(x['id']):x for x in rows(E/'registry.jsonl')};out=O/'probes'
 if a.phase=='smoke':eligible=[x for x in eligible if x['id']==pm['smoke_ids'][a.lane]]
 else:
  for k in range(2):assert json.loads((out/('SMOKE_PASS_'+str(k)+'.json')).read_text())['probe_manifest_sha256']==ph
  eligible=[x for i,x in enumerate(eligible) if i%2==a.lane]
 run=out/a.phase/(os.environ.get('SLURM_JOB_ID','manual')+'_'+str(a.lane)+'_'+str(time.time_ns()));run.mkdir(parents=True);(run/'masks').mkdir()
 import torch,base
 from scale_model_loader import load_model
 from engine import make_engine
 from probe_common import read_candidate
 torch.set_num_threads(4);model,processor,tok,mi=load_model(m['checkpoint'],run);engine=make_engine(model,processor,tok,mi);checks=[]
 for pr in eligible:
  s=public[pr['id']];done=out/'completed'/stem(s)
  if done.exists():
   ac=json.loads((done/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['probe_manifest_sha256']==ph and ac['probe_sha256']==sha(done/'probes.json') and ac['raw_logits_sha256']==sha(done/'raw_logits.pt');continue
  folder=run/stem(s);folder.mkdir();d=sample_inputs(s,reg[pr['id']]);prefix=pr['prefix_token_ids'];event=d['event'];assert prefix==d['bt'][:pr['probe_token_index']] and len(prefix)>=event+4
  try:
   weak=base.sweep.create_mean_mask(s['image'],run/'masks'/(s['id']+'.png'));measurements=[];raw_payload={}
   configs=[('original',s,None,None),('weakened',dict(s,image=weak),None,None),('attention2',s,None,2.),('attention4',s,None,4.),('raw_sI',s,d['raw_si'],None),('real',s,d['action'],None)]
   for name,ss,action,mult in configs:
    eng=make_engine(model,processor,tok,mi) if name=='weakened' else engine;ev=-1 if name=='weakened' else event;rr=[];raw_payload[name]=[]
    for cand in pr['candidates']:
     r,raw=read_candidate(model,eng,ss,prefix,cand,ev,d['bh'],action,mult)
     r['first_logits_sha256']=hashlib.sha256(raw[0].view(torch.uint8).numpy().tobytes()).hexdigest();rr.append(r);raw_payload[name].append(dict(role=cand['role'],token_ids=cand['token_ids'],raw_logits=torch.stack(raw)))
    assert rr[0]['first_logits_sha256']==rr[1]['first_logits_sha256'];assert rr[0]['actual_input_hashes'][:len(prefix)+1]==rr[1]['actual_input_hashes'][:len(prefix)+1]
    if action is not None:assert len(rr[0]['residual_geometry'])==4
    measurements.append(dict(arm=name,readouts=rr,B=rr[0]['phrase_log_probability']-rr[1]['phrase_log_probability'],image_sha256=sha(ss['image'])))
   by={x['arm']:x for x in measurements}
   for j in range(2):match_geometry(by['raw_sI']['readouts'][j]['residual_geometry'],by['real']['readouts'][j]['residual_geometry'],d['rho'])
   zero_check=None
   if a.phase=='smoke':
    r,z=read_candidate(model,engine,s,prefix,pr['candidates'][0],event,d['bh'],None,1.)
    zz=hashlib.sha256(z[0].view(torch.uint8).numpy().tobytes()).hexdigest();assert zz==by['original']['readouts'][0]['first_logits_sha256'];assert r['per_token']==by['original']['readouts'][0]['per_token'];assert r['actual_input_hashes']==by['original']['readouts'][0]['actual_input_hashes'];zero_check=dict(exact_logits=True,exact_inputs=True,exact_multitoken_readout=True)
   torch.save(raw_payload,folder/'raw_logits.pt')
   save(folder/'probes.json',dict(dataset='mmvp',id=s['id'],event=event,probe_token_index=len(prefix),prefix_text=pr['prefix_text'],prefix_token_ids=prefix,candidates=pr['candidates'],diagnostic_only=True,measurements=measurements,zero_smoke=zero_check))
   save(folder/'ACCEPTANCE.json',dict(passed=True,manifest_sha256=mh,probe_manifest_sha256=ph,probe_sha256=sha(folder/'probes.json'),raw_logits_sha256=sha(folder/'raw_logits.pt'),same_text=True,isolated_runs=True,actual_norm_matched=True))
   done.parent.mkdir(exist_ok=True);done.symlink_to(folder,target_is_directory=True);checks.append(s['id']);print('ACCEPTED_PROBE',s['id'],len(prefix),flush=True)
  except BaseException as ex:
   save(folder/'ERROR.json',dict(type=type(ex).__name__,error=str(ex)));raise
  finally:gc.collect();torch.cuda.empty_cache()
 if a.phase=='smoke':save(out/('SMOKE_PASS_'+str(a.lane)+'.json'),dict(passed=True,manifest_sha256=mh,probe_manifest_sha256=ph,ids=checks))
 else:save(run/'LANE_COMPLETE.json',dict(passed=True,manifest_sha256=mh,probe_manifest_sha256=ph,n=len(eligible)))
if __name__=='__main__':main()
