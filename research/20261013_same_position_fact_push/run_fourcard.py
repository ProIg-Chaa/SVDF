from common import *
import argparse,gc,math,traceback

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--lane',type=int,required=True);a=ap.parse_args();assert a.phase=='full' and a.lane in (0,1,2,3)
 m,mh=check_manifest();positions=rows(E/'positions.jsonl');public={str(x['id']):x for x in rows(E/'public.jsonl')};registry={str(x['id']):x for x in rows(E/'registry.jsonl')}
 assert len(positions)==m['position_count'] and len({(p['dataset'],str(p['id']),p['position_id']) for p in positions})==len(positions)
 if a.phase=='smoke':
  smoke=m['smoke_cases'][a.lane];positions=[p for p in positions if p['position_id']==smoke['position_id'] and str(p['id'])==str(smoke['id'])];assert len(positions)==1
 else:
  for k in range(2):
   smoke_pass=json.loads((O/('SMOKE_PASS_'+str(k)+'.json')).read_text());assert smoke_pass['manifest_sha256']==mh and smoke_pass['passed'] and smoke_pass['smoke_zero_validation_confirmed']
  clearance=json.loads((E/'FULL_STORAGE_CLEARANCE.json').read_text());assert clearance['passed'] and clearance['manifest_sha256']==mh
  import shutil
  assert shutil.disk_usage(E).free>=m['minimum_free_bytes_before_full']
  positions=[p for i,p in enumerate(positions) if i%4==a.lane]
 run=O/a.phase/(os.environ.get('SLURM_JOB_ID','manual')+'_'+str(a.lane)+'_'+str(time.time_ns()));run.mkdir(parents=True);(run/'masks').mkdir()
 import torch,base
 from scale_model_loader import load_model
 from same_position_probe import SamePositionRuntime,frozen_difference
 torch.set_num_threads(4);model,processor,tok,mi=load_model(m['checkpoint'],run);engine=SamePositionRuntime(model,processor,tok,mi);accepted=[];reused=[]
 for plan in positions:
  sample=public[str(plan['id'])];done=O/'completed'/stem(plan)
  if done.exists():
   ac=json.loads((done/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh
   for n,h in ac['files_sha256'].items():assert sha(done/n)==h
   if a.phase=='smoke':
    reused_row=json.loads((done/'probes.json').read_text());checks=reused_row['zero_checks'];assert len(checks)==8 and all(z['full_candidate_logits_exact'] and z['actual_input_hashes_exact'] and z['fp64_per_token_exact'] for z in checks);reused.append(dict(id=str(plan['id']),position_id=plan['position_id'],acceptance_sha256=sha(done/'ACCEPTANCE.json')))
   continue
  folder=run/stem(plan);folder.mkdir();d=baseline(sample,registry[str(plan['id'])]);prefix=plan['prefix_token_ids'];k=len(prefix);assert k==plan['probe_token_index'] and prefix==d['bt'][:k]
  assert plan['durations']==[1,4] and k-4>=1 and max(len(c['token_ids']) for c in plan['candidates'])+k-1<1024
  assert {c['role'] for c in plan['candidates']}=={'correct','wrong'} and len(plan['candidates'])==2
  try:
   weak=base.sweep.create_mean_mask(sample['image'],run/'masks'/(str(sample['id'])+'.png')) if not (run/'masks'/(str(sample['id'])+'.png')).exists() else str(run/'masks'/(str(sample['id'])+'.png'))
   captures={};measurement=[];raw_payload={};update_payload={};zero_checks=[];hard_by_duration={}
   for duration in plan['durations']:
    start=k-duration;ti=engine.capture(sample,prefix,start,d['bh']);wi=engine.capture(dict(sample,image=weak),prefix,start);action,geometry=frozen_difference(ti,wi)
    captures[str(duration)]=dict(original=ti,weakened=wi,action=action,geometry=geometry,weak_image_sha256=sha(weak))
    if geometry['difference_norm']<=1e-12:assert not bool(torch.count_nonzero(action))
    configs=[('hard','hard',None,None),('ordinary_soft','soft',None,None),('svdf','svdf',action,None),('attention2','attention',None,2.),('attention4','attention',None,4.)]
    duration_raw={};duration_updates={};duration_reads={}
    for name,mode,act,mult in configs:
     rr=[];rv=[];uv=[]
     for cand in plan['candidates']:
      r,raw,updates=engine.read(sample,prefix,cand,duration,mode=mode,action=act,multiplier=mult,expected_hashes=d['bh'])
      if name=='hard':assert r['actual_input_hashes'][:k+1]==d['bh'][:k+1]
      if name=='ordinary_soft':
       assert torch.equal(updates[0]['soft'].flatten().float(),ti['s']) and torch.equal(updates[0]['hard'].flatten().float(),ti['h'])
       assert all(torch.equal(z['actual'],z['soft']) for z in updates)
      if name=='svdf':
       for z,g in zip(updates,r['feedback_geometry']):
        expected=z['hard']+action.to(z['hard'].dtype).reshape(1,1,-1);assert torch.equal(z['actual'],expected)
        g['nominal_rho']=geometry['rho'];g['actual_minus_nominal_radius']=g['actual_delta_norm']-geometry['rho'];g['actual_vs_nominal_radius_ratio']=g['actual_delta_norm']/geometry['rho'] if geometry['rho']>0 else None;g['nominal_radius_is_not_exact_bf16_update_norm']=True
      norms=[z['actual_delta_norm'] for z in r['feedback_geometry']];r['displacement_sum']=sum(norms);r['displacement_l2_total']=math.sqrt(sum(v*v for v in norms));r['displacement_per_forward']=norms
      rr.append(r);rv.append(dict(role=cand['role'],token_ids=cand['token_ids'],raw_logits=raw));uv.append(dict(role=cand['role'],updates=updates))
     assert torch.equal(rv[0]['raw_logits'][0],rv[1]['raw_logits'][0]),'candidate first logits differ'
     assert rr[0]['actual_input_hashes'][:k+1]==rr[1]['actual_input_hashes'][:k+1]
     role={r['role']:r for r in rr};measurement.append(dict(duration=duration,arm=name,readouts=rr,B=role['wrong']['phrase_log_probability']-role['correct']['phrase_log_probability'],word_B=role['wrong']['word_log_probability']-role['correct']['word_log_probability'],image_sha256=sample['image_sha256']))
     duration_raw[name]=rv;duration_updates[name]=uv;duration_reads[name]=rr
    hard_by_duration[duration]=duration_raw['hard']
    if a.phase=='smoke':
     for zname,zmode,zaction,zmult in [('svdf_zero','svdf',torch.zeros_like(action),None),('attention_zero','attention',None,1.)]:
      for ci,cand in enumerate(plan['candidates']):
       zr,zraw,_=engine.read(sample,prefix,cand,duration,mode=zmode,action=zaction,multiplier=zmult,expected_hashes=d['bh']);ref=duration_reads['hard'][ci]
       assert torch.equal(zraw,duration_raw['hard'][ci]['raw_logits']) and zr['actual_input_hashes']==ref['actual_input_hashes'] and zr['per_token']==ref['per_token']
       zero_checks.append(dict(duration=duration,mode=zname,role=cand['role'],full_candidate_logits_exact=True,actual_input_hashes_exact=True,fp64_per_token_exact=True))
    raw_payload[str(duration)]=duration_raw;update_payload[str(duration)]=duration_updates
   for ci in range(2):assert torch.equal(hard_by_duration[1][ci]['raw_logits'],hard_by_duration[4][ci]['raw_logits'])
   weak_reads=[];weak_raw=[];weak_updates=[]
   for cand in plan['candidates']:
    r,raw,updates=engine.read(dict(sample,image=weak),prefix,cand,1,mode='hard')
    norms=[z['actual_delta_norm'] for z in r['feedback_geometry']];r['displacement_sum']=sum(norms);r['displacement_l2_total']=math.sqrt(sum(v*v for v in norms));r['displacement_per_forward']=norms
    weak_reads.append(r);weak_raw.append(dict(role=cand['role'],token_ids=cand['token_ids'],raw_logits=raw));weak_updates.append(dict(role=cand['role'],updates=updates))
   assert torch.equal(weak_raw[0]['raw_logits'][0],weak_raw[1]['raw_logits'][0]) and weak_reads[0]['actual_input_hashes'][:k+1]==weak_reads[1]['actual_input_hashes'][:k+1]
   raw_payload['weak_hard']=weak_raw;update_payload['weak_hard']=weak_updates
   save_tensors(folder/'raw_logits.pt.gz',raw_payload);save_tensors(folder/'captures_and_updates.pt.gz',dict(captures=captures,updates=update_payload))
   save(folder/'probes.json',dict(dataset=plan['dataset'],id=str(plan['id']),position_id=plan['position_id'],cohort=plan['cohort'],stage=plan['stage'],probe_token_index=k,prefix_token_ids=prefix,candidates=plan['candidates'],position_plan=plan,diagnostic_only=True,initialization='shared Once: rawsoft step0 then hard until local window',measurements=measurement,weak_hard_readouts=weak_reads,weak_image_sha256=sha(weak),weak_hard_independent_fixed_reference=True,capture_geometry={q:z['geometry'] for q,z in captures.items()},zero_checks=zero_checks,hard_F1_F4_logits_exact=True,uncompressed_raw_logits_bytes=sum(x['raw_logits'].numel()*x['raw_logits'].element_size() for du in [raw_payload['1'],raw_payload['4']] for arr in du.values() for x in arr)))
   names=['probes.json','raw_logits.pt.gz','captures_and_updates.pt.gz'];save(folder/'ACCEPTANCE.json',dict(passed=True,manifest_sha256=mh,files_sha256={n:sha(folder/n) for n in names},complete_branches=10,weak_hard_candidate_reference_count=2,independent_candidate_replays=True,first_logits_exact=True,intervention_consumed_before_readout=True,candidate_suffix_hard=True,full_phrase_saved=True))
   done.parent.mkdir(exist_ok=True);done.symlink_to(folder,target_is_directory=True);accepted.append(dict(id=str(plan['id']),position_id=plan['position_id']));print('ACCEPTED_POSITION',str(plan['id']),plan['position_id'],k,flush=True)
  except BaseException as ex:
   save(folder/'ERROR.json',dict(type=type(ex).__name__,error=str(ex),traceback=traceback.format_exc()));raise
  finally:gc.collect();torch.cuda.empty_cache()
 if a.phase=='smoke':save(O/('SMOKE_PASS_'+str(a.lane)+'.json'),dict(passed=True,manifest_sha256=mh,position_ids=accepted,reused_positions=reused,zero_exact_newly_checked=bool(accepted),smoke_zero_validation_confirmed=True))
 else:save(run/'LANE_COMPLETE.json',dict(passed=True,manifest_sha256=mh,n=len(positions),newly_accepted=len(accepted)))
if __name__=='__main__':main()
