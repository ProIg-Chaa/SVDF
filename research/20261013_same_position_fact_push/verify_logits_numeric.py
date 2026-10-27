# Independent CPU reconstruction uses target_raw_logit - logsumexp, not runner log_softmax.
from common import *
import argparse,math,datetime

# CPU/GPU FP32 reductions need precision-aware validation. BF16 inputs remain exact.
NORM_FLOAT32_EPSILON=2.0**-23

def norm_record_bound(reference,dimension):
 return max(1e-12,(math.ceil(math.log2(max(int(dimension),2)))+4)*NORM_FLOAT32_EPSILON*abs(reference))

def check_norm_record(reference,recorded,dimension):
 assert math.isfinite(reference) and math.isfinite(recorded) and reference>=0 and recorded>=0
 if reference==0:assert recorded==0
 bound=norm_record_bound(reference,dimension)
 assert abs(reference-recorded)<=bound,('FP32 norm outside reduction bound',reference,recorded,bound,dimension)
 return bound

def main():
 import torch
 torch.set_num_threads(4)
 ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['smoke','full'],required=True);ap.add_argument('--allow-partial',action='store_true');a=ap.parse_args();m,mh=check_manifest();plans=rows(E/'positions.jsonl')
 if a.scope=='smoke':plans=[p for p in plans if any(str(p['id'])==str(s['id']) and p['position_id']==s['position_id'] for s in m['smoke_cases'])]
 records=[];checked_tokens=0;max_error=0.;receipts=[];max_norm_error=0.;max_norm_relative_error=0.
 for p in plans:
  folder=O/'completed'/stem(p)
  if not folder.exists():assert a.allow_partial,('missing',p['id'],p['position_id']);continue
  ac=json.loads((folder/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh
  for n,h in ac['files_sha256'].items():assert sha(folder/n)==h,n
  row=json.loads((folder/'probes.json').read_text());raw=load_tensors(folder/'raw_logits.pt.gz');vectors=load_tensors(folder/'captures_and_updates.pt.gz');assert row['position_plan']==p
  bydur={}
  for duration in (1,4):
   cap=vectors['captures'][str(duration)];ti,wi=cap['original'],cap['weakened'];si,sw,h=ti['s'].float(),wi['s'].float(),ti['h'].float()
   assert torch.equal(h,wi['h']) and ti['capture_slot']==wi['capture_slot']==len(p['prefix_token_ids'])-duration
   rho=float((si-h).norm());delta=(si-h)-(sw-h);dn=float(delta.norm());action=delta*(rho/dn) if dn>1e-12 else torch.zeros_like(delta)
   assert torch.equal(action,cap['action']) and abs(rho-cap['geometry']['rho'])<1e-10
   for val in (ti['p'],wi['p'],si,sw,h,action):assert torch.isfinite(val).all()
   arms={}
   for arm in [z for z in row['measurements'] if z['duration']==duration]:
    name=arm['arm'];rr=arm['readouts'];rv=raw[str(duration)][name];uv=vectors['updates'][str(duration)][name];assert len(rr)==len(rv)==len(uv)==2 and torch.equal(rv[0]['raw_logits'][0],rv[1]['raw_logits'][0])
    role={}
    for r,v,u in zip(rr,rv,uv):
     assert r['role']==v['role']==u['role'] and r['token_ids']==v['token_ids'];tt=v['raw_logits'];assert tt.dtype==torch.bfloat16 and tt.ndim==2 and tt.shape==(len(v['token_ids']),m['checkpoint']['vocab_size']) and torch.isfinite(tt).all()
     vals=[]
     for offset,(tid,z) in enumerate(zip(v['token_ids'],r['per_token'])):
      rf=tt[offset].double();lp=float(rf[tid]-torch.logsumexp(rf,dim=-1));err=abs(lp-z['log_probability']);max_error=max(max_error,err);assert err<1e-10
      assert float(tt[offset,tid])==z['raw_logit'] and int((tt[offset]>tt[offset,tid]).sum())+1==z['rank']
      assert hashlib.sha256(tt[offset].view(torch.uint8).numpy().tobytes()).hexdigest()==z['raw_logits_sha256']
      vals.append(lp);checked_tokens+=1
     phrase=sum(vals);word=sum(vals[:r['word_token_count']]);assert abs(phrase-r['phrase_log_probability'])<1e-10 and abs(word-r['word_log_probability'])<1e-10
     assert abs(math.exp(phrase)-r['phrase_probability'])<1e-12 and abs(math.exp(word)-r['word_probability'])<1e-12
     norms=[];norm_bounds=[]
     assert len(u['updates'])==duration and [z['feedback_slot'] for z in u['updates']]==list(range(len(p['prefix_token_ids'])-duration,len(p['prefix_token_ids'])))
     for update,g in zip(u['updates'],r['feedback_geometry']):
      hard,soft,actual=update['hard'],update['soft'],update['actual'];norm=float((actual.double()-hard.double()).norm());norms.append(norm);norm_bounds.append(check_norm_record(norm,g['actual_delta_norm'],actual.numel()));norm_error=abs(norm-g['actual_delta_norm']);max_norm_error=max(max_norm_error,norm_error);max_norm_relative_error=max(max_norm_relative_error,norm_error/max(norm,1e-30))
      if name=='ordinary_soft':assert torch.equal(actual,soft)
      elif name=='svdf':assert torch.equal(actual,hard+action.to(hard.dtype).reshape(1,1,-1))
      else:assert torch.equal(actual,hard)
     assert abs(sum(norms)-r['displacement_sum'])<=sum(norm_bounds)+1e-10 and abs(math.sqrt(sum(z*z for z in norms))-r['displacement_l2_total'])<=math.sqrt(sum(z*z for z in norm_bounds))+1e-10
     role[r['role']]=dict(phrase_logp=phrase,word_logp=word,first_token_logp=vals[0],phrase_probability=math.exp(phrase),word_probability=math.exp(word),first_token_probability=math.exp(vals[0]),displacements=norms,displacement_sum=sum(norms),displacement_l2_total=math.sqrt(sum(z*z for z in norms)))
    b=role['wrong']['phrase_logp']-role['correct']['phrase_logp'];assert abs(b-arm['B'])<1e-10
    wb=role['wrong']['word_logp']-role['correct']['word_logp'];assert abs(wb-arm['word_B'])<1e-10
    arms[name]=dict(B=b,word_B=wb,readouts=role)
   assert set(arms)=={'hard','ordinary_soft','svdf','attention2','attention4'}
   for name,z in arms.items():z['delta_B']=z['B']-arms['hard']['B'];z['delta_correct_logp']=z['readouts']['correct']['phrase_logp']-arms['hard']['readouts']['correct']['phrase_logp'];z['delta_wrong_logp']=z['readouts']['wrong']['phrase_logp']-arms['hard']['readouts']['wrong']['phrase_logp']
   bydur[str(duration)]=arms
  assert abs(bydur['1']['hard']['B']-bydur['4']['hard']['B'])<1e-10
  for ci in range(2):assert torch.equal(raw['1']['hard'][ci]['raw_logits'],raw['4']['hard'][ci]['raw_logits'])
  if a.scope=='smoke':assert len(row['zero_checks'])==8 and all(z['full_candidate_logits_exact'] and z['actual_input_hashes_exact'] and z['fp64_per_token_exact'] for z in row['zero_checks'])
  weak={};assert torch.equal(raw['weak_hard'][0]['raw_logits'][0],raw['weak_hard'][1]['raw_logits'][0])
  for r,v in zip(row['weak_hard_readouts'],raw['weak_hard']):
   assert r['role']==v['role'] and r['token_ids']==v['token_ids'];tt=v['raw_logits'];assert tt.dtype==torch.bfloat16 and tt.shape==(len(v['token_ids']),m['checkpoint']['vocab_size']) and torch.isfinite(tt).all();vals=[]
   for offset,(tid,z) in enumerate(zip(v['token_ids'],r['per_token'])):
    rf=tt[offset].double();lp=float(rf[tid]-torch.logsumexp(rf,dim=-1));err=abs(lp-z['log_probability']);max_error=max(max_error,err);assert err<1e-10 and float(tt[offset,tid])==z['raw_logit']
    assert hashlib.sha256(tt[offset].view(torch.uint8).numpy().tobytes()).hexdigest()==z['raw_logits_sha256'];vals.append(lp);checked_tokens+=1
   phrase=sum(vals);word=sum(vals[:r['word_token_count']]);assert abs(phrase-r['phrase_log_probability'])<1e-10 and abs(word-r['word_log_probability'])<1e-10
   weak[r['role']]=dict(phrase_logp=phrase,phrase_probability=math.exp(phrase),word_logp=word,first_token_logp=vals[0])
  weak_B=weak['wrong']['phrase_logp']-weak['correct']['phrase_logp'];original_correct=bydur['1']['hard']['readouts']['correct']['phrase_logp'];image_support=original_correct-weak['correct']['phrase_logp']
  records.append(dict(dataset=p['dataset'],id=str(p['id']),position_id=p['position_id'],cohort=p['cohort'],stage=p['stage'],probe_token_index=p['probe_token_index'],relative_position=p['relative_position'],same_first_token=p['candidates'][0]['token_ids'][0]==p['candidates'][1]['token_ids'][0],durations=bydur,weak_hard=weak,weak_B=weak_B,target_correct_image_support_logp=image_support,image_helps_correct_but_wrong_dominates=image_support>0 and bydur['1']['hard']['B']>0))
  receipts.append(dict(id=str(p['id']),position_id=p['position_id'],acceptance_sha256=sha(folder/'ACCEPTANCE.json')))
 out=E/'evaluation'/a.scope;out.mkdir(parents=True,exist_ok=True);stamp=str(time.time_ns());path=out/('LOGITS_RECOUNT_'+stamp+'.json');complete=len(records)==len(plans)
 payload=dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=a.scope,manifest_sha256=mh,complete=complete,n=len(records),expected=len(plans),checked_tokens=checked_tokens,max_log_probability_abs_error=max_error,full_vocabulary_logits_recomputed=True,all_update_vectors_recomputed=True,displacement_authority='CPU FP64 norm of actual.double()-hard.double() from lossless BF16 inputs',gpu_norm_record_check='(ceil(log2(dimension))+4)*float32_epsilon*abs(fp64_reference), floor1e-12; exact zero retained; totals propagated from per-step bounds',numeric_verifier_sha256=sha(Path(__file__)),max_gpu_norm_abs_error=max_norm_error,max_gpu_norm_relative_error=max_norm_relative_error,records=records,receipts=receipts)
 save(path,payload);(out/'latest.json').write_text(json.dumps(dict(file=str(path),sha256=sha(path),complete=complete,n=len(records)),indent=2));print(json.dumps(dict(path=str(path),complete=complete,n=len(records),expected=len(plans),checked_tokens=checked_tokens,max_error=max_error)),flush=True)
 assert complete or a.allow_partial
if __name__=='__main__':main()
