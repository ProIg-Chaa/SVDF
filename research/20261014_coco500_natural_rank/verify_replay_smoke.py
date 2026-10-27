"""CPU-only independent recount of COCO replay smoke raw full-vocabulary logits."""
import datetime,gzip,hashlib,json,math,sys,traceback
from pathlib import Path
import numpy as np
import torch
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with(gzip.open(p,'rt')if str(p).endswith('.gz')else Path(p).open())as f:return json.load(f)
def check():
 m=load(E/'GENERATION_MANIFEST.json');mh=sha(E/'GENERATION_MANIFEST.json');rm=load(E/'REPLAY_MANIFEST.json');rh=sha(E/'REPLAY_MANIFEST.json');g=load(E/'output/replay/SMOKE_PASS.json')
 assert g['passed']and g['generation_manifest_sha256']==mh and g['replay_manifest_sha256']==rh
 assert sha(E/rm['oracle_path'])==rm['oracle_sha256']
 checks=[];maximum_error=0.;n_raw=0
 for cid in m['smoke_case_ids']:
  for arm in ['cot','once','svdf','pai']:
   p=E/'output/replay/completed'/cid/(arm+'.json.gz');r=load(p);src=E/'output/generation/completed'/cid/(arm+'.json.gz');ref=load(src)
   assert r['generation_manifest_sha256']==mh and r['replay_manifest_sha256']==rh
   assert r['generated_token_ids']==ref['generated_token_ids']and r['actual_input_hashes']==ref['actual_input_hashes']and r['source_generation_sha256']==sha(src)
   if r.get('replay_status')=='no_eligible_candidate':
    assert r['replay_verified']is None and not r['records'];checks.append(dict(case_id=cid,arm=arm,status='no_eligible_candidate'));continue
   assert r['replay_verified']is True and r['own_natural_trajectory']is True and r['shared_prefix']is False
   for k in ['full_historical_tokens_exact','actual_inputs_exact','no_forced_tokens','final_layer_exact','final_norm_once','hook_cleanup_exact','prefill_last_query_only']:assert r['engineering'][k]is True,k
   pair={(v['step'],li)for v in r['points']for li in range(28)}
   assert len(r['records'])==len(pair)and {(v['step'],v['layer_index'])for v in r['records']}==pair
   q=Path(r['raw_smoke_path']);assert sha(q)==r['raw_smoke_sha256']
   with gzip.open(q,'rb')as f:raws=torch.load(f,map_location='cpu',weights_only=False)
   assert {(v['step'],v['layer_index'])for v in raws}=={(min(v['step']for v in r['points']),23),(min(v['step']for v in r['points']),27)}
   counts=[]
   for v in raws:
    n_raw+=1;raw=v['raw_logits'];a=raw.double().numpy();assert len(a)==m['checkpoint']['vocab_size']and np.isfinite(a).all()
    rec=next(z for z in r['records']if(z['step'],z['layer_index'])==(v['step'],v['layer_index']))
    assert hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()==rec['raw_logits_sha256']
    peak=float(a.max());lse=peak+math.log(math.fsum(math.exp(float(x)-peak)for x in a));assert abs(lse-rec['logsumexp'])<1e-10
    for cand in rec['candidates']:
     tid=cand['first_token_id'];value=float(a[tid]);rank=1+int(np.count_nonzero(a>value));ties=int(np.count_nonzero(a==value));lp=value-lse;err=abs(lp-cand['log_probability']);maximum_error=max(maximum_error,err)
     assert rank==cand['rank']and ties==cand['ties']and err<1e-10 and abs(math.exp(lp)-cand['probability'])<1e-12
     counts.append(dict(step=v['step'],layer=v['layer_index']+1,role=cand['role'],rank=rank,ties=ties,error=err))
   if arm=='pai':assert len(r['processed_readouts'])==len(r['points'])and r['engineering']['negative_forward_calls']==r['n_tokens']
   checks.append(dict(case_id=cid,arm=arm,status='passed',output_sha256=sha(p),raw_recounts=counts))
 return dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),generation_manifest_sha256=mh,replay_manifest_sha256=rh,smoke_gate_sha256=sha(E/'output/replay/SMOKE_PASS.json'),checks=checks,n_arms=len(checks),n_raw_layers=n_raw,max_log_probability_error=maximum_error,method='CPU NumPy comparisons and Python math.fsum full-vocabulary logsumexp; no new generation')
try:receipt=check()
except Exception as ex:receipt=dict(passed=False,error=repr(ex),traceback=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=E/'self_audit/contrast_mechanism_geometry'/('REPLAY_SMOKE_INDEPENDENT_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps(dict(path=str(out),sha256=sha(out),passed=receipt['passed'],error=receipt.get('error'))));sys.exit(0 if receipt['passed']else 1)
