"""Independent CPU reconstruction of every reported phrase probability from full raw logits."""
from common import *
import argparse,math,datetime

def main():
 import torch
 ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');a=ap.parse_args();pm=json.loads((E/'probe_manifest.json').read_text());ph=sha(E/'probe_manifest.json');summary=[]
 for pr in rows(E/'probe_registry.jsonl'):
  if not pr['eligible']:continue
  p=O/'probes/completed'/stem(pr)
  if not p.exists():
   assert a.allow_partial,('missing',pr['id']);continue
  ac=json.loads((p/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['probe_manifest_sha256']==ph and ac['raw_logits_sha256']==sha(p/'raw_logits.pt') and ac['probe_sha256']==sha(p/'probes.json')
  row=json.loads((p/'probes.json').read_text());raw=torch.load(p/'raw_logits.pt',map_location='cpu',weights_only=True);bs={};corr={}
  for arm in row['measurements']:
   name=arm['arm'];assert name in raw;reconstructed={}
   assert torch.equal(raw[name][0]['raw_logits'][0],raw[name][1]['raw_logits'][0])
   for r,v in zip(arm['readouts'],raw[name]):
    assert r['role']==v['role'] and r['token_ids']==v['token_ids'];tt=v['raw_logits'];assert tt.ndim==2 and tt.shape[1]==152064 and torch.isfinite(tt).all();assert len(tt)==len(v['token_ids'])
    lp=torch.log_softmax(tt.float(),dim=-1);vals=[float(lp[k,t]) for k,t in enumerate(v['token_ids'])];joint=sum(vals)
    assert abs(joint-r['phrase_log_probability'])<1e-6
    for k,z in enumerate(r['per_token']):assert abs(vals[k]-z['log_probability'])<1e-6 and float(tt[k,v['token_ids'][k]])==z['raw_logit']
    reconstructed[r['role']]=joint
   b=reconstructed['wrong']-reconstructed['correct'];assert abs(b-arm['B'])<1e-6;bs[name]=b;corr[name]=reconstructed['correct']
  summary.append(dict(dataset='mmvp',id=pr['id'],B=bs,delta_B={k:v-bs['original'] for k,v in bs.items()},correct_phrase_logp=corr,image_helps_correct_but_still_wrong=corr['original']>corr['weakened'] and bs['original']>0))
 expected=pm['eligible_count'];complete=len(summary)==expected
 out=E/'evaluation';out.mkdir(exist_ok=True);stamp=time.time_ns()
 save(out/('LOGITS_RECOUNT_'+str(stamp)+'.json'),dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),probe_manifest_sha256=ph,complete=complete,n=len(summary),expected=expected,all_logits_reconstructed=True,records=summary))
 print(json.dumps(dict(complete=complete,n=len(summary),expected=expected,all_logits_reconstructed=True)))
if __name__=='__main__':main()
