import os,sys,json,hashlib,time
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent.parent;O=E/'output'
sys.path.insert(0,str(E/'source'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)
def stem(s):return hashlib.sha256((s['dataset']+':'+str(s['id'])).encode()).hexdigest()[:20]
def check_manifest():
 m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
 for n,h in m['code_sha256'].items():assert sha(E/n)==h,n
 for n,h in m['source_files'].items():assert sha(E/'source'/n)==h,n
 for n,h in m['checkpoint_files'].items():assert sha(Path(m['checkpoint']['path'])/n)==h,n
 assert sha(E/'public.jsonl')==m['public_sha256'] and sha(E/'registry.jsonl')==m['registry_sha256']
 return m,mh
def sample_inputs(s,ref):
 import torch
 assert sha(s['image'])==s['image_sha256'];parent=Path(ref['folder'])
 for n,h in ref['files'].items():assert sha(parent/n)==h,n
 assert sha(Path(ref['reuse_folder'])/'results.jsonl')==ref['reuse_results_sha256']
 old=rows(parent/'results.jsonl');once=next(x for x in old if x['method']=='plain_soft_once');real=next(x for x in old if x['method']=='plain_residual_only')
 bt=once['generated_token_ids'];bh=json.loads((parent/'plain_soft_once_inputs.json').read_text())['hashes'];event=ref['event'];assert 2<=event<=9
 pv=torch.load(parent/'vectors.pt',map_location='cpu',weights_only=False)['b'];si=pv['true']['soft_embedding'].float();hard=pv['true']['hard_embedding'].float();rho=float((si-hard).norm())
 assert rho>0 and torch.isfinite(si).all();raw_si=si*(rho/float(si.norm()));action=pv['simple']
 return dict(once=once,real=real,bt=bt,bh=bh,event=event,discovery=bt[:event+1],pv=pv,rho=rho,raw_si=raw_si,action=action,parent=parent)
def match_geometry(gs,refs,rho):
 rr={g.get('step',g.get('feedback_slot')):g for g in refs}
 for g in gs:
  k=g.get('step',g.get('feedback_slot'));ref=rr[k]['actual_delta_norm']
  g['relative_error_to_rho']=abs(g['actual_delta_norm']-rho)/rho;g['relative_error_to_real']=abs(g['actual_delta_norm']-ref)/ref
  assert g['relative_error_to_rho']<=.02 and g['relative_error_to_real']<=.02,('actual BF16 norm mismatch',g)
