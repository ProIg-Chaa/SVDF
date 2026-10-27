import os,sys,json,hashlib,time,gzip
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent.parent;O=E/'output'
sys.path.insert(0,str(E/'source'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)
def stem(s):return hashlib.sha256((s['dataset']+':'+str(s['id'])+':'+s['position_id']).encode()).hexdigest()[:20]
def save_tensors(p,x):
 import torch
 with gzip.open(p,'wb',compresslevel=6) as f:torch.save(x,f)
def load_tensors(p):
 import torch
 with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=True)
def check_manifest():
 m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
 for n,h in m['code_sha256'].items():assert sha(E/n)==h,n
 for n,h in m['source_files'].items():assert sha(E/'source'/n)==h,n
 for n,h in m['checkpoint_files'].items():assert sha(Path(m['checkpoint']['path'])/n)==h,n
 for n in ('public','registry','positions'):assert sha(E/(n+'.jsonl'))==m[n+'_sha256'],n
 return m,mh
def baseline(s,ref):
 import torch
 assert sha(s['image'])==s['image_sha256'];parent=Path(ref['folder'])
 for n,h in ref['files'].items():assert sha(parent/n)==h,n
 once=next(x for x in rows(parent/'results.jsonl') if x['method']=='plain_soft_once')
 bh=json.loads((parent/'plain_soft_once_inputs.json').read_text())['hashes']
 assert len(bh)==len(once['generated_token_ids'])
 return dict(bt=once['generated_token_ids'],bh=bh,parent=str(parent))
