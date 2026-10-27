"""Mechanical label-free collection only. This module never loads Gold or scores."""
import gzip,hashlib,importlib.util,json,os
from pathlib import Path
R=Path('/home/bingxing2/home/scx9ftv/gushuo')
E=R/'experiments/20261019_r1_ablation_complete/generation'
O=R/'output/experiments/20261019_r1_ablation_complete'
A=E.parent/'automation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def stem(s):return hashlib.sha256((s['dataset']+':'+str(s['id'])).encode()).hexdigest()[:20]
def expected():
 samples=read(E/'public.jsonl');d={(x['dataset'],str(x['id'])):x for x in read(E/'direction_noevent_reuse.jsonl')};u={(x['dataset'],str(x['id'])):x for x in read(E/'dynamic_reuse.jsonl')}
 result=[]
 for s in samples:
  result.extend((n,s) for n in ('start16','start32'))
  if s['dataset']=='visulogic':
   if (s['dataset'],str(s['id'])) not in d:result.extend((n,s) for n in ('true_only','weak_only','neg_original_soft','reverse','random_104729'))
   if (s['dataset'],str(s['id'])) not in u:result.append(('dynamic_f4',s))
 assert len(result)==5282
 return result
def collect():
 m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json');dest=A/'collected'
 if dest.exists():
  assert (dest/'BLIND_COLLECTION_RECEIPT.json').exists(),'incomplete previous collection retained; manual review required'
  return json.loads((dest/'BLIND_COLLECTION_RECEIPT.json').read_text())
 dest.mkdir();public={(s['dataset'],str(s['id'])):s for s in read(E/'public.jsonl')};rows=[]
 for name,s in expected():
  q=O/'completed'/name/stem(s);ac=json.loads((q/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh and ac['results_sha256']==sha(q/'results.jsonl')
  rr=read(q/'results.jsonl');assert len(rr)==1;r=rr[0]
  assert (r['method'],r['dataset'],str(r['id']))==(name,s['dataset'],str(s['id'])) and r['image_sha256']==s['image_sha256'] and not r['runtime_error']
  r['provenance']=dict(physical_reused=False,source=str(q/'results.jsonl'),source_sha256=sha(q/'results.jsonl'),manifest_sha256=mh);rows.append(r)
 cache={}
 def original(path,h,method):
  if path not in cache:
   assert sha(path)==h;cache[path]=read(path)
  return next(r for r in cache[path] if r['method']==method)
 for x in read(E/'direction_noevent_reuse.jsonl'):
  source=original(x['source'],x['source_sha256'],x['source_method'])
  for method in x['methods']:
   r=dict(source);r.update(x['method_fields_override']);r['method']=method;r['physical_seconds']=0.;r['seconds']=None
   r['effective_consumed_steps']=[];r['actual_perstep_geometry']=[];r['online_audit']={};r['forward_calls_main']=0
   r['provenance']=dict(physical_reused=True,reason='no eligible event; complete tokens and actual inputs exact to Once',source=x['source'],source_sha256=x['source_sha256'],fixed_source=x['fixed_source'],fixed_source_sha256=x['fixed_source_sha256'],source_seconds=x['source_once_seconds'],current_physical_seconds=0.,timing_is_not_fresh=True)
   rows.append(r)
 for x in read(E/'dynamic_reuse.jsonl'):
  source=original(x['source'],x['source_sha256'],x['source_method']);r=dict(source);r['method']='dynamic_f4';r['physical_seconds']=0.;r['seconds']=None
  r['provenance']=dict(physical_reused=True,reason='complete frozen dynamic historical output',source=x['source'],source_sha256=x['source_sha256'],source_seconds=source.get('seconds'),current_physical_seconds=0.,timing_is_not_fresh=True)
  rows.append(r)
 ids={(r['method'],r['dataset'],str(r['id'])) for r in rows};assert len(rows)==len(ids)==10512
 rows.sort(key=lambda r:(r['method'],r['dataset'],str(r['id'])))
 with gzip.open(dest/'COLLECTED_ROWS_BLIND.jsonl.gz','wt',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 spec=importlib.util.spec_from_file_location('blind_extract',str(A/'extract_rules.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 autos=[];pending={}
 for r in rows:
  s=public[(r['dataset'],str(r['id']))];x=dict(dataset=r['dataset'],question=s['question'],options=s.get('options',''),text=r['text'])
  v=module.resolve(x);review=hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
  need=v['status']!='resolved' or v['reason'] in ('cow_final_conclusion_region','cow_explicit_terminal_abstention')
  autos.append(dict(method=r['method'],dataset=r['dataset'],id=str(r['id']),review_id=review,needs_fulltext_review=need,**v))
  if need:pending[review]=dict(review_id=review,**x)
 with (dest/'AUTO_EXTRACTIONS_BLIND.jsonl').open('x') as f:
  for r in autos:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 with gzip.open(dest/'REQUIRED_FULLTEXT_REVIEW_BLIND.jsonl.gz','wt',encoding='utf-8') as f:
  for r in pending.values():f.write(json.dumps(r,ensure_ascii=False)+'\n')
 result=dict(passed=True,manifest_sha256=mh,logical_rows=10512,fresh_rows=5282,reused_rows=5230,unique_keys=len(ids),manual_review_unique=len(pending),labels_read=False,predictions_final=False,scoring_performed=False,rows_sha256=sha(dest/'COLLECTED_ROWS_BLIND.jsonl.gz'),auto_extractions_sha256=sha(dest/'AUTO_EXTRACTIONS_BLIND.jsonl'),note='Automated extraction remains provisional until fulltext review/freeze. No score, parameter choice, model selection or Gold load.')
 save(dest/'BLIND_COLLECTION_RECEIPT.json',result);return result
if __name__=='__main__':print(json.dumps(collect(),indent=2))
