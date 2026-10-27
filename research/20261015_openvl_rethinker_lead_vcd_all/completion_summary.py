import json,gzip,hashlib,datetime,os,collections
from pathlib import Path
E=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,obj):
 tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(str(tmp),str(path))
def main():
 mh=hashlib.sha256((E/'manifest.json').read_bytes()).hexdigest()
 rows=[json.loads(l) for l in (E/'public_all.jsonl').read_text().splitlines() if l.strip()]
 public={r['dataset']+':'+str(r['id']):r for r in rows}
 assert len(public)==7438==len(rows)
 folder=E/'completion';folder.mkdir(exist_ok=True)
 totals={};errors=[];allgood=True
 outpath=folder/'BLIND_GENERATED_TEXTS.jsonl.gz'
 with gzip.open(str(outpath)+'.tmp','wt',encoding='utf-8') as out:
  for model in ['openvl','rethinker']:
   groups=collections.defaultdict(lambda:collections.defaultdict(lambda:dict(n=0,seconds=0.0,tokens=0,truncated=0,main_forwards=0,weak_forwards=0)))
   present=set();totalbytes=0
   for file in sorted((E/'results'/model).glob('*.json.gz')):
    with gzip.open(str(file),'rt',encoding='utf-8') as f:x=json.load(f)
    key=x['sample_key'];assert key in public and key not in present
    assert x['model']==model and x['accepted'] is True and x['labels_read'] is False
    assert x['manifest_sha256']==mh
    assert file.name==hashlib.sha256(key.encode()).hexdigest()[:20]+'.json.gz'
    assert x['sample']==public[key]
    assert set(x['arms'])=={'full_lead','vcd_continuous'}
    for arm,value in x['arms'].items():
     ids=value['generated_token_ids'];assert ids and len(ids)==value['generated_tokens']
     assert isinstance(value['text'],str)
     g=groups[x['sample']['dataset']][arm]
     g['n']+=1;g['seconds']+=value['wall_seconds'];g['tokens']+=len(ids);g['truncated']+=bool(value['truncated'])
     fc=value.get('forward_calls',{})
     g['main_forwards']+=int(fc.get('main',0));g['weak_forwards']+=int(fc.get('weak',0))
     out.write(json.dumps(dict(model=model,arm=arm,dataset=x['sample']['dataset'],id=str(x['sample']['id']),text=value['text'],truncated=value['truncated'],stop_reason=value['stop_reason'],source=str(file),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest()),ensure_ascii=False)+'\n')
    present.add(key);totalbytes+=file.stat().st_size
   missing=sorted(set(public)-present);allgood=allgood and not missing
   for dataset,arms in groups.items():
    for arm,g in arms.items():
     g['mean_recorded_seconds']=g['seconds']/g['n'];g['mean_tokens']=g['tokens']/g['n']
   totals[model]=dict(completed_pairs=len(present),method_answers=2*len(present),expected_pairs=7438,missing=missing,compressed_bytes=totalbytes,by_dataset=groups)
 os.replace(str(outpath)+'.tmp',str(outpath))
 summary=dict(utc=now(),complete=allgood,manifest_sha256=mh,models=totals,total_expected_method_answers=29752,labels_read=False,scored=False,runtime_note='Recorded generation wall time; not a separate matched warmup efficiency benchmark',blind_texts=str(outpath))
 write(folder/'GENERATION_SUMMARY.json',summary)
 write(folder/'JUDGE_PENDING.json',dict(utc=now(),api_calls=0,mmhal_per_model_per_method=96,bingo_generated_per_model_per_method=586,bingo_official_reference_per_model_per_method=528,reason='Judge model/interface not configured; raw generation retained without fabricated scores'))
 print(json.dumps(dict(complete=allgood,per_model={k:v['completed_pairs'] for k,v in totals.items()},summary=str(folder/'GENERATION_SUMMARY.json')),ensure_ascii=False))
 if not allgood:raise RuntimeError('Incomplete generation, inspect missing keys')
if __name__=='__main__':main()
