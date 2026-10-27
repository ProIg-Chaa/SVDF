# CPU-only profile of the already accepted two fixed smoke cases.
from common import *
import datetime

def main():
 m,mh=check_manifest();profiles=[]
 for spec in m['smoke_cases']:
  p=next(p for p in rows(E/'positions.jsonl') if str(p['id'])==str(spec['id']) and p['position_id']==spec['position_id']);folder=O/'completed'/stem(p);ac=json.loads((folder/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh
  for n,h in ac['files_sha256'].items():assert sha(folder/n)==h
  row=json.loads((folder/'probes.json').read_text());caps=load_tensors(folder/'captures_and_updates.pt.gz');raw=load_tensors(folder/'raw_logits.pt.gz')
  main_seconds=sum(r['seconds'] for arm in row['measurements'] for r in arm['readouts'])+sum(r['seconds'] for r in row['weak_hard_readouts']);capture_seconds=sum(cap[image]['audit']['seconds'] for cap in caps['captures'].values() for image in ('original','weakened'))
  tensors=[r['raw_logits'] for name in ('1','4') for arm in raw[name].values() for r in arm]+[r['raw_logits'] for r in raw['weak_hard']]
  profiles.append(dict(id=str(p['id']),position_id=p['position_id'],main_readout_seconds=main_seconds,capture_seconds=capture_seconds,main_and_capture_seconds=main_seconds+capture_seconds,actual_full_readout_uncompressed_bytes=sum(t.numel()*t.element_size() for t in tensors),candidate_raw_tensor_count=len(tensors),sizes_bytes={n:(folder/n).stat().st_size for n in ac['files_sha256']},acceptance_sha256=sha(folder/'ACCEPTANCE.json')))
 x=dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),generation_manifest_sha256=mh,n=2,profiles=profiles,only_accepted_smoke_data=True)
 print(json.dumps(x),flush=True)
if __name__=='__main__':main()
