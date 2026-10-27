"""Status polling, fixed frozen timeout continuation, and blind collection only."""
import datetime,hashlib,json,os,re,subprocess,time
from pathlib import Path
from collect_blind import expected,collect,sha
R=Path('/home/bingxing2/home/scx9ftv/gushuo')
E=R/'experiments/20261019_r1_ablation_complete/generation'
A=E.parent/'automation';O=R/'output/experiments/20261019_r1_ablation_complete'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def append(v):
 with (A/'mechanical_events.jsonl').open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
def cmd(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 if p.returncode:raise RuntimeError((args,p.returncode,p.stderr))
 return p.stdout
def progress():
 counts={}
 for method in ('true_only','weak_only','neg_original_soft','reverse','random_104729','dynamic_f4','start16','start32'):
  d=O/'completed'/method;counts[method]=len(list(d.iterdir())) if d.exists() else 0
 return counts
def resource():
 # Expanded arrays include pending potential allocations; dead CPU/N/A dependency consumes zero.
 raw=cmd(['squeue','-h','-r','-u','scx9ftv','-o','%i|%T|%b|%o']);project=account=0;unknown=[]
 for l in raw.splitlines():
  a=l.split('|',3)
  if len(a)!=4:continue
  job,state,tres,command=a
  vals=re.findall(r'(?:gres:)?gpu(?::[^:,| ]+)?:(\d+)',tres);g=sum(map(int,vals))
  account+=g
  if str(R/'experiments') in command:project+=g
  elif g and not any(x in command for x in ('workspace/projects/rasp-lrm','/rasp-lrm/')):unknown.append(l)
 return dict(project_gpus=project,account_gpus=account,unknown_gpu_jobs=unknown,raw=raw)
def main():
 import fcntl
 lock=(A/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 approval=json.loads((A/'ACTIVATION.json').read_text());manifest=json.loads((A/'AUTOMATION_MANIFEST.json').read_text())
 assert approval['approved'] and approval['automation_manifest_sha256']==sha(A/'AUTOMATION_MANIFEST.json')
 assert sha(E/'manifest.json')==manifest['generation_manifest_sha256']
 for n,h in manifest['files'].items():assert sha(A/n)==h,n
 for p,h in manifest['external_dependencies'].items():assert sha(p)==h,p
 full=json.loads((E/'FULL_SUBMITTED.json').read_text());job=full['job_id'];retries=0
 if (A/'CONTINUATION_STATE.json').exists():
  state=json.loads((A/'CONTINUATION_STATE.json').read_text());job=state['job_id'];retries=state['resubmissions']
 deadline=time.monotonic()+72*3600
 append(dict(utc=now(),event='controller_started',job_id=job,role='mechanical_only_no_scientific_review',labels_read=False))
 while time.monotonic()<deadline:
  counts=progress();done=sum(counts.values());res=resource();record=dict(utc=now(),job_id=job,accepted_fresh=done,expected_fresh=5282,counts=counts,resources=res)
  (A/'CURRENT_STATUS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');append(dict(event='status',**record))
  if done==5282:
   result=collect();append(dict(utc=now(),event='blind_collection_complete',result=result));return
  states=cmd(['sacct','-n','-X','-j',str(job),'--format=State','-P']).strip().splitlines()
  active=any(any(x in s for x in ('RUNNING','PENDING','CONFIGURING','COMPLETING')) for s in states)
  if states and not active:
   allowed=all(any(x in s for x in ('COMPLETED','TIMEOUT','PREEMPTED','NODE_FAIL')) for s in states)
   retryable=any(any(x in s for x in ('TIMEOUT','PREEMPTED','NODE_FAIL')) for s in states)
   if not allowed or not retryable or retries>=1:
    append(dict(utc=now(),event='stopped_incomplete_requires_review',states=states,accepted=done,resubmissions=retries));return
   if res['project_gpus']+3<=4 and res['account_gpus']+3<=8 and not res['unknown_gpu_jobs']:
    assert sha(E/'manifest.json')==manifest['generation_manifest_sha256']
    permit=json.loads((E/'FULL_APPROVAL.json').read_text());assert permit['approved'] and permit['manifest_sha256']==sha(E/'manifest.json')
    job=cmd(['sbatch','--parsable',str(E/'worker.sbatch'),'full']).strip().split(';')[0];retries+=1
    (A/'CONTINUATION_STATE.json').write_text(json.dumps(dict(job_id=job,resubmissions=retries,utc=now()),indent=2)+'\n')
    append(dict(utc=now(),event='same_frozen_unfinished_only_resubmitted',job_id=job,resubmissions=retries,resources_before=res))
  time.sleep(120)
 append(dict(utc=now(),event='mechanical_watch_deadline_reached'))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  append(dict(utc=now(),event='controller_error_stop',error=repr(exc)));raise
