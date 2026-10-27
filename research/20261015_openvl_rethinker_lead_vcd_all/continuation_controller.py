import argparse,datetime,json,os,re,subprocess,time,fcntl,hashlib,socket
from pathlib import Path
E=Path(__file__).resolve().parent
R=E.parent.parent
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):
 with open(str(p)) as f: return json.load(f)
def save(name,x):
 p=E/'automation'/name;p.parent.mkdir(exist_ok=True)
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(str(t),str(p))
def log(event,**kw):
 row=dict(utc=now(),event=event,**kw)
 with (E/'automation'/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)
def queue():
 p=subprocess.run(['squeue','-u',os.environ.get('USER','scx9ftv'),'-h','-o','%i|%T|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode:raise RuntimeError(p.stderr.decode())
 return [l.split('|',2) for l in p.stdout.decode().splitlines() if l.strip()]
def project_free():
 total=0;details=[]
 for jid,state,reason in queue():
  if reason in ('DependencyNeverSatisfied',):continue
  p=subprocess.run(['scontrol','show','job','-o',jid],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  if p.returncode:continue
  text=p.stdout.decode();m=re.search(r'\bCommand=(\S+)',text)
  if not m or not m.group(1).startswith(str(R/'experiments')+'/'):continue
  g=re.search(r'\bReqTRES=\S*?gres/gpu=(\d+)',text)
  if not g:g=re.search(r'gres/gpu:(\d+)',text)
  n=int(g.group(1)) if g else 0
  total+=n;details.append(dict(job=jid,state=state,gpus=n))
 return max(0,4-total),details
def validate_fixed():
 h=hashlib.sha256((E/'manifest.json').read_bytes()).hexdigest()
 approval=load(E/'ROOT_APPROVAL.json')
 assert approval.get('manifest_sha256')==h and approval.get('full_authorized') is True
 for model in ['openvl','rethinker']:
  p=load(E/'smoke'/model/'PASS.json')
  assert p.get('passed') and p.get('manifest_sha256')==h
 return h
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--initial-job',required=True);a=ap.parse_args()
 (E/'automation').mkdir(exist_ok=True)
 lock=(E/'automation'/'controller.lock').open('a')
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 mh=validate_fixed();current=a.initial_job;roundno=1
 save('CONTROLLER.json',dict(pid=os.getpid(),host=socket.gethostname(),started=now(),initial_job=current,manifest_sha256=mh,scientific_monitoring=False,scope='resource/completion checks; fixed protocol continuation only'))
 log('start',job=current)
 while True:
  save('STATE.json',dict(utc=now(),state='WAITING_FOR_GPU_JOB',job=current,round=roundno,manifest_sha256=mh))
  while any(jid==current or jid.startswith(current+'_') for jid,_,_ in queue()):time.sleep(45)
  time.sleep(5)
  ac=subprocess.run(['sacct','-n','-P','-j',current,'--format=JobIDRaw,State,ExitCode'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  records=[line.split('|') for line in ac.stdout.decode().splitlines() if line.strip()]
  failures=[dict(job=r[0],state=r[1],exit=r[2]) for r in records if len(r)>=3 and '.' not in r[0] and (r[1] not in ['COMPLETED'] or r[2]!='0:0')]
  if ac.returncode or failures or not records:
   save('STATE.json',dict(utc=now(),state='STOPPED_JOB_FAILURE_OR_UNKNOWN',job=current,failures=failures,sacct_stderr=ac.stderr.decode()))
   log('stopped_job_failure_or_unknown',failures=failures);return
  stat={};remaining=[];bad=[]
  for lane in range(4):
   p=E/'status'/('full_lane%d.json'%lane)
   if not p.exists():bad.append(dict(lane=lane,reason='missing status'));continue
   x=load(p);stat[str(lane)]=x
   if x.get('manifest_sha256')!=mh:bad.append(dict(lane=lane,reason='manifest mismatch'));continue
   st=x.get('state')
   if st=='COMPLETE':continue
   if st=='PAUSE' and x.get('reason') in ['time_budget','signal']:remaining.append(lane);continue
   bad.append(dict(lane=lane,state=st,reason=x.get('reason')))
  if bad:
   save('STATE.json',dict(utc=now(),state='STOPPED_REVIEW_REQUIRED',job=current,round=roundno,bad=bad,lane_status=stat,scientific_monitoring=False))
   log('stopped_review_required',bad=bad);return
  if not remaining:
   save('STATE.json',dict(utc=now(),state='COMPLETE',round=roundno,lane_status=stat,manifest_sha256=mh))
   log('all_complete')
   if (E/'completion_summary.py').exists():
    p=subprocess.run(['python3',str(E/'completion_summary.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    save('COLLECTOR_EXIT.json',dict(utc=now(),returncode=p.returncode,stdout=p.stdout.decode()[-8000:],stderr=p.stderr.decode()[-8000:]))
   return
  if roundno>=12:
   save('STATE.json',dict(utc=now(),state='STOPPED_ROUND_LIMIT',remaining=remaining,round=roundno));return
  validate_fixed()
  while True:
   v=os.statvfs(str(E));free=v.f_bavail*v.f_frsize
   if free<128*2**20:
    save('STATE.json',dict(utc=now(),state='STOPPED_STORAGE',free_bytes=free,remaining=remaining));return
   slots,other=project_free()
   if slots>0:break
   save('STATE.json',dict(utc=now(),state='WAITING_PROJECT_CAPACITY',remaining=remaining,other_jobs=other));time.sleep(45)
  chosen=remaining[:slots]
  # Existing COMPLETE records skip already finished lanes; no new scientific configuration.
  p=subprocess.run(['sbatch','--parsable','--array='+','.join(map(str,chosen))+'%4',str(E/'full.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  if p.returncode:raise RuntimeError(p.stderr.decode())
  current=p.stdout.decode().strip().split(';')[0];assert current.isdigit()
  roundno+=1;log('continued',job=current,round=roundno,lanes=chosen)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  save('STATE.json',dict(utc=now(),state='CONTROLLER_ERROR',error=repr(exc)))
  raise
