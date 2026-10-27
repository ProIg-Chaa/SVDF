import json,hashlib,subprocess,time,os,datetime,sys
from pathlib import Path
E=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(value):
 p=E/'automation'/'BOOTSTRAP_STATE.json';p.parent.mkdir(exist_ok=True)
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(dict(utc=now(),**value),indent=2));os.replace(str(t),str(p))
def active(job):
 p=subprocess.run(['squeue','-h','-j',job,'-o','%i|%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode:raise RuntimeError(p.stderr.decode())
 return bool(p.stdout.decode().strip())
def main():
 smoke,full=sys.argv[1:3]
 write(dict(state='WAITING_ENGINEERING',smoke_job=smoke,full_job=full,pid=os.getpid()))
 while active(smoke):time.sleep(30)
 mh=hashlib.sha256((E/'manifest.json').read_bytes()).hexdigest()
 gates=[]
 for model in ['openvl','rethinker']:
  p=E/'smoke'/model/'PASS.json'
  if p.exists():
   x=json.loads(p.read_text())
   if x.get('passed') and x.get('manifest_sha256')==mh:gates.append(model)
 if len(gates)!=2:
  subprocess.run(['scancel',full],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  write(dict(state='STOPPED_ENGINEERING_FAILED',smoke_job=smoke,dependent_full_job_cancelled=full,passed_models=gates));return
 write(dict(state='ENGINEERING_PASSED',smoke_job=smoke,full_job=full,passed_models=gates))
 os.execv(sys.executable,[sys.executable,'-u',str(E/'continuation_controller.py'),'--initial-job',full])
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(dict(state='BOOTSTRAP_ERROR',error=repr(e)));raise
