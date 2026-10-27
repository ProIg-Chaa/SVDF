"""Mechanical completion relay only. No scientific supervision or new experiment selection."""
from common import *
import subprocess,datetime,traceback

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(status,**kw):
 row=dict(utc=utc(),status=status,mechanical_only=True,**kw)
 with (E/'output/mechanical_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 (E/'output/MECHANICAL_STATE.json').write_text(json.dumps(row,indent=2))
 print(json.dumps(row),flush=True)
def main():
 jobs=json.loads((E/'PROBE_FULL_SUBMITTED.json').read_text());smoke=jobs['dependency_smoke_job'];full=jobs['job_id'];pm=json.loads((E/'probe_manifest.json').read_text());ph=sha(E/'probe_manifest.json')
 log('waiting_for_generation',smoke_job=smoke,full_job=full,pid=os.getpid(),poll_seconds=60,scientific_review_continues=False)
 while True:
  r=subprocess.run(['sacct','-j',smoke+','+full,'--format=JobIDRaw,State%40,ExitCode','--noheader','--parsable2'],text=True,capture_output=True,check=True)
  bad=[];states=[]
  for line in r.stdout.splitlines():
   p=line.split('|')
   if len(p)<3:continue
   jid,state,exitcode=p[:3]
   if '.' in jid:continue
   states.append((jid,state,exitcode))
   if any(state.startswith(k) for k in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED']) or (state=='COMPLETED' and exitcode!='0:0'):bad.append((jid,state,exitcode))
  if bad:
   log('generation_failed_stop',failures=bad,states=states);return 2
  done=O/'probes/completed';complete=list(done.iterdir()) if done.exists() else []
  root_full=[x for x in states if x[0].startswith(full+'_') and '[' not in x[0]]
  if len(complete)==pm['eligible_count'] and len(root_full)==2 and all(x[1]=='COMPLETED' and x[2]=='0:0' for x in root_full):break
  time.sleep(60)
 for lane in range(2):
  receipt=json.loads((O/'probes'/('SMOKE_PASS_'+str(lane)+'.json')).read_text());assert receipt['passed'] and receipt['probe_manifest_sha256']==ph
 log('independent_cpu_recount',completed=len(complete),expected=pm['eligible_count'])
 shell='module load singularity/3.9.2-with-suid\nR='+str(R)+'\nE='+str(E)+'\nsingularity exec --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 python "$E/verify_probe_logits.py"'
 r=subprocess.run(['bash','-lc',shell],text=True,capture_output=True);(O/'cpu_recount.stdout').write_text(r.stdout);(O/'cpu_recount.stderr').write_text(r.stderr);assert r.returncode==0,('cpu recount failed',r.returncode)
 r=subprocess.run(['python3',str(E/'summarize_probes.py')],text=True,capture_output=True);(O/'summary.stdout').write_text(r.stdout);(O/'summary.stderr').write_text(r.stderr);assert r.returncode==0,('summary failed',r.returncode)
 log('completed',n=pm['eligible_count'],probe_manifest_sha256=ph,summary=str(E/'evaluation/PROBE_SUMMARY.json'),all_raw_logits_recounted=True,scientific_interpretation_pending=True)
 return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:
  log('mechanical_failure_stop',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
