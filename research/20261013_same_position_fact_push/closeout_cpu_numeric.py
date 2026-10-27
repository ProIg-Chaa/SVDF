# Mechanical CPU verification inside the last completing allocation; no scientific interpretation.
from common import *
import subprocess,datetime,socket,traceback

def log(status,**kw):
 x=dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),lane=os.environ.get('SLURM_ARRAY_TASK_ID'),status=status,mechanical_only=True,**kw)
 with (O/'closeout_events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 (O/'CLOSEOUT_STATE.json').write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)
def main():
 m,mh=check_manifest();plans=rows(E/'positions.jsonl')
 for p in plans:
  folder=O/'completed'/stem(p)
  if not folder.exists():print('OTHER_LANE_PENDING',flush=True);return
  ac=json.loads((folder/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['manifest_sha256']==mh
  for n,h in ac['files_sha256'].items():assert sha(folder/n)==h
 try:(O/'cpu_recount_once.lock').mkdir()
 except FileExistsError:print('CPU_RECOUNT_OWNED',flush=True);return
 log('cpu_recount_running',n=len(plans));r=subprocess.run([sys.executable,str(E/'verify_logits_numeric.py'),'--scope','full'],capture_output=True,text=True);(O/'cpu_recount.stdout').write_text(r.stdout);(O/'cpu_recount.stderr').write_text(r.stderr);assert r.returncode==0,('CPU verification failed',r.returncode)
 latest=json.loads((E/'evaluation/full/latest.json').read_text());assert latest['complete'] and latest['n']==m['position_count'] and sha(latest['file'])==latest['sha256'];log('completed',n=latest['n'],receipt=latest,scientific_interpretation_pending=True)
if __name__=='__main__':
 try:main()
 except Exception as e:log('failed_stop',error=str(e),traceback=traceback.format_exc());raise
