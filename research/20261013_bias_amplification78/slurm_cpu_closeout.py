"""Deterministic CPU closeout within the last completing Slurm lane; not scientific review."""
from common import *
import datetime,subprocess,socket,traceback

def log(status,**kw):
 x=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),lane=os.environ.get('SLURM_ARRAY_TASK_ID'),mechanical_only=True,**kw)
 with (O/'slurm_closeout_events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 (O/'SLURM_CLOSEOUT_STATE.json').write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)

def main():
 cm=json.loads((E/'closeout_manifest.json').read_text())
 for n,h in cm['code_sha256'].items():assert sha(E/n)==h,n
 m,mh=check_manifest();pm=json.loads((E/'probe_manifest.json').read_text());ph=sha(E/'probe_manifest.json');assert pm['base_manifest_sha256']==mh
 for n,h in pm['code_sha256'].items():assert sha(E/n)==h,n
 reg=[x for x in rows(E/'probe_registry.jsonl') if x['eligible']];assert len(reg)==pm['eligible_count']==52
 for pr in reg:
  p=O/'probes/completed'/stem(pr)
  if not p.exists():
   print(json.dumps(dict(status='this_lane_done_other_lane_pending',expected=52)),flush=True);return 0
  ac=json.loads((p/'ACCEPTANCE.json').read_text());assert ac['passed'] and ac['probe_manifest_sha256']==ph and ac['probe_sha256']==sha(p/'probes.json') and ac['raw_logits_sha256']==sha(p/'raw_logits.pt')
 try:(O/'slurm_cpu_closeout.lock').mkdir()
 except FileExistsError:
  print('CPU_CLOSEOUT_OWNED_BY_OTHER_LANE',flush=True);return 0
 log('recounting_raw_logits',complete=52)
 r=subprocess.run([sys.executable,str(E/'verify_probe_logits.py')],text=True,capture_output=True);(O/'slurm_cpu_recount.stdout').write_text(r.stdout);(O/'slurm_cpu_recount.stderr').write_text(r.stderr);assert r.returncode==0,('independent CPU recount failed',r.returncode)
 r=subprocess.run([sys.executable,str(E/'summarize_probes.py')],text=True,capture_output=True);(O/'slurm_summary.stdout').write_text(r.stdout);(O/'slurm_summary.stderr').write_text(r.stderr);assert r.returncode==0,('summary failed',r.returncode)
 x=json.loads((E/'evaluation/PROBE_SUMMARY.json').read_text());assert x['n']==52 and x['probe_manifest_sha256']==ph
 log('completed',n=52,probe_manifest_sha256=ph,summary=str(E/'evaluation/PROBE_SUMMARY.json'),summary_sha256=sha(E/'evaluation/PROBE_SUMMARY.json'),scientific_interpretation_pending=True)
 return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:
  log('failed_stop',error=str(e),error_type=type(e).__name__,traceback=traceback.format_exc());raise
