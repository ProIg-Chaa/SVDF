"""Fixed mechanical continuation; no tuning, regeneration, or scientific interpretation."""
import datetime,fcntl,getpass,gzip,hashlib,json,os,pathlib,re,shlex,socket,subprocess,time,traceback
E=pathlib.Path('/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_vista_natural_trajectory_rank');R=E.parent.parent;A=E/'automation';V=E/'self_audit/verify_reference_subgroup'
MH='f33f2b952da918c8a8c6ce8d07294d723a3ae857299e7d2d6e73457eb91ea3bc';IH='f1b9813c20c1689133105d31628a3c93f9a5913b4f2f8bf4991504132441e844';VH='7cade5ec5c6b7fcda49f43b05038ed335fc8da6a4cc3a6cdbb3709cd8a9607b1';AH='0618bd509693888ffe621a42ad0d6f4037ec635f805de40a8e93976e68cd35f4'
ARMS=('cot','once','svdf','pai');END=time.monotonic()+12*3600
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else pathlib.Path(p).open()) as f:return json.load(f)
def write(p,obj,exclusive=False):
 with pathlib.Path(p).open('x' if exclusive else 'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def note(state,**kw):
 d=dict(utc=now(),hostname=socket.gethostname(),pid=os.getpid(),state=state,mechanical_only=True,**kw)
 with (A/'events.jsonl').open('a')as f:f.write(json.dumps(d)+'\n')
 write(A/'STATE.tmp',d);os.replace(str(A/'STATE.tmp'),str(A/'STATE.json'));print(json.dumps(d),flush=True)
def cmd(argv,timeout=60):
 p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 if p.returncode:raise RuntimeError((argv,p.returncode,p.stderr[-2000:]))
 return p.stdout
def frozen():
 assert sha(E/'manifest.json')==MH and sha(E/'input/facts_and_trajectories.jsonl')==IH
 assert sha(V/'verify_smoke.py')==VH and sha(E/'analyze_natural.py')==AH
 m=load(E/'manifest.json');assert sha(E/'input/INPUT_FREEZE.json')==m['input_freeze_sha256']
 for n,h in m['code_sha256'].items():assert sha(E/n)==h,n
 for n,h in m['source_files'].items():assert sha(E/'source'/n)==h,n
 for n,h in m['checkpoint_files'].items():assert sha(pathlib.Path(m['checkpoint']['path'])/n)==h,n
 return m
def wait_jobs(job,lanes=None):
 expected={job} if lanes is None else {job+'_'+str(i)for i in range(lanes)};last=None
 while time.monotonic()<END:
  rows={}
  for line in cmd(['sacct','-n','-P','-j',job,'--format=JobID%80,State,ExitCode']).splitlines():
   p=[v.strip() for v in line.split('|')]
   if len(p)>=3 and p[0] in expected:rows[p[0]]=(p[1].split()[0].rstrip('+'),p[2])
  if rows!=last:note('waiting_job',job_id=job,accounting=rows);last=rows
  for k,(s,ec)in rows.items():
   if s not in {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','COMPLETED'}:raise RuntimeError(('job_failed',k,s,ec))
   if s=='COMPLETED' and ec!='0:0':raise RuntimeError(('job_exit',k,ec))
  if set(rows)==expected and all(v==('COMPLETED','0:0')for v in rows.values()):return rows
  time.sleep(45)
 raise TimeoutError('controller12h deadline')
def validate_outputs(ids,smoke=False):
 cases={x['case_id']:x for x in map(json.loads,(E/'input/facts_and_trajectories.jsonl').read_text().splitlines())};found=[]
 for cid in ids:
  for arm in ARMS:
   p=E/'output/completed'/cid/(arm+'.json.gz');d=load(p);g=d['engineering'];ref=cases[cid]['trajectories'][arm]
   assert d['case_id']==cid and d['arm']==arm and d['manifest_sha256']==MH and d['replay_verified'] is True
   assert d['own_natural_trajectory'] is True and d['shared_prefix'] is False
   assert d['n_tokens']==len(ref['token_ids']) and d['token_ids_sha256']==hashlib.sha256(json.dumps(ref['token_ids']).encode()).hexdigest()
   for k in ('full_historical_tokens_exact','no_forced_tokens','final_layer_exact','final_norm_once','hook_cleanup_exact','prefill_last_query_only'):assert g[k] is True,(cid,arm,k)
   if arm in ('once','svdf'):assert g['actual_inputs_exact'] is True
   assert len(d['records'])==len(d['points'])*28
   if smoke:assert g['raw_smoke_cpu_recount_passed'] is True and g['dual_cpu_gpu_check'] is True
   found.append(dict(case_id=cid,arm=arm,path=str(p),sha256=sha(p)))
 return found
def cpu_script(script):
 shell='module load singularity/3.9.2-with-suid; exec singularity exec --bind '+shlex.quote(str(R)+':'+str(R))+' --pwd '+shlex.quote(str(E))+' '+shlex.quote(str(R/'containers/nvidia-pytorch-24.01-py3-arm64.sif'))+' env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python '+shlex.quote(str(script))
 p=subprocess.run(['bash','-lc',shell],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=1800)
 return dict(utc=now(),command=shell,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def independent():
 note('checking_independent_cpu_verifier')
 def reports():
  found=sorted(V.glob('SMOKE_INDEPENDENT_*.json'))
  for p in found:
   v=load(p);assert v['passed'] is True and v['manifest_sha256']==MH and v['input_sha256']==IH and v['n_case_arms']==8
   assert v['smoke_gate_sha256']==sha(E/'output/SMOKE_PASS_0.json')
  return found
 existing=reports()
 if not existing:
  frozen();write(A/'FALLBACK_VERIFY_INTENT.json',dict(utc=now(),verifier_sha256=VH,reason='smoke passed; no existing independent report; root authorizes immediate same frozen CPU checker'),True)
  r=cpu_script(V/'verify_smoke.py');write(A/'FALLBACK_VERIFY_RESULT.json',r,True);assert r['returncode']==0,r
  existing=reports()
 assert existing,'independent checker produced no receipt'
 return existing[0]
def project_empty():
 while time.monotonic()<END:
  busy=[]
  for line in cmd(['squeue','--me','--array','-h','-o','%i|%T|%R']).splitlines():
   jid,state,reason=line.split('|',2);info=cmd(['scontrol','show','job',jid,'-o']);q=re.search(r'\bCommand=(\S+)',info)
   if not q or not q.group(1).startswith(str(R/'experiments')+'/'):continue
   if jid=='1502691' and reason.strip('()')=='DependencyNeverSatisfied':continue
   if re.search(r'(?:gres/gpu|gres:gpu|gpu[:=])',info):busy.append(dict(job_id=jid,state=state,reason=reason))
  if not busy:note('project_capacity_clear',preserved_impossible_dependency='1502691');return
  note('waiting_project_capacity',jobs=busy);time.sleep(45)
 raise TimeoutError('capacity')
def main():
 A.mkdir(exist_ok=True);lock=(A/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a=load(E/'AUTOMATION_APPROVAL.json');assert a['manifest_sha256']==MH and a['controller_sha256']==sha(__file__)
 frozen();note('started',smoke_job='1566090',full_submitted=False);wait_jobs('1566090');frozen()
 g=load(E/'output/SMOKE_PASS_0.json');assert g['passed'] and g['manifest_sha256']==MH
 expected={(c,x)for c in ('M002','M078')for x in ARMS};assert len(g['checks'])==8 and {(x['case_id'],x['arm'])for x in g['checks']}==expected
 found=validate_outputs(('M002','M078'),True)
 for x in found:assert next(v['output_sha256']for v in g['checks']if(v['case_id'],v['arm'])==(x['case_id'],x['arm']))==x['sha256']
 p=independent()
 if (A/'SMOKE_ACCEPTED.json').exists():
  prior=load(A/'SMOKE_ACCEPTED.json');assert prior['manifest_sha256']==MH and prior['outputs']==found
 else:write(A/'SMOKE_ACCEPTED.json',dict(utc=now(),manifest_sha256=MH,independent_report=str(p),independent_sha256=sha(p),outputs=found),True)
 project_empty();frozen();import shutil;free=shutil.disk_usage(str(E)).free;assert free>=400*1024*1024,('filesystem_free',free)
 argv=['sbatch','--parsable','--array=0-3%4','--time=02:00:00',str(E/'worker.sbatch'),'full'];write(A/'FULL_SUBMISSION_INTENT.json',dict(utc=now(),command=argv,manifest_sha256=MH,free_bytes=free,uncertain_response='neverresubmit'),True)
 raw=cmd(argv);job=raw.strip().split(';')[0];assert job.isdigit(),raw
 write(A/'FULL_SUBMITTED.json',dict(utc=now(),job_id=job,command=argv,manifest_sha256=MH,raw_response=raw),True);note('full_submitted',job_id=job,lanes=4)
 accounting=wait_jobs(job,4);frozen();ids=[json.loads(l)['case_id']for l in(E/'input/facts_and_trajectories.jsonl').read_text().splitlines()];assert len(ids)==208
 outputs=validate_outputs(ids);assert len(outputs)==832;write(A/'FULL_ACCEPTED.json',dict(utc=now(),job_id=job,accounting=accounting,n_outputs=832,manifest_sha256=MH,outputs=outputs),True)
 frozen();note('analyzing_complete_832',job_id=job,analyzer_sha256=AH);r=cpu_script(E/'analyze_natural.py');write(A/'ANALYSIS_RECEIPT.json',r,True);assert r['returncode']==0,r;note('completed',job_id=job,n_outputs=832)
if __name__=='__main__':
 try:main()
 except Exception as ex:
  A.mkdir(exist_ok=True);note('stopped_failure',error=repr(ex),traceback=traceback.format_exc(),action='no tuning, no rerun, no further GPU submission');raise
