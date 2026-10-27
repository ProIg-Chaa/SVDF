"""Only a root-authorized fixed pipeline. No scientific decisions, tuning, or retries."""
import argparse,datetime,fcntl,gzip,hashlib,json,math,os,re,shlex,shutil,socket,subprocess,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent.parent;A=E/'automation';ARMS=('cot','once','svdf','pai')
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with(gzip.open(p,'rt')if str(p).endswith('.gz')else Path(p).open())as f:return json.load(f)
def save(p,d):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x')as f:json.dump(d,f,indent=2)
def note(state,**kw):
 d=dict(utc=now(),pid=os.getpid(),hostname=socket.gethostname(),state=state,mechanical_only=True,**kw)
 with(A/'events.jsonl').open('a')as f:f.write(json.dumps(d)+'\n')
 (A/'STATE.tmp').write_text(json.dumps(d,indent=2));os.replace(str(A/'STATE.tmp'),str(A/'STATE.json'));print(json.dumps(d),flush=True)
def call(args,timeout=60):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 if p.returncode:raise RuntimeError((args,p.returncode,p.stderr[-3000:]))
 return p.stdout
def frozen():
 auth=load(E/'ROOT_CONDITIONAL_APPROVAL.json');assert auth['passed']is True
 assert sha(E/'GENERATION_MANIFEST.json')==auth['generation_manifest_sha256']
 for name,h in auth['pipeline_files'].items():assert sha(E/name)==h,name
 m=load(E/'GENERATION_MANIFEST.json')
 for name,h in m['files'].items():assert sha(E/name)==h,name
 assert sha(E/'input/public.jsonl')==m['public_sha256']
 for name,h in m['checkpoint_files'].items():assert sha(Path(m['checkpoint']['path'])/name)==h,name
 return auth,m

def wait_job(job,lanes=None):
 expected={job}if lanes is None else{job+'_'+str(i)for i in range(lanes)};last=None
 while True:
  rows={}
  for line in call(['sacct','-n','-P','-j',job,'--format=JobID%80,State,ExitCode']).splitlines():
   p=[x.strip()for x in line.split('|')]
   if len(p)>=3 and p[0]in expected:rows[p[0]]=(p[1].split()[0].rstrip('+'),p[2])
  if rows!=last:note('waiting_job',job_id=job,accounting=rows);last=rows
  for j,(s,ec)in rows.items():
   assert s in{'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','COMPLETED'},(j,s,ec)
   if s=='COMPLETED':assert ec=='0:0',(j,ec)
  if set(rows)==expected and all(x==('COMPLETED','0:0')for x in rows.values()):return rows
  time.sleep(45)
def capacity():
 while True:
  busy=[]
  for line in call(['squeue','--me','--array','-h','-o','%i|%T|%R']).splitlines():
   jid,state,reason=line.split('|',2);info=call(['scontrol','show','job',jid,'-o']);m=re.search(r'\bCommand=(\S+)',info)
   if not m or not m.group(1).startswith(str(R/'experiments')+'/'):continue
   if jid=='1502691'and reason.strip('()')=='DependencyNeverSatisfied':continue
   if re.search(r'(?:gres/gpu|gres:gpu|gpu[:=])',info):busy.append(dict(id=jid,state=state,reason=reason))
  if not busy:
   free=shutil.disk_usage(str(E)).free;assert free>=250*1024*1024,('insufficient_space_stop',free);return free
  note('waiting_project_capacity',busy=busy);time.sleep(45)
def submit(phase,seconds,full):
 frozen();free=capacity();minutes=max(30,int(math.ceil(seconds/60)));h,mi=divmod(minutes,60);wall='%02d:%02d:00'%(h,mi)
 argv=['sbatch','--parsable','--time='+wall]
 if full:argv+=['--array=0-3%4']
 argv+=[str(E/'worker.sbatch'),phase]
 save(A/(phase+'_SUBMISSION_INTENT.json'),dict(utc=now(),phase=phase,command=argv,free_bytes=free,policy='unique submission; no retry if response uncertain'))
 raw=call(argv);job=raw.strip().split(';')[0];assert job.isdigit(),raw
 save(A/(phase+'_SUBMITTED.json'),dict(utc=now(),phase=phase,job_id=job,raw=raw,command=argv));note('submitted',phase=phase,job_id=job,time_limit=wall,gpus=4 if full else 1);return job

def outputs(stage,ids,mh,rh=None):
 records=[]
 for cid in ids:
  for arm in ARMS:
   p=E/'output'/stage/'completed'/cid/(arm+'.json.gz');x=load(p)
   assert x['case_id']==cid and x['arm']==arm and x['generation_manifest_sha256']==mh
   assert len(x['generated_token_ids'])==x['n_tokens']and len(x['actual_input_hashes']['main'])==x['n_tokens']
   if stage=='replay':
    assert x['replay_manifest_sha256']==rh
    if x.get('replay_status')=='no_eligible_candidate':assert x['replay_verified']is None and not x['records']
    else:
     assert x['replay_verified']is True and x['engineering']['actual_inputs_exact']and x['engineering']['no_forced_tokens']
     assert len(x['records'])==len(x['points'])*28
   records.append(dict(case_id=cid,arm=arm,path=str(p),sha256=sha(p),seconds=x['seconds'],status=x.get('replay_status','complete')))
 return records

def full_seconds(records):
 per={}
 for r in records:per[r['case_id']]=per.get(r['case_id'],0.)+r['seconds']
 # Fixed conservative wall clock from the slower smoke image; no method parameters change.
 return max(3600,1.8*max(per.values())*125+900)
def cpu(script,subcommand,receipt):
 frozen();shell='module load singularity/3.9.2-with-suid; exec singularity exec --bind '+shlex.quote(str(R)+':'+str(R))+' --pwd '+shlex.quote(str(E))+' '+shlex.quote(str(R/'containers/nvidia-pytorch-24.01-py3-arm64.sif'))+' env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(E/'source')+':'+str(R/'container_python/py310'))+' python '+shlex.quote(str(E/script))
 if subcommand:shell+=' '+shlex.quote(subcommand)
 p=subprocess.run(['bash','-lc',shell],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=7200)
 save(A/receipt,dict(utc=now(),script=script,script_sha256=sha(E/script),command=shell,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,(script,p.stderr[-3000:]);return p.stdout

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke-job',required=True);args=ap.parse_args();A.mkdir(exist_ok=True)
 lock=(A/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 auth,m=frozen();mh=sha(E/'GENERATION_MANIFEST.json');ids=[json.loads(l)['case_id']for l in(E/'input/public.jsonl').read_text().splitlines()];assert len(ids)==500 and len(set(ids))==500
 note('started',generation_smoke_job=args.smoke_job);wait_job(args.smoke_job);frozen()
 gate=load(E/'output/generation/SMOKE_PASS.json');assert gate['passed']and gate['generation_manifest_sha256']==mh and gate['active_generation_smoke']
 smoke=outputs('generation',m['smoke_case_ids'],mh);assert len(smoke)==8
 for r in smoke:assert load(r['path'])['engineering_repeat_exact']is True
 save(A/'GENERATION_SMOKE_ACCEPTED.json',dict(utc=now(),job_id=args.smoke_job,gate_sha256=sha(E/'output/generation/SMOKE_PASS.json'),outputs=smoke))
 job=submit('generation_full',full_seconds(smoke),True);account=wait_job(job,4);frozen();allgen=outputs('generation',ids,mh);assert len(allgen)==2000
 save(A/'GENERATION_FULL_ACCEPTED.json',dict(utc=now(),job_id=job,accounting=account,n=2000,outputs=allgen));note('labeling_complete_generation',n=2000)
 cpu('label_and_analyze.py','labels','LABELS_RECEIPT.json')
 candidates=E/'labels/candidates.jsonl';freeze=E/'labels/LABEL_FREEZE.json';assert freeze.exists()
 rows=[json.loads(l)for l in candidates.read_text().splitlines()];assert len(rows)==500 and{r['case_id']for r in rows}==set(ids)
 rm=dict(utc=now(),generation_manifest_sha256=mh,oracle_path='labels/candidates.jsonl',oracle_sha256=sha(candidates),label_freeze_sha256=sha(freeze),labeler_sha256=sha(E/'label_and_analyze.py'),root_conditional_approval_sha256=sha(E/'ROOT_CONDITIONAL_APPROVAL.json'),primary='CoT-frozen three object categories; same candidates in four native trajectories',readouts='8equidistant positions per early/middle/late thirds,28layers,finalRMSNormonce,rank1+strictlygreater',no_eligible_candidate='keep generation; explicit skip rank; no replacement')
 save(E/'REPLAY_MANIFEST.json',rm);rh=sha(E/'REPLAY_MANIFEST.json');save(E/'ROOT_REPLAY_APPROVAL.json',dict(passed=True,utc=now(),manifest_sha256=rh,authorized_by='previous explicit root conditional approval',root_conditional_approval_sha256=sha(E/'ROOT_CONDITIONAL_APPROVAL.json'),mechanically_frozen_labels=True))
 job=submit('replay_smoke',3600,False);wait_job(job);frozen();rg=load(E/'output/replay/SMOKE_PASS.json');assert rg['passed']and rg['replay_manifest_sha256']==rh
 small=outputs('replay',m['smoke_case_ids'],mh,rh);assert len(small)==8
 cpu('verify_replay_smoke.py',None,'REPLAY_CPU_VERIFY_RECEIPT.json')
 verify=sorted((E/'self_audit/contrast_mechanism_geometry').glob('REPLAY_SMOKE_INDEPENDENT_*.json'));assert len(verify)==1
 vr=load(verify[0]);assert vr['passed']and vr['generation_manifest_sha256']==mh and vr['replay_manifest_sha256']==rh
 save(A/'REPLAY_SMOKE_ACCEPTED.json',dict(utc=now(),job_id=job,independent_path=str(verify[0]),independent_sha256=sha(verify[0]),outputs=small))
 job=submit('replay_full',full_seconds(small),True);account=wait_job(job,4);frozen();out=outputs('replay',ids,mh,rh);assert len(out)==2000
 save(A/'REPLAY_FULL_ACCEPTED.json',dict(utc=now(),job_id=job,n=2000,accounting=account,outputs=out));note('analyzing_complete_replay',n=2000)
 cpu('label_and_analyze.py','analyze','ANALYSIS_RECEIPT.json');note('complete',n_images=500,n_arms=2000)
if __name__=='__main__':
 try:main()
 except Exception as ex:
  A.mkdir(exist_ok=True);note('stopped_failure',error=repr(ex),traceback=traceback.format_exc(),action='no retry/tuning/new methods/deletion');raise
