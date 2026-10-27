# Mechanical continuation after root activation. Status collection is not a future scientific audit.
import argparse,datetime,fcntl,hashlib,json,math,os,pathlib,socket,subprocess,sys,time,traceback
E=pathlib.Path(__file__).resolve().parent;R=E.parent.parent;O=E/'output'
sys.path.insert(0,str(E))
from continuation_resource_gate import snapshot

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def jread(p):return json.loads(pathlib.Path(p).read_text())
def save_once(p,x):
 with pathlib.Path(p).open('x') as f:json.dump(x,f,indent=2)
def log(status,**kw):
 x=dict(actual_utc=now(),host=socket.gethostname(),pid=os.getpid(),status=status,mechanical_only=True,scientific_review_not_automated=True,**kw)
 with (O/'continuation_events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 (O/'CONTINUATION_STATE.json').write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)

def check_files(require_activation=True):
 import common
 m,mh=common.check_manifest();cm=jread(E/'continuation_manifest.json');ch=sha(E/'continuation_manifest.json')
 assert cm['generation_manifest_sha256']==mh and cm['smoke_jobid']=='1565544'
 for name,digest in cm['files_sha256'].items():assert sha(E/name)==digest,name
 am=jread(E/'evaluation/analysis_manifest.json');assert am['generation_manifest_sha256']==mh
 assert sha(E/'evaluation/analysis_manifest.json')==cm['analysis_manifest_sha256']
 for n,h in am['analysis_files_sha256'].items():assert sha(E/'evaluation'/n)==h,n
 pm=jread(E/'evaluation/paired_position_analysis_manifest.json');assert sha(E/'evaluation/paired_position_analysis_manifest.json')==cm['paired_analysis_manifest_sha256'] and pm['generation_manifest_sha256']==mh and pm['main_analysis_manifest_sha256']==cm['analysis_manifest_sha256']
 for n,h in pm['paired_files_sha256'].items():assert sha(E/'evaluation'/n)==h,n
 if require_activation:
  x=jread(E/'CONTINUATION_ACTIVATED.json');assert x['continuation_manifest_sha256']==ch and x['generation_manifest_sha256']==mh and x['root_review_passed']
 return m,mh,cm,ch

def state_of(job):
 p=subprocess.run(['sacct','-n','-P','-X','-j',str(job),'-o','JobID%40,State,ExitCode'],capture_output=True,text=True,timeout=60)
 if p.returncode:raise RuntimeError(('sacct',p.returncode,p.stderr))
 d={}
 for line in p.stdout.splitlines():
  z=line.split('|')
  if len(z)>=3:d[z[0].strip()]=dict(state=z[1].split()[0].replace('+',''),exitcode=z[2])
 return {str(lane):d.get(str(job)+'_'+str(lane)) for lane in (0,1)}

def terminal(states):
 active={'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','REQUEUED','RESIZING'}
 for z in states.values():
  if z is None:continue
  if z['state'] not in active and not (z['state']=='COMPLETED' and z['exitcode']=='0:0'):
   raise RuntimeError(('lane failed; preserve partial evidence, no automatic retry',states))
 if any(z is None for z in states.values()):return False
 return all(z['state']=='COMPLETED' and z['exitcode']=='0:0' for z in states.values())

def cpu_command(script,args,tag):
 # No --nv, no GPU model load; standard frozen post-processing inside existing container.
 command=['bash','-lc','module load singularity/3.9.2-with-suid; exec singularity exec --bind '+str(R)+':'+str(R)+' --pwd '+str(E/'source')+' '+str(R/'containers/nvidia-pytorch-24.01-py3-arm64.sif')+' env PYTHONPATH='+str(E)+':'+str(E/'source')+':'+str(R/'container_python/py310')+' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 python '+str(script)+' '+' '.join(args)]
 p=subprocess.run(command,capture_output=True,text=True,timeout=1800)
 (O/(tag+'.stdout')).write_text(p.stdout);(O/(tag+'.stderr')).write_text(p.stderr)
 assert p.returncode==0,(tag,p.returncode,p.stderr[-2000:])
 return p.stdout

def prepare_smoke_clearance(m,mh):
 import common
 guard=jread(E/'PRECISION_GUARD_CHECK_PASS.json');assert guard['passed'] and guard['generation_manifest_sha256']==mh and guard['jobid']==jread(E/'PRECISION_GUARD_CHECK_SUBMITTED.json')['jobid'] and guard['guard_submission_receipt_sha256']==sha(E/'PRECISION_GUARD_CHECK_SUBMITTED.json') and guard['run_precision_guard_sha256']==sha(E/'run_precision_guard.py') and guard['numeric_verifier_sha256']==sha(E/'verify_logits_numeric.py')
 for lane in (0,1):
  receipt=jread(O/('SMOKE_PASS_'+str(lane)+'.json'));assert receipt['passed'] and receipt['manifest_sha256']==mh and receipt['smoke_zero_validation_confirmed']
 cpu_command(E/'verify_logits_numeric.py',['--scope','smoke'],'continuation_smoke_verify')
 latest=jread(E/'evaluation/smoke/latest.json');recount=jread(latest['file']);assert latest['complete'] and latest['n']==2 and sha(latest['file'])==latest['sha256']
 assert recount['manifest_sha256']==mh and recount['complete'] and recount['n']==recount['expected']==2 and recount['full_vocabulary_logits_recomputed'] and recount['all_update_vectors_recomputed'] and recount['max_log_probability_abs_error']<=1e-10
 profile_out=cpu_command(E/'continuation_smoke_profile.py',[],'continuation_smoke_profile');profile=json.loads(profile_out.strip().splitlines()[-1]);assert profile['n']==2 and profile['generation_manifest_sha256']==mh
 profiles={(str(p['id']),p['position_id']):p for p in profile['profiles']};samples=[];main_seconds=[];compressed=0
 for smoke in m['smoke_cases']:
  plan=next(p for p in common.rows(E/'positions.jsonl') if str(p['id'])==str(smoke['id']) and p['position_id']==smoke['position_id'])
  folder=O/'completed'/common.stem(plan);ac=jread(folder/'ACCEPTANCE.json');assert ac['passed'] and ac['manifest_sha256']==mh
  for n,h in ac['files_sha256'].items():assert sha(folder/n)==h
  probe=jread(folder/'probes.json');assert len(probe['zero_checks'])==8 and all(z['full_candidate_logits_exact'] and z['actual_input_hashes_exact'] and z['fp64_per_token_exact'] for z in probe['zero_checks'])
  actual=profiles[(str(smoke['id']),smoke['position_id'])];main=actual['main_and_capture_seconds'];assert actual['candidate_raw_tensor_count']==22
  main_seconds.append(main);sizes={n:(folder/n).stat().st_size for n in ac['files_sha256']};assert sizes==actual['sizes_bytes'];compressed+=sum(sizes.values());samples.append(actual)
 projected=math.ceil(compressed*m['position_count']/2);reserve=int(.8*(1024**3));free=os.statvfs(str(E)).f_bavail*os.statvfs(str(E)).f_frsize;passed=projected+reserve<=free
 clearance=dict(actual_utc=now(),passed=passed,manifest_sha256=mh,smoke_recount=latest,smoke_samples=samples,smoke_actual_compressed_bytes=compressed,projected_full_bytes_from_actual_smoke_case_average=projected,reserve_bytes=reserve,free_bytes=free,no_historical_deletion=True,formula='sum actual smoke gzip+json file bytes *87/2 plus0.8GiB compared with current statvfs free',projection_is_mean_with_reserve_not_a_compression_theorem=True,frozen_full_candidate_token_count=423,smoke_case_average_equivalent_candidate_tokens=435)
 if (E/'FULL_STORAGE_CLEARANCE.json').exists():assert jread(E/'FULL_STORAGE_CLEARANCE.json')['manifest_sha256']==mh
 else:save_once(E/'FULL_STORAGE_CLEARANCE.json',clearance)
 assert passed,('insufficient space; no deletion or full submission',clearance)
 estimate=(m['position_count']/2)*(sum(main_seconds)/2)*1.040265507742*1.5+600
 wall_minutes=next((minutes for minutes in (30,45,60,90,120) if estimate<=minutes*60),None)
 assert wall_minutes is not None
 assert estimate<=120*60,('estimated full work exceeds120min resource envelope',estimate)
 runtime=dict(actual_utc=now(),manifest_sha256=mh,samples=samples,estimated_seconds_per_lane_with_buffer=estimate,requested_wall_minutes=wall_minutes,mean_prefix_work_ratio=1.040265507742,allowance='measured main+capture+weak seconds *prefix ratio1.04 *runtime variance1.5 plus600s startup; select smallest30/45/60/90/120min envelope; model-loading not inferred as zero')
 if not (E/'FULL_TIME_ESTIMATE.json').exists():save_once(E/'FULL_TIME_ESTIMATE.json',runtime)
 return wall_minutes,clearance

def check_full_complete(m,mh):
 import common
 for plan in common.rows(E/'positions.jsonl'):
  folder=O/'completed'/common.stem(plan);ac=jread(folder/'ACCEPTANCE.json');assert ac['passed'] and ac['manifest_sha256']==mh
  for n,h in ac['files_sha256'].items():assert sha(folder/n)==h
 latest_path=E/'evaluation/full/latest.json'
 if latest_path.exists():
  latest=jread(latest_path);recount=jread(latest['file']);assert latest['complete'] and latest['n']==87 and sha(latest['file'])==latest['sha256'] and recount['manifest_sha256']==mh
 else:cpu_command(E/'verify_logits_numeric.py',['--scope','full'],'continuation_full_verify')
 cpu_command(E/'evaluation/mechanical_summary.py',[],'continuation_mechanical_summary')
 receipt=json.loads((O/'continuation_mechanical_summary.stdout').read_text().strip().splitlines()[-1]);assert receipt['complete'] and receipt['n']==87 and sha(receipt['summary_file'])==receipt['summary_sha256']
 summary_path=pathlib.Path(receipt['summary_file']);receipt_path=summary_path.parent/('RECEIPT_'+summary_path.name[len('MECHANICAL_SUMMARY_'):]);assert jread(receipt_path)==receipt
 cpu_command(E/'evaluation/paired_position_readout.py',['--receipt',str(receipt_path)],'continuation_paired_position')
 paired=json.loads((O/'continuation_paired_position.stdout').read_text().strip().splitlines()[-1]);assert paired['complete'] and paired['pair_count']==27 and sha(paired['summary_file'])==paired['summary_sha256']
 return dict(main=receipt,main_receipt_path=str(receipt_path),paired=paired)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');a=ap.parse_args();m,mh,cm,ch=check_files(require_activation=not a.check_only)
 if a.check_only:print(json.dumps(dict(continuation_hashes_passed=True,generation_manifest_sha256=mh,continuation_manifest_sha256=ch,activation_required_to_run=True)));return
 lock=(O/'continuation_dispatcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 log('activated_waiting_smoke',generation_manifest_sha256=mh,continuation_manifest_sha256=ch,authorized_scientific_work_deadline=m['active_work_deadline_utc'])
 while True:
  check_files();states=state_of(cm['smoke_jobid'])
  if terminal(states):break
  log('waiting_smoke',states=states);time.sleep(60)
 wall_minutes,clearance=prepare_smoke_clearance(m,mh);log('smoke_cpu_and_storage_passed',wall_minutes=wall_minutes,storage=clearance)
 while True:
  check_files()
  if (E/'FULL_SUBMITTED.json').exists():
   submission=jread(E/'FULL_SUBMITTED.json');assert submission['manifest_sha256']==mh;job=submission['jobid'];log('existing_full_receipt_reused',jobid=job);break
  s=snapshot();unknown_full=[x for x in s['jobs'] if str(E) in x['command'] and 'full' in x['command']]
  assert not unknown_full,('existing full job without canonical receipt; block duplicate',unknown_full)
  if s['project_gpus']+2>4 or s['account_gpus']+2>8:
   log('waiting_resource_budget',snapshot=s);time.sleep(60);continue
  (O/'continuation_submission_once.lock').mkdir()
  cmd=['sbatch','--parsable','--time='+str(wall_minutes),'--job-name=r1_fixed_fact_full',str(E/'continuation_full.sbatch')]
  intent=dict(actual_utc=now(),command=cmd,manifest_sha256=mh,continuation_manifest_sha256=ch,resource_snapshot=s,reserved_new_gpus=2)
  save_once(E/'FULL_SUBMISSION_INTENT.json',intent)
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
  assert p.returncode==0,('submission failed, no retry',p.returncode,p.stdout,p.stderr)
  job=p.stdout.strip().split(';')[0];assert job.isdigit(),('ambiguous job id, no retry',p.stdout)
  save_once(E/'FULL_SUBMITTED.json',dict(**intent,jobid=job,stdout=p.stdout,stderr=p.stderr,phase='full',lanes=[0,1]));log('full_submitted',jobid=job);break
 while True:
  check_files();states=state_of(job)
  if terminal(states):break
  log('waiting_full',jobid=job,states=states);time.sleep(60)
 receipt=check_full_complete(m,mh);save_once(E/'CONTINUATION_COMPLETION.json',dict(actual_utc=now(),complete=True,jobid=job,generation_manifest_sha256=mh,continuation_manifest_sha256=ch,mechanical_receipt=receipt,scientific_interpretation_pending=True))
 log('completed_fixed_computation_and_summary',jobid=job,n=87,receipt=receipt,scientific_interpretation_pending=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  log('failed_stop',error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True,preserve_existing_outputs=True);raise
