# Resource-only admission; frozen model/generation sources are never imported or modified.
import argparse,datetime,hashlib,json,pathlib,re,subprocess,sys,time
E=pathlib.Path(__file__).resolve().parent;R=E.parent.parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(args):
 p=subprocess.run(args,text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError((args,p.returncode,p.stderr))
 return p.stdout

def gpu_count(detail):
 hits=[]
 for pattern in [r'(?:^|\s)(?:TRES|ReqTRES|AllocTRES)=([^\s]+)',r'(?:^|\s)(?:TresPerJob|TresPerNode)=([^\s]+)']:
  for value in re.findall(pattern,detail):
   for part in value.split(','):
    if part.startswith('gres/gpu='):
     count=part.split('=',1)[1]
     if not count.isdigit():raise RuntimeError('unknown GPU TRES '+part)
     hits.append(int(count))
    elif part.startswith('gres:gpu:'):
     count=part[len('gres:gpu:'):]
     if not count.isdigit():raise RuntimeError('unknown GPU GRES '+part)
     hits.append(int(count))
 if not hits and re.search(r'(?:TRES|TresPerJob|TresPerNode)=[^\s]*gpu',detail):raise RuntimeError('unrecognized GPU resource representation')
 return max(hits,default=0)

def snapshot(allocated_only=False):
 result=[];raw=run(['squeue','-r','-h','-u','scx9ftv','-o','%i|%T|%R'])
 for line in raw.splitlines():
  job,state,reason=line.strip().split('|',2)
  detail=run(['scontrol','show','job',job,'-o'])
  if 'JobId=' not in detail:raise RuntimeError('queue changed during details '+job)
  match=re.search(r'(?:^|\s)Command=([^\n]+?)(?=\s(?:WorkDir|StdErr|StdIn|StdOut|Comment|Power)=|$)',detail)
  command=match.group(1) if match else ''
  if not command:
   command_match=re.search(r'(?:^|\s)Command=([^\s]+)',detail);command=command_match.group(1) if command_match else ''
  impossible=state=='PENDING' and 'DependencyNeverSatisfied' in reason
  n=gpu_count(detail)
  if n and not command:raise RuntimeError('GPU job has no command '+job)
  allocated=state in ('RUNNING','COMPLETING','CONFIGURING','SUSPENDED')
  result.append(dict(jobid=job,state=state,reason=reason,command=command,requested_gpus=n,
                     counted_gpus=n if (not impossible and (allocated or not allocated_only)) else 0,
                     project_member=command.startswith(str(R/'experiments')+'/'),
                     excluded_impossible_dependency=impossible,allocated=allocated))
 return dict(actual_utc=now(),allocated_only=allocated_only,jobs=result,
             account_gpus=sum(x['counted_gpus'] for x in result),
             project_gpus=sum(x['counted_gpus'] for x in result if x['project_member']))

def validate_manifest():
 import common
 m,mh=common.check_manifest();c=json.loads((E/'continuation_manifest.json').read_text())
 assert mh==c['generation_manifest_sha256']
 for name,digest in c['files_sha256'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==digest,name
 activation=json.loads((E/'CONTINUATION_ACTIVATED.json').read_text())
 assert activation['continuation_manifest_sha256']==hashlib.sha256((E/'continuation_manifest.json').read_bytes()).hexdigest()
 assert activation['generation_manifest_sha256']==mh and activation['root_review_passed']
 return m,mh,c

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--test-only',action='store_true');ap.add_argument('--snapshot-only',action='store_true');a=ap.parse_args()
 if a.test_only:
  assert gpu_count('TRES=cpu=32,gres/gpu=2 TresPerNode=gres:gpu:2')==2
  assert gpu_count('TRES=cpu=2,mem=2G TresPerNode=(null)')==0
  try:gpu_count('TresPerNode=gres:gpu:a100:1')
  except RuntimeError:pass
  else:raise AssertionError('unknown typed GPU must fail closed')
  print('RESOURCE_PARSER_SELFTEST_PASS');return
 if a.snapshot_only:print(json.dumps(snapshot()));return
 m,mh,c=validate_manifest();s=snapshot(allocated_only=True);passed=s['project_gpus']<=4 and s['account_gpus']<=8
 p=E/'output'/('RUNTIME_RESOURCE_GATE_'+str(__import__('os').environ.get('SLURM_JOB_ID','manual'))+'_'+str(__import__('os').environ.get('SLURM_ARRAY_TASK_ID','none'))+'.json')
 with p.open('x') as f:json.dump(dict(actual_utc=now(),passed=passed,generation_manifest_sha256=mh,snapshot=s,own_allocation_included=True,no_unrelated_job_changes=True),f,indent=2)
 assert passed,'resource cap exceeded at model start; own allocation exits without loading model'
if __name__=='__main__':main()
