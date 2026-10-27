"""Resource-only PAI continuation. Never submits/resumes LEAD/VCD or changes inference."""
import datetime,fcntl,hashlib,json,os,re,subprocess,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent
R=E.parent.parent
OLD=R/'experiments/20261015_openvl_rethinker_lead_vcd_all'
A=E/'automation'
A.mkdir(exist_ok=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def log(event,**kw):
    v=dict(kw);v['utc']=utc();v['event']=event
    with (A/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n');f.flush()
    save(A/'STATE.json',v);print(json.dumps(v),flush=True)
def call(args,timeout=90):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError((args,p.returncode,p.stderr[-3000:]))
    return p.stdout
def queue():
    out=call(['squeue','-u',os.environ['USER'],'--array','-h','-o','%i|%T|%R|%j|%K|%F'])
    rows=[]
    for line in out.splitlines():
        parts=line.split('|')
        if len(parts)!=6:raise RuntimeError(('queue parse',line))
        jid,state,reason,name,task,array=parts
        if reason.strip('()')=='DependencyNeverSatisfied':continue
        details=call(['scontrol','show','job',jid,'-o'])
        cmd=re.search(r'(?:^|\s)Command=(\S+)',details)
        cmd=cmd.group(1) if cmd else ''
        vals=re.findall(r'(?:gres/gpu[:=])(?:[A-Za-z0-9_-]+:)?(\d+)',details)
        gp=max([int(x) for x in vals] or [0])
        if gp==0:continue
        rows.append(dict(job_id=jid,state=state,reason=reason,name=name,task=task,array=array,command=cmd,gpus=gp))
    return rows
def gates(m,mh):
    approval=read(E/'ROOT_APPROVAL.json');assert approval['approved'] and approval['manifest_sha256']==mh
    for f,h in m['files'].items():assert sha(E/f)==h,('manifest drift',f)
    for model in ('r1','openvl','rethinker'):
        s=read(E/'smoke'/model/'PASS.json');assert s['passed'] and s['manifest_sha256']==mh
    cpu=read(E/'CPU_IMPORT_RECEIPT.json');assert cpu['passed'] and cpu['manifest_sha256']==mh

def scheduler_rows(out,jid):
    # JobID preserves array_job_task identity; JobIDRaw is the numeric internal task job.
    return [parts for parts in (line.split('|') for line in out.splitlines()) if parts[0].strip()==jid]

def account_snapshot(rows):
    old_slots=0
    for lane in range(4):
        st=read(OLD/'status'/('full_lane'+str(lane)+'.json'))
        if st['state']!='COMPLETE':old_slots+=1
    old_actual=sum(x['gpus'] for x in rows if x['command'].startswith(str(OLD)+'/'))
    old_reserve=max(old_slots,old_actual)
    pai=[x for x in rows if x['command'].startswith(str(E)+'/')]
    other=[x for x in rows if x['command'].startswith(str(R/'experiments')+'/') and x not in pai and not x['command'].startswith(str(OLD)+'/')]
    total=sum(x['gpus'] for x in rows)
    reserved=old_reserve+sum(x['gpus'] for x in pai+other)
    return dict(old_unfinished_lanes=old_slots,old_actual_gpus=old_actual,old_reserved_gpus=old_reserve,
        pai=pai,other_project=other,project_reserved_gpus=reserved,account_reserved_gpus=total,
        free_slots=max(0,min(4-reserved,8-total)))
def main():
    lock=(A/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(E/'manifest.json');mh=sha(E/'manifest.json');gates(m,mh)
    journal=read(A/'submissions.json') if (A/'submissions.json').exists() else {'submissions':[]}
    checked=set();initial=True
    log('STARTED',manifest_sha256=mh,controller_pid=os.getpid(),old_reservation='reserve all unfinished LEAD/VCD lanes before PAI slots')
    while True:
        try:
            gates(m,mh);rows=queue();snap=account_snapshot(rows)
            active_lanes={int(x['task']) for x in snap['pai'] if x['task'].isdigit()}
            if len(active_lanes)!=len(snap['pai']):raise RuntimeError(('PAI task identification',snap['pai']))
            active_ids={x['job_id'] for x in rows}
            unresolved=[]
            for submission in journal['submissions']:
                jid=submission['task_job_id']
                if jid in checked or jid in active_ids:continue
                out=call(['sacct','-j',jid,'--format=JobID%40,State,ExitCode','-P','-n'])
                match=scheduler_rows(out,jid)
                if not match:
                    unresolved.append(submission);continue
                if any(x[1] in ('RUNNING','PENDING','CONFIGURING','COMPLETING','REQUEUED') for x in match):
                    unresolved.append(submission);continue
                if any(x[1]!='COMPLETED' or x[2]!='0:0' for x in match):raise RuntimeError(('scheduler failure',jid,match))
                st=read(E/'status'/('full_lane'+str(submission['lane'])+'.json'))
                if st.get('array_job_id')==submission['array_job_id']:
                    assert st['state']=='COMPLETE' or (st['state']=='PAUSE' and st.get('reason') in ('time_budget','signal')),('worker stop',jid,st)
                checked.add(jid)
            # Submitted-but-not-yet-visible tasks still reserve their lane and GPU.
            for submission in unresolved:
                if submission['lane'] not in active_lanes:
                    active_lanes.add(submission['lane']);snap['project_reserved_gpus']+=1;snap['account_reserved_gpus']+=1
            snap['free_slots']=max(0,min(4-snap['project_reserved_gpus'],8-snap['account_reserved_gpus']))
            complete=[];todo=[]
            for lane in range(6):
                p=E/'status'/('full_lane'+str(lane)+'.json')
                st=read(p) if p.exists() else None
                if st and st['manifest_sha256']!=mh:raise RuntimeError(('status drift',lane))
                if st and st['state']=='COMPLETE':complete.append(lane);continue
                if lane in active_lanes:continue
                if st and st['state']=='FAILED':raise RuntimeError(('worker failure',lane,st))
                if st and st['state']=='PAUSE' and st.get('reason') not in ('time_budget','signal'):raise RuntimeError(('pause needs root',lane,st))
                todo.append(lane)
            if len(complete)==6 and not snap['pai']:
                counts={model:len(list((E/'results'/model).glob('*.json.gz'))) for model in ('r1','openvl','rethinker')}
                assert all(n==7438 for n in counts.values()),counts
                collector=call(['python3','-u',str(E/'completion_summary.py')],timeout=1800)
                log('ALL_COMPLETE',manifest_sha256=mh,counts=counts,collector_exit_code=0,collector_stdout_tail=collector[-6000:]);return
            if snap['project_reserved_gpus']>4 or snap['account_reserved_gpus']>8:raise RuntimeError(('resource oversubscription',snap))
            if snap['free_slots'] and todo:
                st=os.statvfs(E);free=st.f_bavail*st.f_frsize
                if free<128*1024**2:raise RuntimeError(('storage requires root',free))
                # One lane at a time; R1/OpenVL/Rethinker interleaved, each model retains modulo ownership.
                order=[0,4,2,1,5,3];todo.sort(key=order.index)
                lane=todo[0]
                attempts=sum(x['lane']==lane for x in journal['submissions'])
                if attempts>=12:raise RuntimeError(('bounded continuation exhausted',lane,attempts))
                out=call(['sbatch','--parsable','--array='+str(lane),str(E/'full.sbatch')])
                job=out.strip().split(';')[0]
                assert job.isdigit(),out
                entry=dict(utc=utc(),lane=lane,array_job_id=job,task_job_id=job+'_'+str(lane),manifest_sha256=mh,resource_snapshot=snap,attempt=attempts+1)
                journal['submissions'].append(entry);save(A/'submissions.json',journal)
                log('SUBMITTED',**entry)
                # Wait before the next submission so the scheduler can expose the reserved task.
                time.sleep(15);continue
            log('WAITING',complete_lanes=complete,todo_lanes=todo,resource_snapshot=snap)
            time.sleep(45)
        except subprocess.TimeoutExpired as ex:
            log('COMMAND_TIMEOUT_RETRY',error=repr(ex));time.sleep(45)
        except Exception as ex:
            log('STOPPED_REQUIRES_REVIEW',error=repr(ex),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
