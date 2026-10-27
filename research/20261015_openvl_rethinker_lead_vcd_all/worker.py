"""Two models, official Full LEAD and continuous VCD, frozen public 14 datasets."""
import argparse, datetime, gc, gzip, hashlib, json, os, signal, sys, time, traceback
from pathlib import Path
E = Path(__file__).resolve().parent
for p in (E/'source', E/'audit_source'):sys.path.insert(0, str(p))
import torch
import base
from scale_model_loader import load_model
from state_isolation import StateIsolation, tensor_sha
from methods import Methods, ARMS, CONFIG, LEAD_KW, VCD_KW
STOP_SIGNAL = None

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(s): return s['dataset']+':'+str(s['id'])
def stem(s): return hashlib.sha256(key(s).encode()).hexdigest()[:20]
def read_lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write_json(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def write_gz(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            z.write(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode())
        f.flush();os.fsync(f.fileno())
    assert not p.exists(),('refuse overwrite',str(p))
    os.replace(tmp,p)
def read_gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def storage_bytes():
    x=os.statvfs(E);return x.f_bavail*x.f_frsize

def verify_manifest():
    m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
    for f,h in m['files'].items():assert sha(E/f)==h,('frozen file changed',f)
    for model,spec in m['models'].items():
        for f,h in spec['checkpoint_files'].items():assert sha(Path(spec['path'])/f)==h,(model,f)
    return m,mh

def valid_result(p,s,model,mh):
    if not p.exists():return False
    r=read_gz(p)
    assert r['accepted'] and r['manifest_sha256']==mh and r['sample']==s
    assert r['model']==model and r['sample_key']==key(s) and set(r['arms'])==set(ARMS)
    for a in ARMS:
        v=r['arms'][a];assert v['generated_tokens']==len(v['generated_token_ids'])>0
    return True

class Runtime:
    def __init__(self,spec,run):
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        t=time.perf_counter()
        self.model,self.processor,self.tokenizer,_=load_model(spec,run)
        self.isolation=StateIsolation(self.model);self.initial=self.isolation.snapshot()
        self.methods=Methods(self.model,self.tokenizer)
        torch.cuda.synchronize();self.load_seconds=time.perf_counter()-t
    def restore(self):
        self.isolation.restore_checked(self.initial)
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    def inputs(self,s):
        assert sha(s['image'])==s['image_sha256'],('changed image',key(s))
        return base.prepare(self.processor,s,s['image'],self.model.device)
    def generate(self,s,arm,smoke=False):
        self.restore();state={'branch':'main'}
        count={'main':0,'weak':0,'prefill_main':0,'prefill_weak':0}
        cache_ids={};finite_checked=0
        def counter(module,args,kws):
            b=state['branch'];count[b]+=1
            if kws.get('past_key_values') is None:count['prefill_'+b]+=1
        def inspect(module,args,kws,out):
            nonlocal finite_checked
            assert torch.isfinite(out.logits[:,-1,:]).all(),('nonfinite',arm)
            finite_checked+=1
            if kws.get('past_key_values') is None:cache_ids[state['branch']]=id(out.past_key_values)
        h=self.model.register_forward_pre_hook(counter,with_kwargs=True)
        hh=self.model.register_forward_hook(inspect,with_kwargs=True) if smoke else None
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        try:
            inputs,length=self.inputs(s)
            with torch.no_grad():tokens=self.methods.generate(arm,inputs,state)
            torch.cuda.synchronize();wall=time.perf_counter()-t
            peak=int(torch.cuda.max_memory_allocated());reserved=int(torch.cuda.max_memory_reserved())
            prompt_sha=tensor_sha(inputs['input_ids']);grid=inputs['image_grid_thw'].cpu().tolist()
        finally:
            h.remove()
            if hh:hh.remove()
        assert tokens and 0<len(tokens)<=CONFIG['max_new_tokens']
        assert count['main']==len(tokens) and count['prefill_main']==1,(arm,count,len(tokens))
        if arm=='vcd_continuous':
            assert count['weak']==len(tokens) and count['prefill_weak']==1
            if smoke:assert cache_ids['main']!=cache_ids['weak']
        else:assert count['weak']==0
        eos=tokens[-1]==self.tokenizer.eos_token_id
        stop='eos' if eos else ('max_new_tokens' if len(tokens)==1024 else 'native_LEAD_answer_budget')
        assert eos or len(tokens)==1024 or arm=='full_lead'
        text=self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        result=dict(text=text,generated_token_ids=tokens,generated_tokens=len(tokens),prompt_tokens=int(length),
            prompt_token_sha256=prompt_sha,image_grid_thw=grid,forward_calls=count,wall_seconds=wall,
            timing_scope='prepare image and prompt + full generation; excludes model load, inter-arm GC, IO, scoring',
            peak_cuda_allocated_bytes=peak,peak_cuda_reserved_bytes=reserved,reached_eos=eos,
            truncated=not eos and len(tokens)==1024,stop_reason=stop,method_metadata=self.methods.metadata(arm))
        if smoke:result['engineering']={'finite_forwards':finite_checked,'cache_ids':cache_ids}
        del inputs
        self.isolation.restore_checked(self.initial)
        return result
    def smoke_checks(self,s,values):
        from adapters.lead_adapter import generate as lead_ref
        from adapters.vcd_adapter import generate as vcd_ref
        from adapters.common_decode import generate_baseline
        checks={}
        for arm in ARMS:
            self.restore();inputs,_=self.inputs(s)
            fn=lead_ref if arm=='full_lead' else vcd_ref
            cfg=dict(CONFIG,**(LEAD_KW if arm=='full_lead' else VCD_KW))
            with torch.no_grad():ref=fn(self.model,self.tokenizer,inputs,variant='original',seed=42,config=cfg,context={'math_ids':self.methods.math_ids})
            assert ref['token_ids']==values[arm]['generated_token_ids'],('reference token mismatch',key(s),arm)
            checks[arm]={'exact_reference_tokens':True,'n':len(ref['token_ids']),'reference_metadata':ref['metadata']}
            del inputs,ref
        refs={}
        for kind in ('native_cot','common_cot','vcd_zero'):
            self.restore();inputs,n=self.inputs(s)
            with torch.no_grad():
                if kind=='native_cot':
                    out=self.methods.native['generate_cot'](self.model,self.tokenizer,**inputs,**CONFIG);ids=out[0,n:].tolist()
                elif kind=='common_cot':ids=generate_baseline(self.model,self.tokenizer,inputs,config=CONFIG)['token_ids']
                else:ids=vcd_ref(self.model,self.tokenizer,inputs,variant='zero',config=dict(CONFIG,**VCD_KW))['token_ids']
            refs[kind]=[int(x) for x in ids]
        assert refs['native_cot']==refs['common_cot']==refs['vcd_zero'],('zero mismatch',key(s))
        checks['baseline']={'native_common_zero_exact':True,'n':len(refs['native_cot']),
            'token_sha256':hashlib.sha256(json.dumps(refs['native_cot']).encode()).hexdigest()}
        self.isolation.restore_checked(self.initial)
        return checks

def main(args):
    global STOP_SIGNAL
    start=time.monotonic();m,mh=verify_manifest()
    model='openvl' if args.lane<2 else 'rethinker'
    run_id=str(os.environ.get('SLURM_JOB_ID','manual'))+'_'+str(time.time_ns())
    run=E/'runs'/args.phase/('lane'+str(args.lane))/run_id;run.mkdir(parents=True)
    status_base=dict(lane=args.lane,model=model,phase=args.phase,manifest_sha256=mh,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
        run=str(run),started_utc=utc(),completed=0,expected=0)
    def status(state,**kwargs):
        value=dict(status_base,state=state,utc=utc(),**kwargs)
        write_json(run/'STATUS.json',value);write_json(E/'status'/(args.phase+'_lane'+str(args.lane)+'.json'),value)
    def receive(sig,frame):
        global STOP_SIGNAL
        STOP_SIGNAL=signal.Signals(sig).name
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,receive)
    rows=read_lines(E/('smoke_public.jsonl' if args.phase=='smoke' else 'public_all.jsonl'))
    if args.phase=='full':rows=[s for i,s in enumerate(rows) if i%2==args.lane%2]
    status_base['expected']=len(rows)
    try:
        if args.phase=='full':
            for other in ('openvl','rethinker'):
                gate=json.loads((E/'smoke'/other/'PASS.json').read_text())
                assert gate['passed'] and gate['manifest_sha256']==mh
        status('RUNNING')
        target=(E/'results'/model) if args.phase=='full' else (run/'outputs')
        target.mkdir(parents=True,exist_ok=True)
        todo=[]
        for s in rows:
            p=target/(stem(s)+'.json.gz')
            if valid_result(p,s,model,mh):status_base['completed']+=1
            else:todo.append(s)
        if not todo:status('COMPLETE',newly_completed=0);return
        if storage_bytes()<128*1024**2:status('PAUSE',reason='storage_pause',free_bytes=storage_bytes());return
        rt=Runtime(m['models'][model],run)
        write_json(run/'LOAD.json',{'load_seconds':rt.load_seconds,'utc':utc(),'manifest_sha256':mh})
        receipts=[];fresh=0
        for index,s in enumerate(todo):
            reason='signal' if STOP_SIGNAL else ('time_budget' if time.monotonic()-start>=23*3600 else ('storage_pause' if storage_bytes()<128*1024**2 else None))
            if reason:status('PAUSE',reason=reason,signal=STOP_SIGNAL,newly_completed=fresh,free_bytes=storage_bytes());return
            order=ARMS if int(stem(s),16)%2==0 else tuple(reversed(ARMS))
            began=utc();values={a:rt.generate(s,a,args.phase=='smoke') for a in order}
            assert values[ARMS[0]]['prompt_token_sha256']==values[ARMS[1]]['prompt_token_sha256']
            assert values[ARMS[0]]['image_grid_thw']==values[ARMS[1]]['image_grid_thw']
            check=rt.smoke_checks(s,values) if args.phase=='smoke' else None
            row=dict(schema_version=1,manifest_sha256=mh,model=model,sample=s,sample_key=key(s),sample_hash=stem(s),
                lane=args.lane,slurm_job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                started_utc=began,completed_utc=utc(),order=list(order),accepted=True,labels_read=False,arms=values)
            if check:row['engineering_checks']=check
            p=target/(stem(s)+'.json.gz');write_gz(p,row)
            fresh+=1;status_base['completed']+=1
            receipts.append({'sample_key':key(s),'path':str(p),'sha256':sha(p),'gzip_bytes':p.stat().st_size})
            status('RUNNING',newly_completed=fresh,last_sample=key(s))
            print('CASE_COMPLETE',model,key(s),fresh,'gzip_bytes',p.stat().st_size,flush=True)
        if args.phase=='smoke':
            write_json(E/'smoke'/model/'PASS.json',dict(passed=True,manifest_sha256=mh,utc=utc(),run=str(run),
                n_samples=len(rows),receipts=receipts,gzip_bytes_total=sum(x['gzip_bytes'] for x in receipts),
                checks=['official_vs_audited_LEAD_exact','lean_vs_audited_VCD_exact','native_common_zero_exact',
                    'finite_all_forwards','VCD_distinct_caches','native_processor','common_prompt','forward_counts','state_isolation']))
        status('COMPLETE',newly_completed=fresh)
    except Exception as ex:
        status('FAILED',error=repr(ex),traceback=traceback.format_exc());raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','full'],required=True)
    p.add_argument('--lane',type=int,choices=range(4),required=True);args=p.parse_args()
    try:main(args)
    except Exception:print('FAILED',traceback.format_exc(),flush=True);sys.exit(1)
