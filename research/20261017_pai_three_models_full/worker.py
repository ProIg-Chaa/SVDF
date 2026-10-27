"""Frozen full PAI adaptation on R1/OpenVL/Rethinker; no scoring labels."""
import argparse,datetime,gc,gzip,hashlib,json,os,signal,sys,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent
for p in (E/'source',E/'audit_source'):sys.path.insert(0,str(p))
import torch,base
from scale_model_loader import load_model
from state_isolation import StateIsolation,tensor_sha
from methods import Methods,CONFIG,PAI_KW
STOP_SIGNAL=None
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(s):return s['dataset']+':'+str(s['id'])
def stem(s):return hashlib.sha256(key(s).encode()).hexdigest()[:20]
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def write_gz(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
        f.flush();os.fsync(f.fileno())
    assert not p.exists();os.replace(tmp,p)
def read_gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def storage():
    st=os.statvfs(E);return st.f_bavail*st.f_frsize
def manifest():
    m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
    for f,h in m['files'].items():assert sha(E/f)==h,('frozen source',f)
    for model,spec in m['models'].items():
        for f,h in spec['checkpoint_files'].items():assert sha(Path(spec['path'])/f)==h,(model,f)
    return m,mh
def valid(p,s,model,mh):
    if not p.exists():return False
    r=read_gz(p)
    assert r['accepted'] and r['manifest_sha256']==mh and r['sample']==s and r['model']==model and r['sample_key']==key(s)
    a=r['arms']['pai_full'];assert a['generated_tokens']==len(a['generated_token_ids'])>0
    return True
class Runtime:
    def __init__(self,spec,run):
        torch.set_num_threads(4);torch.set_num_interop_threads(4);began=time.perf_counter()
        self.model,self.processor,self.tokenizer,_=load_model(spec,run)
        self.iso=StateIsolation(self.model);self.initial=self.iso.snapshot();self.methods=Methods(self.model,self.tokenizer)
        torch.cuda.synchronize();self.load_seconds=time.perf_counter()-began
    def restore(self):
        self.iso.restore_checked(self.initial);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    def inputs(self,s):
        assert sha(s['image'])==s['image_sha256']
        return base.prepare(self.processor,s,s['image'],self.model.device)
    def generate(self,s,smoke=False):
        self.restore();state={'branch':'main'};count={'main':0,'weak':0,'prefill_main':0,'prefill_weak':0};cache_ids={};finite=0
        def counter(module,args,kws):
            b=state['branch'];count[b]+=1
            if kws.get('past_key_values') is None:count['prefill_'+b]+=1
        def inspect(module,args,kws,out):
            nonlocal finite
            assert torch.isfinite(out.logits[:,-1,:]).all();finite+=1
            if kws.get('past_key_values') is None:cache_ids[state['branch']]=id(out.past_key_values)
        h=self.model.register_forward_pre_hook(counter,with_kwargs=True)
        hh=self.model.register_forward_hook(inspect,with_kwargs=True) if smoke else None
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
        try:
            inputs,n=self.inputs(s);raw=self.methods.generate(inputs,state)
            torch.cuda.synchronize();wall=time.perf_counter()-began;peak=int(torch.cuda.max_memory_allocated());reserved=int(torch.cuda.max_memory_reserved())
            prompt=tensor_sha(inputs['input_ids']);grid=inputs['image_grid_thw'].cpu().tolist()
            image_positions=torch.where(inputs['input_ids'][0]==self.model.config.image_token_id)[0]
            assert image_positions.numel()==raw['attention']['image_key_count']>0
            assert torch.equal(image_positions,torch.arange(image_positions[0],image_positions[-1]+1,device=image_positions.device))
        finally:
            h.remove()
            if hh:hh.remove()
        tokens=raw['token_ids'];nt=len(tokens);att=raw['attention'];meta=raw['metadata']
        assert count['main']==count['weak']==nt and count['prefill_main']==count['prefill_weak']==1
        assert meta['forward_calls']==2*nt and meta['cache_isolated'] and meta['branch_rope_isolated']
        assert att['positive_calls']==att['negative_calls']==att['modified_queries']==26*nt
        if smoke:assert att['nonzero_delta']>0 and meta['cfg_nonzero_steps']>0
        if smoke:assert cache_ids['main']!=cache_ids['weak']
        eos=tokens[-1]==self.tokenizer.eos_token_id
        assert eos or nt==1024
        result=dict(text=self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(),
            generated_token_ids=tokens,generated_tokens=nt,prompt_tokens=int(n),prompt_token_sha256=prompt,image_grid_thw=grid,
            forward_calls=count,wall_seconds=wall,timing_scope='prepare image and prompt + full generation; excludes model load, inter-case GC, IO, scoring',
            peak_cuda_allocated_bytes=peak,peak_cuda_reserved_bytes=reserved,reached_eos=eos,truncated=not eos and nt==1024,
            stop_reason='eos' if eos else 'max_new_tokens',method_metadata=meta,attention_statistics=att,
            upstream_commit='9bbd8bd57a0b0923f996197e4bd3e02cc10b8d58',adaptation='Qwen2.5VL; layers2..27, native processors, greedy matched protocol')
        if smoke:result['engineering']=dict(finite_forwards=finite,cache_ids=cache_ids,image_key_count=int(image_positions.numel()),visual_keys_contiguous=True)
        del inputs;self.iso.restore_checked(self.initial)
        return result
    def smoke_checks(self,s,result,model,reuse):
        from common_decode import generate_baseline
        import pai_adapter as canonical
        self.restore();inp,_=self.inputs(s)
        ref=canonical.generate(self.model,self.tokenizer,inp,config=CONFIG,**PAI_KW)
        assert ref['token_ids']==result['generated_token_ids'],('wrapped official adaptation differs',key(s))
        checks={'direct_adapter_tokens_exact':True,'n':len(ref['token_ids'])}
        refs={}
        for kind in ('native_cot','common_cot','pai_zero'):
            self.restore();inp,n=self.inputs(s)
            with torch.no_grad():
                if kind=='native_cot':
                    out=self.methods.native['generate_cot'](self.model,self.tokenizer,**inp,**CONFIG);ids=out[0,n:].tolist()
                elif kind=='common_cot':ids=generate_baseline(self.model,self.tokenizer,inp,config=CONFIG)['token_ids']
                else:ids=self.methods.generate(inp,{'branch':'main'},zero=True)['token_ids']
            refs[kind]=[int(x) for x in ids]
        assert refs['native_cot']==refs['common_cot']==refs['pai_zero'],('zero mismatch',key(s),[len(x) for x in refs.values()])
        checks['zero_alpha0_gamma1_native_common_exact']=True
        checks['baseline_n']=len(refs['native_cot'])
        if model=='r1' and key(s) in reuse:
            old=reuse[key(s)];assert sha(old['path'])==old['sha256']
            historical=json.loads(Path(old['path']).read_text())
            assert historical['token_ids']==result['generated_token_ids'],('historical R1 PAI tokens mismatch',key(s))
            assert historical['prompt_input_ids_sha256']==result['prompt_token_sha256']
            checks['historical_full_tokens_and_prompt_exact']=True
        self.iso.restore_checked(self.initial);return checks
def reuse_row(s,entry,mh):
    assert sha(entry['path'])==entry['sha256']
    old=json.loads(Path(entry['path']).read_text());n=len(old['token_ids'])
    assert old['dataset']==s['dataset'] and str(old['id'])==str(s['id'])
    assert old['question']==s['question'] and old['options']==s['options'] and old['image_sha256']==s['image_sha256'] and not old['labels_read']
    assert old['metadata']['alpha']==.2 and old['metadata']['gamma']==1.1 and old['metadata']['beta']==.1
    arm=dict(text=old['text'],generated_token_ids=old['token_ids'],generated_tokens=n,prompt_token_sha256=old['prompt_input_ids_sha256'],
        wall_seconds=old['seconds'],timing_scope='historical 20261013 PAI: generation only; image/prompt preparation excluded; do not pool with new end-to-end timing',
        method_metadata=old['metadata'],attention_statistics=old['attention'],forward_calls={'main':n,'weak':n,'prefill_main':1,'prefill_weak':1},
        reached_eos=old['token_ids'][-1]==151645,truncated=n==1024 and old['token_ids'][-1]!=151645,
        reused=True,source_path=entry['path'],source_sha256=entry['sha256'],source_manifest_sha256=old['manifest_sha256'])
    return dict(schema_version=1,accepted=True,manifest_sha256=mh,model='r1',sample=s,sample_key=key(s),sample_hash=stem(s),
        completed_utc=utc(),labels_read=False,reused=True,arms={'pai_full':arm})
def main(a):
    global STOP_SIGNAL
    begun=time.monotonic();m,mh=manifest();model=m['lanes'][str(a.lane)]['model'];mod=m['lanes'][str(a.lane)]['modulo']
    run_id=str(os.environ.get('SLURM_JOB_ID','manual'))+'_'+str(time.time_ns());run=E/'runs'/a.phase/('lane'+str(a.lane))/run_id;run.mkdir(parents=True)
    state=dict(lane=a.lane,model=model,phase=a.phase,manifest_sha256=mh,slurm_job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),run=str(run),started_utc=utc(),completed=0,expected=0)
    def status(st,**kw):
        value=dict(state,state=st,utc=utc(),**kw);save(run/'STATUS.json',value);save(E/'status'/(a.phase+'_lane'+str(a.lane)+'.json'),value)
    def receive(sig,frame):
        global STOP_SIGNAL
        STOP_SIGNAL=signal.Signals(sig).name
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,receive)
    rows=lines(E/('smoke_public.jsonl' if a.phase=='smoke' else 'public_all.jsonl'))
    if a.phase=='full':rows=[s for i,s in enumerate(rows) if i%2==mod]
    reuse={x['sample_key']:x for x in lines(E/'reuse_r1_mmvp.jsonl')}
    state['expected']=len(rows)
    try:
        if a.phase=='full':
            gate=json.loads((E/'smoke'/model/'PASS.json').read_text());assert gate['passed'] and gate['manifest_sha256']==mh
            approval=json.loads((E/'ROOT_APPROVAL.json').read_text());assert approval['approved'] and approval['manifest_sha256']==mh
        status('RUNNING')
        target=E/'results'/model if a.phase=='full' else run/'outputs';target.mkdir(parents=True,exist_ok=True)
        todo=[];reused=0
        for s in rows:
            p=target/(stem(s)+'.json.gz')
            if valid(p,s,model,mh):state['completed']+=1
            elif a.phase=='full' and model=='r1' and key(s) in reuse:
                write_gz(p,reuse_row(s,reuse[key(s)],mh));state['completed']+=1;reused+=1
            else:todo.append(s)
        if not todo:status('COMPLETE',newly_completed=0,reused_this_run=reused);return
        if storage()<128*1024**2:status('PAUSE',reason='storage_pause',free_bytes=storage());return
        rt=Runtime(m['models'][model],run);save(run/'LOAD.json',dict(load_seconds=rt.load_seconds,utc=utc()))
        fresh=0;receipts=[]
        for s in todo:
            reason='signal' if STOP_SIGNAL else ('time_budget' if time.monotonic()-begun>=23*3600 else ('storage_pause' if storage()<128*1024**2 else None))
            if reason:status('PAUSE',reason=reason,signal=STOP_SIGNAL,newly_completed=fresh,reused_this_run=reused,free_bytes=storage());return
            value=rt.generate(s,a.phase=='smoke');check=rt.smoke_checks(s,value,model,reuse) if a.phase=='smoke' else None
            row=dict(schema_version=1,accepted=True,manifest_sha256=mh,model=model,sample=s,sample_key=key(s),sample_hash=stem(s),lane=a.lane,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),completed_utc=utc(),labels_read=False,arms={'pai_full':value})
            if check:row['engineering_checks']=check
            p=target/(stem(s)+'.json.gz');write_gz(p,row);fresh+=1;state['completed']+=1
            receipts.append(dict(sample_key=key(s),path=str(p),sha256=sha(p),gzip_bytes=p.stat().st_size))
            status('RUNNING',newly_completed=fresh,reused_this_run=reused,last_sample=key(s));print('CASE_COMPLETE',model,key(s),fresh,p.stat().st_size,flush=True)
        if a.phase=='smoke':
            save(E/'smoke'/model/'PASS.json',dict(passed=True,manifest_sha256=mh,utc=utc(),run=str(run),n_samples=len(rows),receipts=receipts,
                checks=['unmodified_20261013_adapter_full_tokens','native_common_zero_alpha0_gamma1_exact','finite_all_forwards','distinct_caches','native_processor','contiguous_visual_mask','counts26layers2branches','state_isolation','R1_historical_MMVP_exact_when_applicable']))
        status('COMPLETE',newly_completed=fresh,reused_this_run=reused)
    except Exception as ex:status('FAILED',error=repr(ex),traceback=traceback.format_exc());raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','full'],required=True);p.add_argument('--lane',type=int,choices=range(6),required=True)
    try:main(p.parse_args())
    except Exception:print('FAILED',traceback.format_exc(),flush=True);sys.exit(1)
