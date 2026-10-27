"""Same resident R1, same 32 cases, natural generation without per-token audit I/O."""
import argparse,functools,gc,hashlib,itertools,json,os,sys,time,traceback
from pathlib import Path
import frozen_engine as eng
from lean_generation import generate_lean_fixed
E=Path(__file__).resolve().parent;R=E.parent.parent;O=R/'output/experiments'/E.name
C=R/'experiments/20261001_cross_method_audit_r1_medium';sys.path.insert(0,str(C))
from adapters.lead_adapter import _compile_official,official_math_ids
from adapters.common_decode import prefill,advance
from adapters.vcd_adapter import add_diffusion_noise
ARMS=('cot_hard','once_soft','input_fixed4','full_lead','vcd_continuous')
CONFIG={'temperature':.6,'top_k':20,'top_p':.95,'min_p':0.,'do_sample':False,'max_new_tokens':1024}
LEAD_KW={'alpha_0':.6,'beta_0':.7,'window_size':256,'max_switch_count':5,'termination_max_tokens':32,'convergence_words':'</think>','termination_words':'</think>\n\nThe final answer is'}
THETA=1.25
native=None;leadmath=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def key(s):return s['dataset']+':'+str(s['id'])
def stem(s):return hashlib.sha256(key(s).encode()).hexdigest()[:20]
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def choose(logits):
 from lead.generation_utils import apply_sampling_filter
 logits=logits/CONFIG['temperature'];logits=apply_sampling_filter(logits,top_k=20,top_p=.95,min_p=0.)
 return eng.torch.argmax(eng.torch.nn.functional.softmax(logits,dim=-1),dim=-1)
def plain_kw(maximum=1024):
 kw=eng.base.sweep.talr_kwargs(eng.math_ids,maximum,**eng.base.controller(THETA))
 kw.update(lead_refinement_soft_cap=0,lead_refinement_start_offset=0,lead_refinement_soft_mix_lambda=.95,lead_disable_step0_linebreak_mix=True,lead_disable_to_normal_transition=True,lead_force_initial_transition_step1=True,math_ids_tensor=None)
 return kw
class LeanProvider:
 """Same online readout/action as frozen OnlineProvider; no cache/vector hashes."""
 site='input_embedding'
 def __init__(self,s,path):self.sample=s;self.mask_path=path;self.event=None;self.vector=None;self.in_weak=False;self.raw_true=None;self.raw_weak=None;self.hard=None;self.capture_forwards=0
 def __call__(self,step,selected_token_id,raw_soft,hard,transition_step,entropy_proposed,locked_normal,controller_mode):
  assert self.event is None and 2<=step<=9 and transition_step==1 and entropy_proposed and not locked_normal and controller_mode==1
  self.event=step;live=sys._getframe(1).f_locals;prefix=live['all_generated'][0][live['prompt_lens'][0]:]+[selected_token_id]
  self.hard=hard.detach().float().cpu();self.raw_true=raw_soft.detach().float().cpu();main_state=eng.isolation.snapshot();self.in_weak=True;ACTIVE_STATE['branch']='weak'
  try:
   mask=eng.base.sweep.create_mean_mask(self.sample['image'],self.mask_path);wi,_=eng.base.prepare(eng.processor,self.sample,mask,eng.model.device);vectors=[]
   kw=plain_kw(step+1);kw.update(forced_prefix_ids=prefix,trace_soft_vector_collector=vectors,trace_capture_soft_vector_step=step)
   with eng.torch.no_grad():eng.original(eng.model,eng.tokenizer,**wi,**kw)
   assert len(vectors)==1 and vectors[0]['selected_token_id']==selected_token_id
   self.raw_weak=vectors[0]['soft_embedding'];assert eng.torch.equal(vectors[0]['hard_embedding'],self.hard)
  finally:eng.isolation.restore(main_state);self.in_weak=False;ACTIVE_STATE['branch']='main'
  ui=self.raw_true-self.hard;uw=self.raw_weak-self.hard;rho=float(ui.norm());difference=ui-uw;dn=float(difference.norm())
  self.vector=difference*(rho/dn) if dn>1e-12 else eng.torch.zeros_like(difference)
  return {'event':self.event,'vector':self.vector}

def vcd_generate(inputs,state):
 torch=eng.torch;state['branch']='main';original,so=prefill(eng.model,inputs)
 noisy_inputs=dict(inputs);noisy_inputs['pixel_values']=add_diffusion_noise(inputs['pixel_values'],500,12345)
 state['branch']='weak';noisy,sn=prefill(eng.model,noisy_inputs);generated=[]
 for step in range(1024):
  ol=original.logits[:,-1,:];nl=noisy.logits[:,-1,:]
  cutoff=torch.log(torch.tensor(.1,device=ol.device))+ol.max(dim=-1,keepdim=True).values
  allowed=ol>=cutoff;scores=(1+1.)*ol-1.*nl;scores=scores.masked_fill(~allowed,-float('inf'))
  selected=choose(scores);tid=int(selected.item());generated.append(tid)
  if tid==eng.tokenizer.eos_token_id:break
  if step+1<1024:
   state['branch']='main';original,so=advance(eng.model,selected,so)
   state['branch']='weak';noisy,sn=advance(eng.model,selected,sn)
 state['branch']='main';return generated

def timed_generate(s,arm,run,tag):
 inputs,length=eng.base.prepare(eng.processor,s,s['image'],eng.model.device)
 provider=LeanProvider(s,run/'masks'/(tag+'_'+stem(s)+'.png')) if arm=='input_fixed4' else None
 if arm=='cot_hard':out=native['generate_cot'](eng.model,eng.tokenizer,**inputs,**CONFIG);tokens=out[0,length:].tolist()
 elif arm=='full_lead':out=native['generate_lead'](eng.model,eng.tokenizer,**inputs,**CONFIG,**LEAD_KW,math_ids_tensor=leadmath);tokens=out[0,length:].tolist()
 elif arm in ('once_soft','input_fixed4'):
  kw=plain_kw();kw.update(trace_online_residual_provider=provider)
  out=generate_lean_fixed(eng.model,eng.tokenizer,**inputs,**kw);tokens=out[0,length:].tolist()
 else:tokens=vcd_generate(inputs,ACTIVE_STATE)
 text=eng.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
 return tokens,text,provider,int(length),list(inputs['image_grid_thw'].shape)
ACTIVE_STATE=None

def run_case(s,ref,run,index,warmup=False,behavior_check=False):
 global ACTIVE_STATE
 torch=eng.torch;rotation=index%5;order=ARMS[rotation:]+ARMS[:rotation]
 if (index//5)%2:order=tuple(reversed(order))
 values={}
 for arm in order:
  eng.isolation.restore_checked(eng.initial);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  ACTIVE_STATE={'branch':'main'};count={'main':0,'weak':0,'prefill_main':0,'prefill_weak':0};provider_box={'current':None}
  def counter(module,args,kws):
   # The provider itself runs synchronously inside generate; the weak call is
   # identifiable from its prepared pixel input after the first prefill or from
   # a Python stack flag set by weak_forward_wrapper, never a tensor transfer.
   branch=ACTIVE_STATE['branch'];count[branch]+=1
   if kws.get('past_key_values') is None:count['prefill_'+branch]+=1
  handle=eng.model.register_forward_pre_hook(counter,with_kwargs=True)
  torch.cuda.synchronize();t0=time.perf_counter()
  try:
   with torch.no_grad():tokens,text,provider,prompt_tokens,grid_shape=timed_generate(s,arm,run,'warm' if warmup else 'formal')
   torch.cuda.synchronize();wall=time.perf_counter()-t0;peak_allocated=int(torch.cuda.max_memory_allocated());peak_reserved=int(torch.cuda.max_memory_reserved())
  finally:handle.remove()
  expected=ref['arms'].get(arm);matched=None
  if expected is not None:matched=tokens==expected['token_ids'];assert matched,('fresh_full_tokens_differ',key(s),arm)
  if arm=='input_fixed4':assert (provider.event if provider.event is not None else -1)==ref['input_event']
  if arm=='vcd_continuous':assert count['main']==count['weak']==len(tokens)
  elif arm=='input_fixed4':assert count['main']==len(tokens) and count['weak']==(provider.event+1 if provider.event is not None else 0)
  else:assert count['main']==len(tokens) and count['weak']==0
  extra_check=None
  if behavior_check and arm=='full_lead':
   eng.isolation.restore_checked(eng.initial);inputs,length=eng.base.prepare(eng.processor,s,s['image'],eng.model.device)
   from adapters.lead_adapter import generate
   check=generate(eng.model,eng.tokenizer,inputs,variant='original',config=dict(CONFIG,**LEAD_KW),context={'math_ids':leadmath})
   assert check['token_ids']==tokens,('unmodified_official_vs_original_adapter',key(s));extra_check='official_unmodified_exact_original_audit_adapter'
  if behavior_check and arm=='input_fixed4':
   from resident_original import OnlineProvider
   eng.isolation.restore_checked(eng.initial);pr=OnlineProvider(s,run/'masks'/('reference_'+stem(s)+'.png'),'input_embedding')
   check_tokens,_,_=eng.generate(s,cap=0,lam=.95,provider=pr,generator=generate_lean_fixed)
   assert check_tokens==tokens and pr.event==provider.event
   if provider.event is not None:assert torch.equal(pr.actual_vector,provider.vector) and torch.equal(pr.actual_vector.to(eng.model.dtype),provider.vector.to(eng.model.dtype))
   extra_check='lean_provider_exact_frozen_online_action_and_full_tokens'
  values[arm]={'wall_seconds':wall,'peak_cuda_allocated_bytes':peak_allocated,'peak_cuda_reserved_bytes':peak_reserved,'generated_tokens':len(tokens),'prompt_tokens':prompt_tokens,'image_grid_shape':grid_shape,'forward_calls':count,'event':provider.event if provider else None,'feedback_count':min(4,max(0,len(tokens)-1-provider.event)) if provider and provider.event is not None else 0,'generated_token_ids':tokens,'text':text,'historical_complete_tokens_exact':matched,'historical_reference_available':expected is not None,'stop_reason':'eos' if tokens[-1]==eng.tokenizer.eos_token_id else ('horizon' if len(tokens)==1024 else 'native_LEAD_answer_budget'),'behavior_check':extra_check}
  print('ARM_COMPLETE',key(s),arm,round(wall,4),'tokens',len(tokens),'warmup',warmup,flush=True)
 return {'sample':s,'arms':values,'order':list(order),'warmup':warmup,'labels_read':False,'fresh_all_arms':True}

def main():
 global native,leadmath
 m=json.loads((E/'manifest.json').read_text());mh=sha(E/'manifest.json')
 for f,h in m['files'].items():assert sha(E/f)==h,f
 for f,h in m['external_files'].items():assert sha(f)==h,f
 for f,h in m['checkpoint_files'].items():assert sha(Path(m['checkpoint']['path'])/f)==h,f
 refs={key(x['sample']):x for x in read_lines(E/'references.jsonl')}
 for v in refs.values():
  assert sha(v['sample']['image'])==v['sample']['image_sha256']
  for a,x in v['arms'].items():assert sha(x['source'])==x['source_sha256']
 run=O/(os.environ.get('SLURM_JOB_ID','manual')+'_'+str(time.time_ns()));run.mkdir(parents=True);(run/'masks').mkdir()
 t=time.perf_counter();eng.init(m['checkpoint'],run,1.25);eng.torch.cuda.synchronize();load_seconds=time.perf_counter()-t
 native=_compile_official();leadmath=eng.torch.tensor(sorted(official_math_ids(eng.tokenizer)),device=eng.model.device,dtype=eng.torch.long)
 # Same frozen OnlineProvider reference is imported only for untimed engineering checks.
 save(run/'LOAD.json',{'model_load_seconds':load_seconds,'runtime_identity':json.loads((run/'runtime_identity.json').read_text()),'manifest_sha256':mh})
 for i,s in enumerate(read_lines(E/'warmup4.jsonl')):save(run/('WARMUP_'+stem(s)+'.json'),run_case(s,refs[key(s)],run,i,True,behavior_check=i<2))
 save(run/'WARMUP_PASS.json',{'passed':True,'n_samples':4,'n_arms':20,'behavior_checked_first2':True,'tokens_exact_to_available_history':True,'manifest_sha256':mh})
 rows=[]
 for i,s in enumerate(read_lines(E/'public32.jsonl')):
  row=run_case(s,refs[key(s)],run,i);save(run/(stem(s)+'.json'),row);rows.append(row)
 assert len(rows)==32;save(run/'RESULTS.json',{'n':32,'rows':rows,'manifest_sha256':mh,'labels_read':False})
 save(run/'ACCEPTANCE.json',{'passed':True,'n':32,'fresh_formal_generations':160,'warmup_generations':20,'all_available_historical_full_tokens_exact':True,'results_sha256':sha(run/'RESULTS.json'),'manifest_sha256':mh,'labels_read':False})
 save(O/'COMPLETE.json',{'run':str(run),'acceptance_sha256':sha(run/'ACCEPTANCE.json'),'manifest_sha256':mh});print('COMPLETE_5METHODS',run,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:print('FAILED',traceback.format_exc(),flush=True);raise
