"""Exact final-clean capture and generation functions, new model initialization only."""
import gc,hashlib,json,os,sys,time
from pathlib import Path
E=Path(__file__).resolve().parent;R=Path('/home/bingxing2/home/scx9ftv/gushuo')
weak_observer=None
SOFT_START=True
sys.path.insert(0,str(E/'source'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def stem(s):return hashlib.sha256((s['dataset']+':'+str(s['id'])).encode()).hexdigest()[:20]
FIELDS=('step','token_id','raw_entropy','raw_top1_prob','cur_ref_entropy','mode','mode_before','to_soft','to_normal','locked_normal','route_suppressed_by','lead_refinement_active','lead_refinement_candidate','lead_refinement_entropy_proposed','lead_refinement_count_before','lead_initial_transition_step','lead_refinement_elapsed','lead_refinement_start_offset','lead_refinement_window','lead_refinement_soft_cap','lead_refinement_soft_mix_lambda','lead_answer_zone_lock_enabled','lead_refinement_reference_ratio','lead_refinement_entropy_threshold','route_override_active')

def eligible(row):
 tau=row.get('lead_initial_transition_step')
 return tau is not None and 0<row['step']-tau<=8 and bool(row['lead_refinement_entropy_proposed']) and not row['locked_normal']

def assert_no_later_switch(row,controller_mode):
 # row.mode reports routed feedback (is_soft), while controller mode remains hard.
 assert controller_mode==1 and row['mode_before']=='normal' and not row['to_soft'] and not row['to_normal'],('later FullLEAD switching must remain disabled',row['step'],controller_mode,row['mode_before'],row['to_soft'],row['to_normal'])
 expected_route='soft' if row['lead_refinement_active'] else 'normal'
 assert row['mode']==expected_route,('unexpected feedback route after handoff',row['step'],row['mode'],expected_route)
class Stop(Exception):pass
def generate(s,cap=2,trace_on=True,plain_init=True,audit=False,stop_first=False,prefix=None,event=-1,action=None,generator=None,lam=1.,provider=None,feedback_count=4):
 isolation.restore_checked(initial);inputs,length=base.prepare(processor,s,s['image'],model.device)
 ih=hashlib.sha256(json.dumps(inputs['input_ids'].detach().cpu().tolist()).encode()).hexdigest();key=(s['dataset'],str(s['id']))
 if key in input_hashes:assert input_hashes[key]==ih
 input_hashes[key]=ih
 kw=base.sweep.talr_kwargs(math_ids,1024,**base.controller(threshold));kw.update(lead_refinement_soft_cap=cap,lead_refinement_start_offset=0,lead_refinement_soft_mix_lambda=lam)
 if provider is not None:kw['trace_online_residual_provider']=provider
 if plain_init:kw.update(lead_disable_step0_linebreak_mix=True,lead_disable_to_normal_transition=True,lead_force_initial_transition_step1=True,math_ids_tensor=None,lead_disable_step0_soft=not SOFT_START)
 frozen=dict(lead_refinement_window=8,lead_refinement_soft_mix_lambda=lam,lead_refinement_entropy_threshold=threshold,lead_refinement_threshold_mode='fixed',lead_disable_answer_zone_lock=True,do_sample=False,max_new_tokens=1024)
 assert all(kw[k]==v for k,v in frozen.items())
 assert kw['lead_initial_transition_with_refinement'] and not kw.get('lead_initial_transition_only',False)
 assert not kw.get('lead_soft_veto_on_diffuse',False) and not kw.get('lead_format_cooldown',False) and not kw.get('lead_soft_quota_ratio',0)
 if prefix is not None:kw['forced_prefix_ids']=prefix
 if action is not None:
  assert event>=2;kw.update(trace_route_override_step=event,trace_route_override_kind='method_plus_external_residual',trace_route_override_mix_lambda=1.,trace_route_override_duration=feedback_count,trace_external_route_vector=action)
 pending=[];consumed=[];hashes=[];effective=[];initial_geometry=[];residual_geometry=[]
 class Trace(list):
  def append(self,row):
   super().append({k:row[k] for k in FIELDS if k in row})
   live=sys._getframe(1).f_locals;step=int(row['step']);hard=live['E'][live['next_tokens']];feedback=live['last_emb']
   before=live['route_origin'] if row.get('route_override_active') else feedback
   tau=row.get('lead_initial_transition_step')
   self[-1]['residual_eligible']=eligible(row)
   self[-1]['residual_elapsed']=None if tau is None else step-tau
   if plain_init and step==0:
    expected=torch.matmul(live['probs_original'].to(live['E'].dtype),live['E']) if SOFT_START else hard;assert torch.equal(before,expected),'plain initial raw soft changed'
   elif row.get('lead_refinement_active'):
    assert cap>0 and torch.equal(before,lam*live['soft_emb']+(1.-lam)*hard),'main refinement mixture mismatch'
   elif (plain_init and step>=1) or (tau is not None and step>tau):
    assert torch.equal(before,hard),'post-initialization non-refinement feedback must be exact hard'
   if plain_init and step==1:assert tau==1 and torch.equal(before,hard)
   if tau is not None and step>tau:
    assert_no_later_switch(row,int(live['mode'][0].item()))
   if cap==0:assert not row['lead_refinement_active']
   if row.get('route_override_active'):
    assert torch.equal(feedback,before+live['external_emb'])
    expected_action=action if action is not None else provider.vector
    assert torch.equal(live['external_emb'][0].detach().cpu(),expected_action.to(model.dtype).cpu())
    if not torch.equal(before,feedback):effective.append(step)
    rg=dict(step=step,base_feedback_sha256=tensor_sha(before),cast_action_norm=float(live['external_emb'].float().norm()),actual_delta_norm=float((feedback.float()-before.float()).norm()),actual_noop=bool(torch.equal(feedback,before)))
    self[-1]['residual_geometry']=rg;residual_geometry.append(rg)
   if row.get('lead_refinement_active'):
    fg={}
    for label,value in [('soft',live['soft_emb']),('refinement',before),('final',feedback)]:
     fg[label+'_delta_norm']=float(torch.linalg.vector_norm(value.float()-hard.float()).item());fg[label+'_bitwise_noop']=bool(torch.equal(value,hard))
    self[-1]['feedback_geometry']=fg
   if step<=1:initial_geometry.append(dict(step=step,exact_hard=bool(torch.equal(feedback,hard)),delta_norm=float((feedback.float()-hard.float()).norm()),feedback_sha256=tensor_sha(feedback)))
   assert torch.isfinite(feedback).all()
   pending.append((step,feedback.detach().clone().unsqueeze(1)))
   if stop_first and eligible(row):raise Stop()
 trace=Trace()
 if trace_on:kw.update(token_trace=trace,trace_topk=0)
 def consumer(module,pos,kws):
  if provider is not None and provider.in_weak:return
  value=kws.get('inputs_embeds',kws.get('input_ids'));hashes.append(tensor_hash(value))
  if pending:
   step,expected=pending.pop(0);assert kws.get('inputs_embeds') is not None and torch.equal(kws['inputs_embeds'],expected),'actual next-forward input mismatch';consumed.append(step)
 handle=model.register_forward_pre_hook(consumer,with_kwargs=True);t=time.monotonic();stopped=False;out=None
 try:
  with torch.no_grad():out=(generator or original)(model,tokenizer,**inputs,**kw)[0]
 except Stop:stopped=True
 finally:handle.remove()
 tokens=[r['token_id'] for r in trace] if trace_on else [int(x) for x in out[length:].tolist()]
 assert 0<len(tokens)<=1024 and len(tokens)==len(hashes)
 if not stopped:assert tokens==[int(x) for x in out[length:].tolist()]
 if prefix is not None:assert tokens[:len(prefix)]==prefix
 if trace_on:
  assert consumed==list(range(len(tokens)-1))
  assert [r['step'] for r in trace]==list(range(len(tokens)))
  rs=[r for r in trace if r.get('lead_refinement_active')];assert len(rs)<=cap
  for r in rs:
   assert eligible(r)
   assert r['raw_entropy']>threshold and r['raw_entropy']>r['cur_ref_entropy']
  if len(tokens)>1 and plain_init:
   assert {r['lead_initial_transition_step'] for r in trace if r['lead_initial_transition_step'] is not None}=={1}
   assert all(r['cur_ref_entropy']==trace[1]['raw_entropy'] for r in trace[1:]),'reference must stay at step1 raw entropy'
  if provider is not None and provider.event is not None:event=provider.event
  if action is not None or (provider is not None and provider.event is not None):assert [r['step'] for r in trace if r.get('route_override_active')]==list(range(event,min(event+feedback_count,len(tokens))))
 return tokens,list(trace),dict(seconds=time.monotonic()-t,consumed=consumed,input_ids_sha256=ih,hashes=hashes,stopped=stopped,effective_consumed=[i for i in effective if i<len(tokens)-1],initial_geometry=initial_geometry,residual_geometry=residual_geometry)
def capture(s,image,prefix,event,plain_init=True,lam=1.):
 isolation.restore_checked(initial);inputs,_=base.prepare(processor,s,image,model.device);vectors=[];pending=[];consumed=[]
 class CaptureTrace(list):
  def append(self,row):
   live=sys._getframe(1).f_locals;step=int(row['step']);hard=live['E'][live['next_tokens']]
   if weak_observer is not None:weak_observer(sys._getframe(1))
   if plain_init and step==0:assert torch.equal(live['last_emb'],torch.matmul(live['probs_original'].to(live['E'].dtype),live['E']) if SOFT_START else hard)
   if plain_init and step==1:assert torch.equal(live['last_emb'],hard) and row['lead_initial_transition_step']==1
   if row.get('lead_refinement_active'):
    assert torch.equal(live['last_emb'],lam*live['soft_emb']+(1.-lam)*hard)
    if step<event:pending.append((step,live['last_emb'].detach().clone().unsqueeze(1)))
   super().append(row)
 ct=CaptureTrace()
 kw=base.sweep.talr_kwargs(math_ids,event+1,**base.controller(threshold))
 kw.update(lead_refinement_soft_cap=0,lead_refinement_soft_mix_lambda=lam,lead_refinement_start_offset=0,forced_prefix_ids=prefix,trace_soft_vector_collector=vectors,trace_capture_soft_vector_step=event,token_trace=ct,trace_topk=0)
 if plain_init:kw.update(lead_disable_step0_linebreak_mix=True,lead_disable_to_normal_transition=True,lead_force_initial_transition_step1=True,math_ids_tensor=None,lead_disable_step0_soft=not SOFT_START)
 assert kw['lead_refinement_window']==8 and kw['lead_refinement_soft_cap']==0 and kw['lead_refinement_soft_mix_lambda']==lam and 'lead_refinement_fixed_slots' not in kw
 def consumer(module,pos,kws):
  if pending:
   step,value=pending.pop(0);assert torch.equal(kws['inputs_embeds'],value);consumed.append(step)
 handle=model.register_forward_pre_hook(consumer,with_kwargs=True);t=time.monotonic()
 try:
  with torch.no_grad():original(model,tokenizer,**inputs,**kw)
 finally:handle.remove()
 assert not pending
 assert len(vectors)==1 and [r['token_id'] for r in ct]==prefix and [r['step'] for r in ct]==list(range(event+1))
 v=vectors[0];assert v['step']==event and v['selected_token_id']==prefix[event]
 if plain_init:assert {r['lead_initial_transition_step'] for r in ct if r['lead_initial_transition_step'] is not None}=={1}
 audit=dict(seconds=time.monotonic()-t,horizon=event+1,start_offset=0,forced_prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),handoff_steps=sorted({r['lead_initial_transition_step'] for r in ct if r['lead_initial_transition_step'] is not None}),refinement_steps=[r['step'] for r in ct if r.get('lead_refinement_active')],selected_token_id=v['selected_token_id'],soft_pre_feedback_mix=True,main_kv_used=False,initialization='plain' if plain_init else 'original',window=8,soft_cap=0,refinement_lambda=lam,consumed_refinement_steps=consumed)
 return v,audit

def native_plain_once(s):
 # Engineering reference for one raw-soft step, followed by native discrete tokens.
 isolation.restore_checked(initial);inputs,length=base.prepare(processor,s,s['image'],model.device);hashes=[]
 kw=dict(temperature=.6,top_p=.95,top_k=20,min_p=0.,max_new_tokens=1024,do_sample=False,trace_route_override_step=0,trace_route_override_duration=1,trace_route_override_kind='raw_soft')
 def consume(module,pos,kws):
  if not hashes:value=kws['input_ids']
  elif kws.get('inputs_embeds') is not None:value=kws['inputs_embeds']
  else:value=weight[kws['input_ids']]
  hashes.append(tensor_hash(value))
 handle=model.register_forward_pre_hook(consume,with_kwargs=True)
 try:
  with torch.no_grad():out=generate_cot(model,tokenizer,**inputs,**kw)[0]
 finally:handle.remove()
 return [int(x) for x in out[length:].tolist()],hashes

def init(spec,folder,theta):
 global torch,base,model,processor,tokenizer,math_ids,weight,isolation,initial,original,input_hashes,threshold,tensor_sha,tensor_hash,generate_cot
 import torch,base
 from scale_model_loader import load_model
 from state_isolation import StateIsolation,tensor_sha
 from route_tape import tensor_hash
 from lead.generation_utils import generate_cot
 threshold=theta;torch.set_num_threads(4)
 model,processor,tokenizer,math_ids=load_model(spec,folder)
 weight=model.get_input_embeddings().weight
 assert type(model.get_output_embeddings()).__name__!='LastTokenHead'
 isolation=StateIsolation(model);initial=isolation.snapshot();original=base.generate_lead;input_hashes={}
