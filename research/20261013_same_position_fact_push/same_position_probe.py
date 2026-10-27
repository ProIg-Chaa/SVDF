# Local-window feedback interventions on an unchanged Once teacher-forced history.
import contextlib,hashlib,json,math,time
class FixedEndpoint(Exception):pass
class SamePositionRuntime:
 def __init__(self,model,processor,tokenizer,math_ids):
  from state_isolation import StateIsolation
  self.model=model;self.processor=processor;self.tokenizer=tokenizer;self.math_ids=math_ids
  self.isolation=StateIsolation(model);self.initial=self.isolation.snapshot();self.weight=model.get_input_embeddings().weight
 def _generate(self,sample,forced):
  import base,torch
  self.isolation.restore_checked(self.initial);inputs,_=base.prepare(self.processor,sample,sample['image'],self.model.device)
  kw=base.sweep.talr_kwargs(self.math_ids,1024,**base.controller(1.25))
  kw.update(lead_refinement_soft_cap=0,lead_refinement_start_offset=0,lead_refinement_soft_mix_lambda=1.,lead_disable_step0_linebreak_mix=True,lead_disable_to_normal_transition=True,lead_force_initial_transition_step1=True,math_ids_tensor=None,forced_prefix_ids=forced)
  assert kw['lead_initial_transition_with_refinement'] and kw['lead_refinement_window']==8 and kw['lead_disable_answer_zone_lock']
  assert not kw.get('lead_disable_simple_visual_anchor',False) and not kw.get('lead_early_visual_anchor',False) and not kw.get('lead_format_cooldown',False) and not kw.get('lead_soft_quota_ratio',0)
  assert not kw.get('lead_initial_transition_only',False) and kw['do_sample'] is False and kw['max_new_tokens']==1024
  with torch.no_grad():base.generate_lead(self.model,self.tokenizer,**inputs,**kw)
 def _run(self,sample,forced,last,slots=(),mode='hard',action=None,multiplier=None,read_index=None,expected_hashes=None,capture=False):
  import torch
  import torch.nn.functional as F
  from route_tape import tensor_hash
  from attention_boost import VisualAttentionBoost
  assert mode in ('hard','soft','svdf','attention') and 0<=last<1024
  assert self.model.config.image_token_id not in forced and self.model.config.eos_token_id not in forced
  slots=tuple(slots)
  if slots:assert len(slots) in (1,4) and slots==tuple(range(slots[0],slots[0]+len(slots))) and slots[0]>=1 and slots[-1]+1<=last
  assert (action is not None)==(mode=='svdf') and (multiplier is not None)==(mode=='attention')
  assert not capture or (mode=='hard' and not slots and read_index is None)
  step=-1;previous_soft=None;previous_output_sha=None;hashes=[];geometry=[];raws=[];soft_vectors=[];collected=None
  modules=[self.model]+[x.self_attn for x in self.model.model.layers];hooks_before=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules];original_sdpa=F.scaled_dot_product_attention
  applied=None if action is None else action.to(device=self.weight.device,dtype=self.weight.dtype).reshape(1,1,-1)
  def before(module,args,kw):
   nonlocal step
   step+=1;assert step<=last;value=kw.get('inputs_embeds',kw.get('input_ids'));assert value is not None
   if step==0:assert kw.get('past_key_values') is None and kw.get('input_ids') is not None
   else:
    assert kw.get('inputs_embeds') is not None and kw.get('input_ids') is None
    hard=self.weight[forced[step-1]].detach().reshape(1,1,-1);expected=previous_soft if step==1 else hard
    assert expected is not None and torch.equal(value,expected),('source feedback history changed',step)
    if step-1 in slots:
     incoming=value
     if mode=='soft':value=previous_soft;assert value.dtype==hard.dtype and value.shape==hard.shape
     elif mode=='svdf':value=hard+applied
     else:assert torch.equal(value,hard)
     kw=dict(kw);kw['inputs_embeds']=value;delta=value.float()-hard.float()
     g=dict(forward_step=step,feedback_slot=step-1,previous_raw_logits_sha256=previous_output_sha,token_id=forced[step-1],mode=mode,hard_input_sha256=tensor_hash(hard),source_input_sha256=tensor_hash(incoming),actual_input_sha256=tensor_hash(value),actual_delta_norm=float(delta.norm()),actual_noop=bool(torch.equal(value,hard)),actual_input_norm=float(value.float().norm()),raw_soft_input_norm=float(previous_soft.float().norm()),raw_soft_hard_delta_norm=float((previous_soft.float()-hard.float()).norm()))
     if applied is not None:g['cast_action_norm']=float(applied.float().norm())
     geometry.append(g);soft_vectors.append(dict(forward_step=step,feedback_slot=step-1,soft=previous_soft.detach().contiguous().cpu(),hard=hard.detach().contiguous().cpu(),actual=value.detach().contiguous().cpu()))
   hashes.append(tensor_hash(value))
   if expected_hashes is not None:
    boundary=slots[0] if slots else (read_index if read_index is not None else last)
    if step<=boundary:assert hashes[-1]==expected_hashes[step],('common historical input changed',step)
   return args,kw
  def after(module,args,kw,out):
   nonlocal previous_soft,previous_output_sha,collected
   if not (step==0 or step in slots or (capture and step==last) or (read_index is not None and step>=read_index)):return
   live=out.logits[:,-1,:].detach();assert live.shape==(1,self.model.config.vocab_size) and torch.isfinite(live).all()
   raw=live[0].contiguous().cpu();assert torch.equal(raw.to(torch.bfloat16).to(raw.dtype),raw),'final raw logits not losslessly BF16 encodable'
   stored=raw.to(torch.bfloat16);previous_output_sha=tensor_hash(stored);p=F.softmax(live,dim=-1)
   assert p.dtype==self.weight.dtype,'ordinarysoft raw-p dtype differs from legacy source'
   previous_soft=torch.matmul(p,self.weight).reshape(1,1,-1);assert torch.isfinite(previous_soft).all()
   if capture and step==last:
    hard=self.weight[forced[step]].detach();collected=dict(p=p[0].float().cpu(),s=previous_soft[0,0].float().cpu(),h=hard.float().cpu(),selected_token_id=forced[step],capture_slot=step,raw_logits=stored,raw_probability_dtype=str(p.dtype),soft_embedding_dtype=str(previous_soft.dtype),raw_logits_sha256=previous_output_sha)
   if read_index is not None and step>=read_index:raws.append(stored)
   if step==last:raise FixedEndpoint()
  guard=None if multiplier is None else VisualAttentionBoost(self.model,slots[0],multiplier,duration=len(slots))
  handles=[self.model.register_forward_pre_hook(before,with_kwargs=True),self.model.register_forward_hook(after,with_kwargs=True)];stopped=False;t=time.monotonic()
  try:
   with guard if guard is not None else contextlib.nullcontext():
    try:self._generate(sample,forced)
    except FixedEndpoint:stopped=True
  finally:
   for h in reversed(handles):h.remove()
  assert stopped and step==last and len(hashes)==last+1 and F.scaled_dot_product_attention is original_sdpa
  assert [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules]==hooks_before
  assert [g['feedback_slot'] for g in geometry]==list(slots)
  obs=None if guard is None else guard.validate(last+1)
  return dict(actual_input_hashes=hashes,feedback_geometry=geometry,attention_observation=obs,consumed_feedback_slots=list(slots),consumed_forward_steps=[x+1 for x in slots],hook_cleanup_exact=True,fresh_cache_at_prefill=True,seconds=time.monotonic()-t),raws,collected,soft_vectors
 def capture(self,sample,prefix,slot,expected_hashes=None):
  assert len(prefix)>=slot+1 and slot>=1
  meta,_,v,_=self._run(sample,prefix[:slot+1],slot,expected_hashes=expected_hashes,capture=True);assert v is not None;v['audit']=meta;return v
 def read(self,sample,prefix,candidate,duration,mode='hard',action=None,multiplier=None,expected_hashes=None):
  import torch
  k=len(prefix);assert duration in (1,4) and k-duration>=1;ids=candidate['token_ids'];assert ids and 1<=candidate['word_token_count']<=len(ids)
  slots=tuple(range(k-duration,k));forced=prefix+ids[:-1]
  meta,raw,_,vectors=self._run(sample,forced,k+len(ids)-1,slots=slots,mode=mode,action=action,multiplier=multiplier,read_index=k,expected_hashes=expected_hashes)
  assert len(raw)==len(ids);metrics=[]
  for offset,(logits,tid) in enumerate(zip(raw,ids)):
   lp=torch.log_softmax(logits.double(),dim=-1);lse=float(torch.logsumexp(logits.double(),dim=-1));target=float(lp[tid]);assert abs(target-(float(logits[tid])-lse))<1e-10
   metrics.append(dict(forward_step=k+offset,predicted_token_index=k+offset,candidate_offset=offset,target_token_id=tid,raw_logit=float(logits[tid]),log_probability=target,probability=math.exp(target),rank=int((logits>logits[tid]).sum())+1,top1_token_id=int(logits.argmax()),raw_logits_dtype=str(logits.dtype),logsumexp=lse,raw_logits_sha256=hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest()))
  phrase=sum(x['log_probability'] for x in metrics);word=sum(x['log_probability'] for x in metrics[:candidate['word_token_count']])
  meta.update(text=candidate['text'],word_text=candidate['word_text'],role=candidate['role'],token_ids=ids,word_token_count=candidate['word_token_count'],first_token_probability=metrics[0]['probability'],word_log_probability=word,word_probability=math.exp(word),phrase_log_probability=phrase,phrase_probability=math.exp(phrase),per_token=metrics,prefix_token_ids=prefix,fixed_prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),forced_candidate_history_sha256=hashlib.sha256(json.dumps(forced).encode()).hexdigest(),ordinary_soft_is_raw_probability_embedding_replacement=True,readout_after_all_updates_consumed=True,candidate_suffix_feedback='hard')
  return meta,torch.stack(raw),vectors

def frozen_difference(original,weakened):
 import torch
 from state_isolation import tensor_sha
 assert original['capture_slot']==weakened['capture_slot'] and original['selected_token_id']==weakened['selected_token_id'] and torch.equal(original['h'],weakened['h'])
 si,sw,h=original['s'].float(),weakened['s'].float(),original['h'].float();ui,uw=si-h,sw-h;rho=float(ui.norm());delta=ui-uw;norm=float(delta.norm());action=delta*(rho/norm) if norm>1e-12 else torch.zeros_like(delta)
 assert torch.isfinite(action).all()
 if norm>1e-12:assert abs(float(action.norm())-rho)<=max(1e-5,rho*1e-4)
 return action,dict(capture_slot=original['capture_slot'],rho=rho,difference_norm=norm,action_norm=float(action.norm()),action_sha256=tensor_sha(action),mapping='FP32(sI-h)-(sW-h), normalized to ||sI-h||; BF16 soft embeddings from raw p',no_candidate_labels_used=True,fixed_once_per_window=True)
