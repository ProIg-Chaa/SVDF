import contextlib,math
import torch
import torch.nn.functional as F
from common_decode import prefill,advance,select_token
class PAIAttention:
 # Official last-query abs-logit operation; native Qwen rotary/cache outside patch.
 def __init__(self,model,alpha=.2,start=2,end=28):
  self.model=model;self.alpha=float(alpha);self.start=start;self.end=end;self.active=None;self.negative=False;self.positions=None;self.handles=[];self.stats=dict(calls=0,positive_calls=0,negative_calls=0,modified_queries=0,nonzero_delta=0,visual_mass_before_sum=0.,visual_mass_after_sum=0.)
 def model_pre(self,module,args,kw):
  if not self.negative and self.positions is None:
   ids=kw.get('input_ids');assert ids is not None;self.positions=torch.where(ids[0]==self.model.config.image_token_id)[0];assert self.positions.numel()>0
 def layer_pre(self,i):
  def fn(module,args,kw):assert self.active is None;self.active=i
  return fn
 def layer_post(self,module,args,kw,out):self.active=None
 def __enter__(self):
  self.orig=F.scaled_dot_product_attention;self.handles.append(self.model.register_forward_pre_hook(self.model_pre,with_kwargs=True));assert len(self.model.model.layers)==28
  for i,l in enumerate(self.model.model.layers):self.handles.extend([l.self_attn.register_forward_pre_hook(self.layer_pre(i),with_kwargs=True),l.self_attn.register_forward_hook(self.layer_post,with_kwargs=True)])
  def wrapped(*a,**kw):
   if self.active is None or not self.start<=self.active<self.end:return self.orig(*a,**kw)
   self.stats['calls']+=1
   if self.negative:self.stats['negative_calls']+=1;return self.orig(*a,**kw)
   self.stats['positive_calls']+=1
   if self.alpha==0.:return self.orig(*a,**kw)
   q,k,v=a[:3];mask=kw.get('attn_mask',a[3] if len(a)>3 else None);dropout=kw.get('dropout_p',a[4] if len(a)>4 else 0.);causal=kw.get('is_causal',a[5] if len(a)>5 else False)
   assert dropout==0. and (not causal or q.shape[-2]>1) and q.shape[0]==1;assert int(self.positions.max())<k.shape[-2]
   scale=kw.get('scale');scores=torch.matmul(q[..., -1:, :],k.transpose(-2,-1));scores=scores/math.sqrt(q.shape[-1]) if scale is None else scores*scale
   if mask is not None:
    lm=mask[..., -1:, :] if mask.shape[-2]>1 else mask
    if lm.dtype==torch.bool:scores=scores.masked_fill(~lm,-float('inf'))
    else:scores=scores+lm
   pos=self.positions.to(q.device);target=scores[...,pos];assert torch.isfinite(target).all();delta=target.abs()*self.alpha;before=F.softmax(scores,dim=-1,dtype=torch.float32)[...,pos].sum(-1).mean();altered=scores.clone();altered[...,pos]=target+delta
   weight=F.softmax(altered,dim=-1,dtype=torch.float32).to(q.dtype);after=weight[...,pos].sum(-1).float().mean();last=torch.matmul(weight,v);assert torch.isfinite(last).all()
   self.stats['modified_queries']+=1;self.stats['nonzero_delta']+=int(bool((delta!=0).any()));self.stats['visual_mass_before_sum']+=float(before);self.stats['visual_mass_after_sum']+=float(after)
   if q.shape[-2]==1:return last
   native=self.orig(*a,**kw);native[..., -1:, :]=last;return native
  F.scaled_dot_product_attention=wrapped;return self
 def __exit__(self,*unused):
  F.scaled_dot_product_attention=self.orig
  for h in self.handles:h.remove()
  self.handles=[];self.active=None
 @contextlib.contextmanager
 def no_image(self):
  old=self.negative;self.negative=True
  try:yield
  finally:self.negative=old
 def report(self):
  x=dict(self.stats);n=x['modified_queries'];x['mean_visual_mass_before']=x['visual_mass_before_sum']/n if n else None;x['mean_visual_mass_after']=x['visual_mass_after_sum']/n if n else None;x.update(alpha=self.alpha,start_layer=self.start,end_layer_exclusive=self.end,all_steps=True,last_query_only=True,negative_attention_disabled=True,image_key_count=int(self.positions.numel()));return x

def without_image(model,inputs):
 ids=inputs['input_ids'];cfg=model.config;start=int(getattr(cfg,'vision_start_token_id',151652));end=int(getattr(cfg,'vision_end_token_id',151653));image=int(cfg.image_token_id);starts=torch.where(ids[0]==start)[0];ends=torch.where(ids[0]==end)[0];assert starts.numel()==ends.numel()==1
 lo=int(starts[0]);hi=int(ends[0]);assert lo<hi and bool((ids[0,lo+1:hi]==image).all());keep=torch.ones(ids.shape[1],device=ids.device,dtype=torch.bool);keep[lo:hi+1]=False;out=dict(input_ids=ids[:,keep].clone())
 if inputs.get('attention_mask') is not None:out['attention_mask']=inputs['attention_mask'][:,keep].clone()
 assert not bool((out['input_ids']==image).any()) and out['input_ids'].shape[1]>0
 return out,dict(removed_tokens=hi-lo+1,main_prompt_length=ids.shape[1],no_image_prompt_length=out['input_ids'].shape[1],same_text_tokens=True,removed_entire_vision_block=True)
def rope_snapshot(model):return model.rope_deltas.detach().clone() if model.rope_deltas is not None else None
def branch_prefill(model,inputs):
 model.rope_deltas=None;out,state=prefill(model,inputs);state['rope_deltas']=rope_snapshot(model);return out,state
def branch_advance(model,token,state):
 model.rope_deltas=state['rope_deltas'].clone() if state['rope_deltas'] is not None else None;out,ns=advance(model,token,state);ns['rope_deltas']=rope_snapshot(model);return out,ns
@torch.inference_mode()
def generate(model,tok,inputs,*,alpha=.2,gamma=1.1,beta=.1,maximum=1024,config=None):
 config=dict(config or {});outno=None;sn=None;geom=None;tokens=[];trace=[];changed=0;forwards=0;cfg_nonzero=0
 with PAIAttention(model,alpha) as att:
  out,st=branch_prefill(model,inputs);forwards+=1
  if gamma!=1.:
   ni,geom=without_image(model,inputs)
   with att.no_image():outno,sn=branch_prefill(model,ni)
   forwards+=1;assert st['past_key_values'] is not sn['past_key_values']
  for step in range(maximum):
   logits=out.logits[:,-1,:];assert torch.isfinite(logits).all();base,_,_=select_token(logits,config)
   if gamma==1.:scores=F.log_softmax(logits,dim=-1);allowed=torch.ones_like(scores,dtype=torch.bool)
   else:
    no=outno.logits[:,-1,:];assert torch.isfinite(no).all();lp=F.log_softmax(logits,dim=-1);ln=F.log_softmax(no,dim=-1);cutoff=torch.log(torch.tensor(beta,device=logits.device))+lp.max(-1,keepdim=True).values;allowed=lp>=cutoff;scores=(gamma*(lp-ln)+ln).masked_fill(~allowed,-float('inf'));cfg_nonzero+=int(bool(((scores-lp)[allowed]!=0).any()))
   assert bool(allowed.any()) and torch.isfinite(scores[allowed]).all();token,_,_=select_token(scores,config);ti=int(token);assert bool(allowed[0,ti]);tokens.append(ti);changed+=int(ti!=int(base))
   if step<16:trace.append(dict(step=step,token=ti,original_top1=int(base),cfg_changed=ti!=int(base),main_cache=int(st['past_key_values'].get_seq_length()),no_cache=int(sn['past_key_values'].get_seq_length()) if sn else None,allowed=int(allowed.sum()),main_rope=st['rope_deltas'].tolist() if st['rope_deltas'] is not None else None,no_rope=sn['rope_deltas'].tolist() if sn and sn['rope_deltas'] is not None else None))
   if ti==tok.eos_token_id:break
   if step+1<maximum:
    out,st=branch_advance(model,token,st);forwards+=1
    if gamma!=1.:
     with att.no_image():outno,sn=branch_advance(model,token,sn)
     forwards+=1;assert st['past_key_values'] is not sn['past_key_values'];assert int(st['past_key_values'].get_seq_length())-geom['main_prompt_length']==int(sn['past_key_values'].get_seq_length())-geom['no_image_prompt_length']==step+1
  stats=att.report()
 return dict(token_ids=tokens,text=tok.decode(tokens,skip_special_tokens=False),n_tokens=len(tokens),attention=stats,metadata=dict(method='PAI-R1 adaptation (both components)',alpha=alpha,gamma=gamma,beta=beta,initialization='ordinary hard',maximum=maximum,forward_calls=forwards,cfg_changed_token_count=changed,cfg_nonzero_steps=cfg_nonzero,negative=geom,cache_isolated=True,branch_rope_isolated=True,trace_first16=trace,no_gold=True))
