"""Time-gated PAI attention module; no classifier-free guidance or second branch."""
import math
import torch
import torch.nn.functional as F

class PAITemporalAttention:
 def __init__(self,model,start_slot,alpha=.2,duration=4,start_layer=2,end_layer=28):
  self.model=model;self.start_slot=int(start_slot);self.alpha=float(alpha);self.duration=int(duration)
  self.start_layer=int(start_layer);self.end_layer=int(end_layer)
  assert self.alpha in (0.,.2) and self.duration==4 and self.start_slot>=1
  assert self.start_layer==2 and self.end_layer==28
  self.step=-1;self.active=None;self.positions=None;self.records=[];self.handles=[];self.images_seen=0
 def _model_pre(self,module,args,kw):
  self.step+=1
  if self.step==0:
   ids=kw.get('input_ids');assert ids is not None and kw.get('past_key_values') is None
   self.positions=torch.where(ids[0]==self.model.config.image_token_id)[0]
   assert self.positions.numel()>0
   self.prompt_length=int(ids.shape[1]);self.images_seen+=1
 def _layer_pre(self,i):
  def hook(module,args,kw):assert self.active is None;self.active=i
  return hook
 def _layer_post(self,module,args,kw,out):self.active=None
 def __enter__(self):
  self.orig=F.scaled_dot_product_attention
  assert len(self.model.model.layers)==28
  self.handles.append(self.model.register_forward_pre_hook(self._model_pre,with_kwargs=True))
  for i,l in enumerate(self.model.model.layers):
   self.handles.extend([l.self_attn.register_forward_pre_hook(self._layer_pre(i),with_kwargs=True),l.self_attn.register_forward_hook(self._layer_post,with_kwargs=True)])
  def wrapped(*a,**kw):
   enabled=(self.active is not None and self.start_layer<=self.active<self.end_layer and self.start_slot+1<=self.step<=self.start_slot+self.duration)
   if not enabled:return self.orig(*a,**kw)
   q,k,v=a[:3]
   assert q.shape[-2]==1 and q.shape[0]==1 and k.shape[-2]==self.prompt_length+self.step
   mask=kw.get('attn_mask',a[3] if len(a)>3 else None)
   dropout=kw.get('dropout_p',a[4] if len(a)>4 else 0.)
   causal=kw.get('is_causal',a[5] if len(a)>5 else False)
   assert dropout==0. and not causal and q.dtype==torch.bfloat16
   scale=kw.get('scale')
   scores=torch.matmul(q,k.transpose(-2,-1))
   scores=scores/math.sqrt(q.shape[-1]) if scale is None else scores*scale
   if mask is not None:
    if mask.dtype==torch.bool:scores=scores.masked_fill(~mask,-float('inf'))
    else:scores=scores+mask
   pos=self.positions.to(q.device);target=scores[...,pos]
   assert torch.isfinite(target).all()
   delta=target.abs()*self.alpha
   altered=scores.clone();altered[...,pos]=target+delta
   assert torch.isfinite(altered[...,pos]).all()
   # Output always uses the original native SDPA backend. Only its additive mask changes.
   if mask is None:base_bias=torch.zeros_like(scores)
   elif mask.dtype==torch.bool:base_bias=torch.zeros_like(scores).masked_fill(~mask,-float('inf'))
   else:base_bias=mask.expand_as(scores).clone()
   assert base_bias.dtype==q.dtype
   extra=torch.zeros_like(scores);extra[...,pos]=delta
   changed=base_bias+extra
   actual_extra=torch.zeros_like(scores)
   actual_extra[...,pos]=changed[...,pos]-base_bias[...,pos]
   other=torch.ones(k.shape[-2],device=q.device,dtype=torch.bool);other[pos]=False
   assert torch.equal(changed[...,other],base_bias[...,other]),'nonimage attention mask changed'
   assert torch.equal(actual_extra[...,pos],delta),'image delta changed during mask encoding'
   # Observations reconstruct qk in FP32; neither diagnostic softmax drives the output.
   fp32_scores=torch.matmul(q.float(),k.float().transpose(-2,-1))
   fp32_scores=fp32_scores/math.sqrt(q.shape[-1]) if scale is None else fp32_scores*scale
   fp32_scores=fp32_scores+base_bias.float()
   p0=F.softmax(fp32_scores,dim=-1);p1=F.softmax(fp32_scores+actual_extra.float(),dim=-1)
   w0=p0.to(q.dtype);w1=p1.to(q.dtype)
   before=p0[...,pos].sum(-1);after=p1[...,pos].sum(-1)
   before_bf16=w0[...,pos].float().sum(-1);after_bf16=w1[...,pos].float().sum(-1)
   self.records.append(dict(forward_step=self.step,feedback_slot=self.step-1,layer=self.active,
    before_fp32=float(before.mean()),after_fp32=float(after.mean()),
    before_bf16_weights=float(before_bf16.mean()),after_bf16_weights=float(after_bf16.mean()),
    per_head_before_fp32=before.flatten().cpu().tolist(),per_head_after_fp32=after.flatten().cpu().tolist(),
    per_head_before_bf16_weights=before_bf16.flatten().cpu().tolist(),per_head_after_bf16_weights=after_bf16.flatten().cpu().tolist(),
    min_delta=float(delta.min()),max_delta=float(delta.max()),alpha=self.alpha,
    image_keys=int(pos.numel()),key_length=k.shape[-2],score_dtype=str(scores.dtype),
    weight_dtype=str(w1.dtype),mask_delta_dtype=str(actual_extra.dtype),mask_delta_min=float(actual_extra[...,pos].min()),mask_delta_max=float(actual_extra[...,pos].max()),nonimage_mask_unchanged=True,mass_is_fp32_qk_reconstruction=True,bf16_weights_are_quantized_diagnostic_only=True,output_backend='unchanged native SDPA',operation='a_image += alpha * abs(a_image)'))
   # alpha zero is a strict native-SDPA control, never the manual reconstruction.
   if self.alpha==0.:return self.orig(*a,**kw)
   aa=list(a);kk=dict(kw)
   if len(aa)>3:aa[3]=changed
   else:kk['attn_mask']=changed
   return self.orig(*aa,**kk)
  F.scaled_dot_product_attention=wrapped
  return self
 def __exit__(self,*exc):
  F.scaled_dot_product_attention=self.orig
  for h in reversed(self.handles):h.remove()
  self.handles=[];self.active=None
 def validate(self):
  keys=[(r['forward_step'],r['layer']) for r in self.records]
  expected={(s,l) for s in range(self.start_slot+1,self.start_slot+self.duration+1) for l in range(self.start_layer,self.end_layer)}
  assert len(keys)==len(set(keys)) and set(keys)==expected
  assert self.images_seen==1 and self.positions.numel()>0
  assert self.alpha==0. or any(r['max_delta']>0 for r in self.records)
  if self.alpha==0.:
   assert all(r['before_fp32']==r['after_fp32'] and r['before_bf16_weights']==r['after_bf16_weights'] for r in self.records)
  return dict(passed=True,alpha=self.alpha,forward_steps=list(range(self.start_slot+1,self.start_slot+self.duration+1)),
   feedback_slots=list(range(self.start_slot,self.start_slot+self.duration)),layers=list(range(2,28)),
   all_heads=True,prefill_unchanged=True,no_cfg=True,no_image_branch=False,
   original_image_only=True,full_method=False,operation='a_image += alpha * abs(a_image)',records=self.records)
