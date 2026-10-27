"""Temporary direct increase of attention to existing image keys; no pooling."""
import math, hashlib
import torch
import torch.nn.functional as F

class VisualAttentionBoost:
 def __init__(self, model, event, multiplier, observe=True):
  self.model=model;self.event=int(event);self.multiplier=float(multiplier)
  assert self.multiplier in (1.,2.,4.)
  self.delta=math.log(self.multiplier);self.observe=observe
  self.step=-1;self.active_layer=None;self.positions=None;self.records=[];self.handles=[]
 def _model_pre(self,module,args,kwargs):
  self.step+=1
  if self.step==0:
   ids=kwargs.get('input_ids');assert ids is not None and ids.ndim==2 and ids.shape[0]==1
   self.positions=torch.where(ids[0]==self.model.config.image_token_id)[0]
   assert self.positions.numel()>0
   self.prompt_length=int(ids.shape[1])
 def _layer_pre(self,index):
  def hook(module,args,kwargs):
   assert self.active_layer is None;self.active_layer=index
  return hook
 def _layer_post(self,module,args,kwargs,out):self.active_layer=None
 def __enter__(self):
  self.original=F.scaled_dot_product_attention
  self.handles.append(self.model.register_forward_pre_hook(self._model_pre,with_kwargs=True))
  layers=self.model.model.layers
  assert len(layers)==28
  for index,layer in enumerate(layers):
   self.handles.append(layer.self_attn.register_forward_pre_hook(self._layer_pre(index),with_kwargs=True))
   self.handles.append(layer.self_attn.register_forward_hook(self._layer_post,with_kwargs=True))
  def wrapped(*args,**kwargs):
   if not (self.active_layer is not None and self.event+1<=self.step<=self.event+4):
    return self.original(*args,**kwargs)
   assert len(args)>=3
   q,k,v=args[:3];assert q.shape[-2]==1 and q.shape[0]==1
   mask=kwargs.get('attn_mask',args[3] if len(args)>3 else None)
   dropout=kwargs.get('dropout_p',args[4] if len(args)>4 else 0.)
   causal=kwargs.get('is_causal',args[5] if len(args)>5 else False)
   assert dropout==0. and not causal
   assert k.shape[-2]==self.prompt_length+self.step
   pos=self.positions.to(q.device);assert int(pos.max())<k.shape[-2]
   if mask is None:
    bias=torch.zeros((1,1,1,k.shape[-2]),device=q.device,dtype=q.dtype)
   elif mask.dtype==torch.bool:
    bias=torch.zeros_like(mask,dtype=q.dtype).masked_fill(~mask,-float('inf'))
   else:
    assert mask.dtype==q.dtype,'existing floating SDPA mask must match query'
    bias=mask.clone()
   # The original Qwen SDPA mask is broadcastable to B,H,1,K; never alter padded/masked keys.
   valid=torch.isfinite(bias[...,pos]) & (bias[...,pos]>-1e20)
   assert torch.all(bias[...,pos][valid]==0),'valid image mask base must be exactly zero'
   extra=torch.zeros_like(bias)
   extra[...,pos]=torch.where(valid,torch.full_like(bias[...,pos],self.delta),torch.zeros_like(bias[...,pos]))
   changed=bias+extra
   actual_beta=float(torch.tensor(self.delta,dtype=q.dtype))
   actual_multiplier=math.exp(actual_beta)
   assert torch.all((changed[...,pos]-bias[...,pos])[valid]==actual_beta)
   assert torch.equal(changed[...,pos][~valid],bias[...,pos][~valid]),'masked image keys changed'
   if self.observe:
    scale=kwargs.get('scale')
    if scale is None:scale=q.shape[-1]**-0.5
    scores=torch.matmul(q.float(),k.float().transpose(-2,-1))*scale+bias.float()
    p0=torch.softmax(scores,dim=-1)
    p1=torch.softmax(scores+extra.float(),dim=-1)
    before=p0[...,pos].sum(-1);after=p1[...,pos].sum(-1)
    formula=before*actual_multiplier/(1.+before*(actual_multiplier-1.))
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    assert torch.allclose(after,formula,atol=2e-6,rtol=2e-5),'mass odds scaling mismatch'
    assert torch.all(after>=before-2e-6)
    self.records.append(dict(forward_step=self.step,feedback_slot=self.step-1,layer=self.active_layer,
      visual_mass_before=float(before.mean()),visual_mass_after=float(after.mean()),
      per_head_before=before.flatten().cpu().tolist(),per_head_after=after.flatten().cpu().tolist(),
      image_keys=int(pos.numel()),key_length=int(k.shape[-2]),multiplier=self.multiplier,
      delta=self.delta,actual_delta=actual_beta,actual_multiplier=actual_multiplier,mask_applied_dtype=str(changed.dtype),mask_original_dtype=str(mask.dtype) if mask is not None else None,
      image_internal_relative_scores_unchanged=True,all_heads=True))
   if self.delta==0.:return self.original(*args,**kwargs)
   args=list(args);kwargs=dict(kwargs)
   if len(args)>3:args[3]=changed
   else:kwargs['attn_mask']=changed
   return self.original(*args,**kwargs)
  self.wrapper=wrapped;F.scaled_dot_product_attention=wrapped
  return self
 def __exit__(self,exc_type,exc,tb):
  F.scaled_dot_product_attention=self.original
  for h in self.handles:h.remove()
  self.handles=[];self.active_layer=None
 def validate(self,token_count):
  expected=list(range(self.event+1,min(self.event+5,int(token_count))))
  keys=[(r['forward_step'],r['layer']) for r in self.records]
  assert len(keys)==len(set(keys)) and set(keys)=={(s,l) for s in expected for l in range(28)},'wrong layer/forward scope'
  return dict(passed=True,event=self.event,forward_steps=expected,feedback_slots=[s-1 for s in expected],
   all_28_layers=True,prefill_unchanged=True,initial_soft_unchanged=True,original_image=True,
   no_extra_model_forward=True,image_keys=int(self.positions.numel()),multiplier=self.multiplier,
   nominal_delta=self.delta,records=self.records)
