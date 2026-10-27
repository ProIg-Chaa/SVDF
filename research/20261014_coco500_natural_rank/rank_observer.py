"""Pure observations on each method's native trajectory, never teacher forcing."""
import math,hashlib
class RankObserver:
 def __init__(self,model,candidates,points,*,positive_branch=lambda:True,dual_check=False):
  self.model=model;self.candidates=candidates;self.points={p['step']:p for p in points};self.positive_branch=positive_branch;self.dual_check=dual_check
  self.step=-1;self.active=False;self.records=[];self.handles=[];self.pending=None;self.main_calls=0;self.negative_calls=0;self.projecting=False;self.final_hidden=None;self.final_normalized=None;self.raw_dumps=[]
 def record(self,li,projected):
  torch=self.torch;raw=projected.detach().contiguous().cpu();assert raw.ndim==1 and raw.numel()==self.model.config.vocab_size and torch.isfinite(raw).all()
  rf=raw.double();lse=float(torch.logsumexp(rf,dim=-1));cs=[]
  for candidate in self.candidates:
   tid=int(candidate['token_ids'][0]);v=float(rf[tid]);rank=int((raw>raw[tid]).sum())+1;ties=int((raw==raw[tid]).sum());lp=v-lse
   if self.dual_check:
    assert rank==int((projected>projected[tid]).sum())+1 and ties==int((projected==projected[tid]).sum())
    assert abs(float(torch.log_softmax(projected.double(),dim=-1)[tid])-lp)<2e-5
   cs.append(dict(role=candidate['role'],first_token_id=tid,rank=rank,ties=ties,raw_logit=v,log_probability=lp,probability=math.exp(lp)))
  self.records.append(dict(step=self.step,stage=self.points[self.step]['stage'],relative_position=self.points[self.step]['relative_position'],layer_index=li,layer_number=li+1,logsumexp=lse,raw_logits_dtype=str(raw.dtype),raw_logits_sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest(),candidates=cs))
  if self.dual_check and self.step==min(self.points) and li in (23,27):self.raw_dumps.append(dict(step=self.step,layer_index=li,raw_logits=raw.clone()))
  if li==27:self.pending=raw
 def __enter__(self):
  import torch
  self.torch=torch;self.modules=[self.model]+list(self.model.model.layers)+[self.model.model.norm,self.model.lm_head]
  assert len(self.model.model.layers)==28
  self.before_hooks=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in self.modules]
  def before(module,args,kw):
   self.active=bool(self.positive_branch())
   if self.active:self.step+=1;self.main_calls+=1
   else:self.negative_calls+=1
  def layer_hook(li):
   def hook(module,args,output):
    if not self.active or self.step not in self.points:return
    h=output[0] if isinstance(output,(tuple,list)) else output;assert h.ndim==3
    if li==27:self.final_hidden=h[:,-1:,:].detach();return
    self.projecting=True
    try:projected=self.model.lm_head(self.model.model.norm(h[:,-1:,:]))[0,-1,:].detach()
    finally:self.projecting=False
    self.record(li,projected)
   return hook
  def norm_hook(module,args,output):
   if self.projecting or not self.active or self.step not in self.points:return
   assert self.final_hidden is not None and torch.equal(args[0][:,-1:,:],self.final_hidden),'final norm must consume layer28 output exactly once'
   self.final_normalized=output[:,-1:,:].detach()
  def head_hook(module,args,output):
   if self.projecting or not self.active or self.step not in self.points:return
   assert self.final_normalized is not None and torch.equal(args[0][:,-1:,:],self.final_normalized),'native head must consume native final RMSNorm'
   self.record(27,output[0,-1,:].detach());self.final_hidden=None;self.final_normalized=None
  def after(module,args,kw,out):
   if not self.active or self.step not in self.points:return
   actual=out.logits[0,-1,:].detach().contiguous().cpu()
   assert self.pending is not None and torch.equal(actual,self.pending),('native final norm/head mismatch',self.step)
   self.pending=None
  self.handles.append(self.model.register_forward_pre_hook(before,with_kwargs=True))
  self.handles.extend(l.register_forward_hook(layer_hook(i)) for i,l in enumerate(self.model.model.layers))
  self.handles.append(self.model.model.norm.register_forward_hook(norm_hook));self.handles.append(self.model.lm_head.register_forward_hook(head_hook))
  self.handles.append(self.model.register_forward_hook(after,with_kwargs=True));return self
 def __exit__(self,*unused):
  for h in reversed(self.handles):h.remove()
  self.handles=[]
  assert [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in self.modules]==self.before_hooks,'observation hook cleanup mismatch'
 def validate(self,n_tokens):
  assert self.main_calls==n_tokens and self.step==n_tokens-1 and self.pending is None and self.final_hidden is None and self.final_normalized is None
  pairs=[(r['step'],r['layer_index']) for r in self.records]
  assert len(pairs)==len(set(pairs)) and set(pairs)=={(s,l) for s in self.points for l in range(28)}
  return dict(main_forward_calls=self.main_calls,negative_forward_calls=self.negative_calls,n_layer_points=len(pairs),final_layer_exact=True,final_norm_once=True,hook_cleanup_exact=True,dual_cpu_gpu_check=self.dual_check,prefill_last_query_only=True)
