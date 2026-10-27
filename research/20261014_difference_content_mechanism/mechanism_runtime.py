
from pathlib import Path
import sys,json,hashlib,gzip,math,time
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'source'))
import torch
from same_position_probe import SamePositionRuntime
from route_tape import tensor_hash

def rows(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_tensors(p):
 with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=True)
def save_json(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def save_tensor(p,x):
 with gzip.open(p,'wb',compresslevel=4) as f:torch.save(x,f)
def norm_match(direction,hard,reference):
 # Only scalar BF16 quantization compensation. No scientific strength search.
 v=direction.float().to(hard.device);target=float(((hard+reference.to(hard.dtype))-hard).float().norm())
 if target==0:return torch.zeros_like(v),dict(target_norm=0.,actual_norm=0.,relative_error=0.,scale=0.)
 assert v.norm()>0
 unit=v/v.norm();lo=0.;hi=max(target*4,1e-5);best=None
 for _ in range(32):
  s=(lo+hi)/2;a=(unit*s).to(hard.dtype);n=float(((hard+a)-hard).float().norm());err=abs(n-target)
  if best is None or err<best[0]:best=(err,a.clone(),n,s)
  if n<target:lo=s
  else:hi=s
 error=best[0]/target
 assert error<=.02,('actual BF16 norm matching failed',error,target)
 return best[1].float(),dict(target_norm=target,actual_norm=best[2],relative_error=error,scale=best[3])

class Runtime(SamePositionRuntime):
 def natural(self,sample,prefix,actions,probe_meta,first_logits):
  import base
  self.isolation.restore_checked(self.initial)
  inputs,plen=base.prepare(self.processor,sample,sample['image'],self.model.device)
  k=len(prefix);start=k-4;step=-1;hashes=[];previous_soft=None;previous_selected=None;first=None;consumed=[]
  kw=base.sweep.talr_kwargs(self.math_ids,1024,**base.controller(1.25))
  trace=[];kw['token_trace']=trace
  kw.update(lead_refinement_soft_cap=0,lead_refinement_start_offset=0,lead_refinement_soft_mix_lambda=1.,lead_disable_step0_linebreak_mix=True,lead_disable_to_normal_transition=True,lead_force_initial_transition_step1=True,math_ids_tensor=None,forced_prefix_ids=prefix)
  def before(module,args,kwargs):
   nonlocal step
   step+=1;value=kwargs.get('inputs_embeds',kwargs.get('input_ids'))
   if step==0:assert kwargs.get('past_key_values') is None
   else:
    assert len(trace)==step
    hard=self.weight[trace[-1]['token_id']].detach().reshape(1,1,-1)
    expected=previous_soft if step==1 else hard
    assert torch.equal(value,expected),('feedback is not standard Once/hard',step)
    if start<=step-1<k:
     if actions is not None:
      act=actions[step-1-start].to(hard.device,hard.dtype).reshape(1,1,-1)
      value=hard+act;kwargs=dict(kwargs);kwargs['inputs_embeds']=value
     consumed.append(step-1)
   hashes.append(tensor_hash(value))
   if step<=k:assert hashes[-1]==probe_meta['actual_input_hashes'][step],('natural/probe input mismatch',step)
   return args,kwargs
  def after(module,args,kwargs,out):
   nonlocal previous_soft,previous_selected,first
   live=out.logits[:,-1,:].detach()
   if step==0:previous_soft=torch.matmul(torch.softmax(live,dim=-1),self.weight).reshape(1,1,-1)
   previous_selected=prefix[step] if step<k else int(live.argmax(dim=-1).item())
   if step==k:
    first=live[0].to(torch.bfloat16).cpu();assert torch.equal(first,first_logits[0]),'natural/readout first logits mismatch'
  hh=[self.model.register_forward_pre_hook(before,with_kwargs=True),self.model.register_forward_hook(after,with_kwargs=True)]
  t=time.monotonic()
  try:
   with torch.no_grad():out=base.generate_lead(self.model,self.tokenizer,**inputs,**kw)
  finally:
   for h in hh[::-1]:h.remove()
  ids=out[0,plen:].tolist()
  assert ids[:k]==prefix and ids[k]==trace[k]['token_id'] and consumed==list(range(start,k))
  assert all(t['mode']=='normal' for t in trace[1:]),'unexpected continuing soft feedback'
  return dict(prefix_token_ids=prefix,generated_token_ids=ids,text=self.tokenizer.decode(ids,skip_special_tokens=False),suffix=self.tokenizer.decode(ids[k:],skip_special_tokens=False),first_free_token_id=ids[k],first_logits_sha256=tensor_hash(first),actual_input_hashes=hashes,consumed_feedback_slots=consumed,seconds=time.monotonic()-t,truncated=len(ids)>=1024,forced_prefix_diagnostic=True)

def read_pair(rt,sample,plan,actions,bh):
 result=[];raw=[]
 for c in plan['candidates']:
  q,l,v=rt.read(sample,plan['prefix_token_ids'],c,4,mode='hard' if actions is None else 'svdf',action=actions,expected_hashes=bh)
  result.append(q);raw.append(l)
 assert torch.equal(raw[0][0],raw[1][0])
 by={x['role']:x for x in result}
 return dict(readouts=result,B=by['wrong']['phrase_log_probability']-by['correct']['phrase_log_probability'],correct_logp=by['correct']['phrase_log_probability'],wrong_logp=by['wrong']['phrase_log_probability']),raw
