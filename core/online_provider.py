"""Online weak-image replay provider; unchanged class body."""
import sys,time
import frozen_engine as eng

class OnlineProvider:
 def __init__(self,s,mask_path,zero=False):
  self.sample=s;self.mask_path=mask_path;self.zero=zero;self.event=None;self.vector=None;self.actual_vector=None;self.in_weak=False;self.audit={};self.raw_true=None;self.raw_weak=None;self.hard=None;self.signatures=[]
 def __call__(self,step,selected_token_id,raw_soft,hard,transition_step,entropy_proposed,locked_normal,controller_mode):
  if self.event is not None:return dict(event=self.event,vector=self.vector) if self.event<=step<self.event+4 else None
  allowed=transition_step>=0 and 0<step-transition_step<=8 and entropy_proposed and not locked_normal
  if not allowed:return None
  assert 2<=step<=9 and controller_mode==1
  self.event=step;frame=sys._getframe(1);live=frame.f_locals;prefix=live['all_generated'][0][live['prompt_lens'][0]:]+[selected_token_id];assert len(prefix)==step+1
  self.raw_true=raw_soft.detach().float().cpu();self.hard=hard.detach().float().cpu();main_state=eng.isolation.snapshot()
  from state_isolation import cache_signature,assert_disjoint
  before=cache_signature(frame);self.signatures=[before];weak=[]
  def observe(wframe):
   if int(wframe.f_locals['step']) in (0,step):
    sig=cache_signature(wframe);assert_disjoint([before,sig]);weak.append(sig)
  assert eng.weak_observer is None;eng.weak_observer=observe;self.in_weak=True;t0=time.perf_counter()
  try:
   mask=eng.base.sweep.create_mean_mask(self.sample['image'],self.mask_path)
   wi,wa=eng.capture(self.sample,mask,prefix,step,lam=.95)
   assert wa['refinement_steps']==wa['consumed_refinement_steps']==[] and wi['selected_token_id']==selected_token_id
   self.raw_weak=wi['soft_embedding'];assert eng.torch.equal(wi['hard_embedding'],self.hard)
  finally:
   eng.isolation.restore_checked(main_state);eng.weak_observer=None;self.in_weak=False
  after=cache_signature(frame);assert before==after,'main KV changed during independent weak replay';assert len(weak)==2;self.signatures.extend(weak);self.signatures.append(after)
  ui=self.raw_true-self.hard;uw=self.raw_weak-self.hard;rho=float(ui.norm());difference=ui-uw;dn=float(difference.norm());self.actual_vector=difference*(rho/dn) if dn>1e-12 else eng.torch.zeros_like(difference);self.vector=eng.torch.zeros_like(self.actual_vector) if self.zero else self.actual_vector
  assert eng.torch.isfinite(self.vector).all()
  from state_isolation import tensor_sha
  self.audit=dict(event=step,rho=rho,difference_norm=dn,weak_capture=wa,weak_replay_and_state_restore_seconds=time.perf_counter()-t0,main_cache_unchanged=True,independent_weak_cache=True,state_RNG_restored=True,raw_true_sha256=tensor_sha(self.raw_true),raw_weak_sha256=tensor_sha(self.raw_weak),actual_action_sha256=tensor_sha(self.actual_vector),cast_action_sha256=tensor_sha(self.actual_vector.to(eng.model.dtype)),zero=self.zero)
  return dict(event=step,vector=self.vector)
