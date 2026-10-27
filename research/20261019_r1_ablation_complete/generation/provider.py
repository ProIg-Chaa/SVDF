"""One independent mean-RGB replay; all policies and directions are frozen."""
import sys,time
import frozen_engine as eng
class OnlineProvider:
 def __init__(self,s,mask_path,kind='difference',start=2,dynamic=False,zero=False):
  self.sample=s;self.mask_path=mask_path;self.kind=kind;self.start=start;self.dynamic=dynamic;self.zero=zero
  self.event=None;self.vector=None;self.actual_vector=None;self.in_weak=False;self.audit={}
  self.raw_true=None;self.raw_weak=None;self.hard=None;self.budget=0.;self.closed=False;self.decisions=[]
 def __call__(self,step,selected_token_id,raw_soft,hard,transition_step,entropy_proposed,locked_normal,controller_mode):
  frame=sys._getframe(1);live=frame.f_locals
  entropy=float(live['cur_entropy'][0].item());reference=float(live['cur_ref_entropy'][0].item())
  if self.dynamic:
   waiting=step>=2 and not self.closed
   proposed=waiting and not locked_normal and entropy>eng.threshold and entropy>reference
   reason=None
   if proposed:self.closed=True;reason='triggered'
   elif waiting:
    self.budget+=entropy
    if self.budget>=eng.threshold:self.closed=True;reason='budget_exhausted'
   self.decisions.append(dict(step=step,H=entropy,H1=reference,eligible=bool(proposed),budget=self.budget,closed=self.closed,reason=reason))
  else:
   proposed=self.start<=step<self.start+8 and not locked_normal and entropy>eng.threshold and entropy>reference
   if self.start==2 and step>=2:assert bool(proposed)==bool(0<step-transition_step<=8 and entropy_proposed and not locked_normal)
  if self.event is not None:return dict(event=self.event,vector=self.vector,duration=4) if self.event<=step<self.event+4 else None
  if not proposed:return None
  assert transition_step==1 and controller_mode==1 and step>=2
  self.event=step
  prefix=live['all_generated'][0][live['prompt_lens'][0]:]+[selected_token_id];assert len(prefix)==step+1
  self.raw_true=raw_soft.detach().float().cpu();self.hard=hard.detach().float().cpu();state=eng.isolation.snapshot()
  from state_isolation import cache_signature,assert_disjoint,tensor_sha
  before=cache_signature(frame);weak=[]
  def observe(wframe):
   if int(wframe.f_locals['step']) in (0,step):
    sig=cache_signature(wframe);assert_disjoint([before,sig]);weak.append(sig)
  assert eng.weak_observer is None;eng.weak_observer=observe;self.in_weak=True;t=time.perf_counter()
  try:
   mask=eng.base.sweep.create_mean_mask(self.sample['image'],self.mask_path)
   wi,wa=eng.capture(self.sample,mask,prefix,step,lam=.95)
   assert wa['refinement_steps']==wa['consumed_refinement_steps']==[] and wi['selected_token_id']==selected_token_id
   self.raw_weak=wi['soft_embedding'];assert eng.torch.equal(wi['hard_embedding'],self.hard)
  finally:
   eng.isolation.restore_checked(state);eng.weak_observer=None;self.in_weak=False
  after=cache_signature(frame);assert before==after and len(weak)==2
  ui=self.raw_true-self.hard;uw=self.raw_weak-self.hard;rho=float(ui.norm());d=ui-uw;dn=float(d.norm())
  complete=d*(rho/dn) if dn>1e-12 else eng.torch.zeros_like(d)
  if self.kind=='difference':a=complete
  elif self.kind=='reverse':a=-complete
  elif self.kind in ('true_only','weak_only','neg_original_soft'):
   raw=ui if self.kind=='true_only' else uw if self.kind=='weak_only' else -ui;norm=float(raw.norm())
   assert dn<=1e-12 or norm>1e-12
   a=raw*(rho/norm) if dn>1e-12 and norm>1e-12 else eng.torch.zeros_like(raw)
  elif self.kind=='random_104729':
   from action_kernel import matched_random
   key='r1:'+self.sample['dataset']+':'+str(self.sample['id'])+':seed=104729'
   a=matched_random(complete,float(complete.norm()),key) if dn>1e-12 else eng.torch.zeros_like(complete)
  else:raise ValueError(self.kind)
  self.actual_vector=a;self.vector=eng.torch.zeros_like(a) if self.zero else a
  assert eng.torch.isfinite(self.vector).all()
  if dn>1e-12:assert abs(float(a.norm())-rho)<=max(1e-7,rho*2e-6)
  self.audit=dict(event=step,kind=self.kind,trigger_start=self.start,dynamic=self.dynamic,rho=rho,difference_norm=dn,
   weak_capture=wa,weak_replay_and_state_restore_seconds=time.perf_counter()-t,main_cache_unchanged=True,
   independent_weak_cache=True,state_RNG_restored=True,raw_true_sha256=tensor_sha(self.raw_true),
   raw_weak_sha256=tensor_sha(self.raw_weak),complete_action_sha256=tensor_sha(complete),
   actual_action_sha256=tensor_sha(a),cast_action_sha256=tensor_sha(a.to(eng.model.dtype)),
   action_norm=float(a.norm()),cast_action_norm=float(a.to(eng.model.dtype).float().norm()),zero=self.zero)
  return dict(event=step,vector=self.vector,duration=4)
