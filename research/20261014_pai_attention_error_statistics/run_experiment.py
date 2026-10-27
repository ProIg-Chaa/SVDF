from pathlib import Path
import os,sys,json,hashlib,gzip,time,argparse,gc
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'source'))
import torch
import torch.nn.functional as F
from same_position_probe import SamePositionRuntime
from pai_temporal_attention import PAITemporalAttention
from scale_model_loader import load_model
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def load_tensors(p):
 with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=True)
def save_tensor(p,x):
 with gzip.open(p,'wb',compresslevel=4) as f:torch.save(x,f)
def hook_signature(model):
 return [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in [model]+[l.self_attn for l in model.model.layers]]
def pair(rt,sample,plan,alpha,expected):
 k=len(plan['prefix_token_ids']);out=[];raw=[];att=[]
 signature=hook_signature(rt.model);original=F.scaled_dot_product_attention
 for candidate in plan['candidates']:
  with PAITemporalAttention(rt.model,k-4,alpha=alpha) as guard:
   q,l,_=rt.read(sample,plan['prefix_token_ids'],candidate,4,mode='hard',expected_hashes=expected)
  assert hook_signature(rt.model)==signature and F.scaled_dot_product_attention is original
  out.append(q);raw.append(l);att.append(guard.validate())
 assert torch.equal(raw[0][0],raw[1][0]),'candidate first logits not identical'
 assert att[0]==att[1],'four-step attention trace must be candidate-independent'
 by={q['role']:q for q in out}
 result=dict(readouts=out,correct_logp=by['correct']['phrase_log_probability'],wrong_logp=by['wrong']['phrase_log_probability'],
  B=by['wrong']['phrase_log_probability']-by['correct']['phrase_log_probability'],
  attention=att[0],only_one_role_attention_used=True,no_cfg=True,no_extra_model_forward=True)
 return result,raw
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--lane',type=int,default=0);args=ap.parse_args()
 assert args.lane in (0,1)
 mf=json.loads((E/'MANIFEST.json').read_text());mh=sha(E/'MANIFEST.json')
 for f,h in mf['files'].items():assert sha(E/f)==h,f
 for f,h in mf['parent_files'].items():assert sha(f)==h,f
 items=rows(E/'positions.jsonl');assert len(items)==59 and len({x['id'] for x in items})==59
 if args.phase=='smoke':items=[x for x in items if x['id'] in ('13','72')]
 else:
  assert json.loads((E/'output/SMOKE_PASS.json').read_text())['passed']
  items=[x for j,x in enumerate(items) if j%2==args.lane]
 run=E/'output'/('run_'+args.phase+'_'+os.environ.get('SLURM_JOB_ID','manual')+'_'+str(args.lane)+'_'+str(time.time_ns()));run.mkdir()
 torch.set_num_threads(4)
 model,processor,tok,mi=load_model(json.loads((E/'model.json').read_text()),run)
 rt=SamePositionRuntime(model,processor,tok,mi)
 public={x['id']:x for x in rows(E/'public.jsonl')};registry={x['id']:x for x in rows(E/'registry.jsonl')}
 destination=E/'output'/('smoke' if args.phase=='smoke' else 'completed');destination.mkdir(exist_ok=True)
 accepted=[]
 for plan in items:
  sid=plan['id'];done=destination/sid
  if done.exists():
   a=json.loads((done/'ACCEPTANCE.json').read_text());assert a['passed'] and a['manifest_sha256']==mh
   accepted.append(sid);print('REUSE_ACCEPTED',sid,flush=True);continue
  sample=public[sid];assert sha(sample['image'])==sample['image_sha256']
  stem=hashlib.sha256((plan['dataset']+':'+sid+':first_fact').encode()).hexdigest()[:20]
  parent=Path(mf['parent_experiment'])/'output/completed'/stem
  ref=json.loads((parent/'probes.json').read_text());old=load_tensors(parent/'raw_logits.pt.gz')['4']['hard']
  old_hard=next(x for x in ref['measurements'] if x['duration']==4 and x['arm']=='hard')['readouts']
  bh=json.loads((Path(registry[sid]['folder'])/'plain_soft_once_inputs.json').read_text())['hashes']
  out=run/sid;out.mkdir();results={};raw={}
  for arm,alpha in [('hard',0.),('pai_attention',.2)]:
   q,l=pair(rt,sample,plan,alpha,bh);results[arm]=q;raw[arm]=l
   for j,cr in enumerate(plan['candidates']):
    historical=next(x['raw_logits'] for x in old if x['role']==cr['role'])
    if arm=='hard':
     assert torch.equal(l[j],historical),('historical native logits mismatch',sid,cr['role'])
     prev=next(x for x in old_hard if x['role']==cr['role']);assert q['readouts'][j]['actual_input_hashes']==prev['actual_input_hashes']
    else:
     assert q['readouts'][j]['actual_input_hashes']==results['hard']['readouts'][j]['actual_input_hashes'],'attention must not change forced input embeddings'
   print('BRANCH_DONE',sid,arm,flush=True)
  weak={x['role']:x for x in ref['weak_hard_readouts']}
  weak_correct=weak['correct']['phrase_log_probability'];weak_wrong=weak['wrong']['phrase_log_probability']
  sub=results['hard']['correct_logp']>weak_correct and results['hard']['B']>0
  smoke={}
  if args.phase=='smoke':
   # Plain native path after all hooks are removed must still match the control.
   for j,cr in enumerate(plan['candidates']):
    q0,l0,_=rt.read(sample,plan['prefix_token_ids'],cr,4,mode='hard',expected_hashes=bh)
    assert torch.equal(l0,raw['hard'][j])
   again,againraw=pair(rt,sample,plan,.2,bh)
   assert all(torch.equal(x,y) for x,y in zip(againraw,raw['pai_attention']))
   assert again['attention']==results['pai_attention']['attention']
   smoke=dict(alpha_zero_exact=True,plain_after_hook_cleanup_exact=True,repeated_pai_exact=True,hook_state_cleanup_exact=True)
  write(out/'readouts.json',dict(id=sid,position=plan,results=results,old_weak_hard=dict(correct_logp=weak_correct,wrong_logp=weak_wrong,B=weak_wrong-weak_correct),image_support_wrong_dominant=sub,source_parent=str(parent)))
  save_tensor(out/'raw_logits.pt.gz',raw)
  write(out/'ACCEPTANCE.json',dict(passed=True,id=sid,manifest_sha256=mh,historical_hard_exact=True,first_logits_candidate_independent=True,forced_input_hashes_all_exact=True,attention_four_steps_only=True,smoke=smoke))
  (destination/sid).symlink_to(out,target_is_directory=True);accepted.append(sid)
  gc.collect();torch.cuda.empty_cache()
 if args.phase=='smoke':
  assert set(accepted)=={'13','72'}
  write(E/'output/SMOKE_PASS.json',dict(passed=True,ids=accepted,manifest_sha256=mh))
 write(run/'COMPLETE.json',dict(passed=True,ids=accepted,manifest_sha256=mh));print('LANE_DONE',accepted,flush=True)
if __name__=='__main__':main()
