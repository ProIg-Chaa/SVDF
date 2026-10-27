
from mechanism_runtime import *
import argparse,os,gc
from scale_model_loader import load_model
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True);ap.add_argument('--stage',type=int,choices=[2,3],required=True);ap.add_argument('--lane',type=int,default=0);args=ap.parse_args()
 mf=json.loads((E/'MANIFEST.json').read_text())
 for f,h in mf['files'].items():assert sha(E/f)==h,f
 folder=E/('stage'+str(args.stage));items=json.loads((folder/'selected.json').read_text())
 if args.phase=='smoke':items=[x for x in items if x['id'] in mf['smoke_ids'][str(args.stage)]]
 else:
  assert json.loads((folder/'SMOKE_PASS.json').read_text())['passed']
  items=[x for j,x in enumerate(items) if j%4==args.lane]
 run=E/'output'/('stage'+str(args.stage)+'_'+args.phase+'_'+os.environ.get('SLURM_JOB_ID','manual')+'_'+str(args.lane));run.mkdir()
 (run/'masks').mkdir()
 torch.set_num_threads(4);model,processor,tok,mi=load_model(json.loads((E/'model.json').read_text()),run);rt=Runtime(model,processor,tok,mi)
 public={str(x['id']):x for x in rows(E/'public.jsonl')};registry={str(x['id']):x for x in rows(E/'registry.jsonl')}
 completed=[]
 for item in items:
  plan=item['position_plan'];sid=item['id']
  exclusion=folder/'ENGINEERING_EXCLUSIONS.json'
  if exclusion.exists() and sid in json.loads(exclusion.read_text()):
   print('ENGINEERING_EXCLUDED',sid,flush=True);continue
  done=folder/('smoke' if args.phase=='smoke' else 'completed')/sid
  if done.exists():
   assert json.loads((done/'ACCEPTANCE.json').read_text())['passed']
   completed.append(sid);print('REUSE_ACCEPTED',sid,flush=True);continue
  sample=public[sid];prefix=plan['prefix_token_ids'];k=len(prefix);src=Path(item['source'])
  out=run/sid;out.mkdir()
  ref=json.loads((src/'probes.json').read_text());capall=load_tensors(src/'captures_and_updates.pt.gz');cap=capall['captures']['4'];oldraw=load_tensors(src/'raw_logits.pt.gz')['4']
  bh=json.loads((Path(registry[sid]['folder'])/'plain_soft_once_inputs.json').read_text())['hashes']
  original=cap['original'];weak=cap['weakened'];action=cap['action'].float().to(model.device);hard=[rt.weight[prefix[k-4+j]].detach() for j in range(4)]
  full=action.repeat(4,1);configs={'hard':None,'full':full};meta={}
  if args.stage==2:
   d=(original['p']-weak['p']).float().to(model.device);h=original['h'].float().to(model.device)
   def component(ids):
    v=torch.tensor(ids,device=model.device,dtype=torch.long)
    return (d[v,None]*(rt.weight[v].float()-h)).sum(0)
   ct=component(item['target_ids']);cu=component(item['unrelated_ids']);assert cu.norm()>0
   cu=cu*(ct.norm()/cu.norm());delta=(original['s']-original['h'])-(weak['s']-weak['h']);delta=delta.to(model.device)
   for name,cut in [('target_removed',ct),('unrelated_removed',cu)]:
    raw=delta-cut;seq=[];checks=[]
    for hslot in hard:
     a,m=norm_match(raw,hslot,action);seq.append(a);checks.append(m)
    configs[name]=torch.stack(seq);meta[name]=dict(norm_checks=checks,removed_norm=float(cut.norm()),raw_remaining_norm=float(raw.norm()),target_component_norm=float(ct.norm()),raw_original_norm=float(delta.norm()))
   save_tensor(out/'edited_vectors.pt.gz',dict(full=full.cpu(),target_component=ct.cpu(),unrelated_component=cu.cpu(),target_removed=configs['target_removed'].cpu(),unrelated_removed=configs['unrelated_removed'].cpu()))
  else:
   from PIL import Image
   import numpy as np
   im=Image.open(sample['image']).convert('RGB');arr=np.array(im);fill=tuple(int(x) for x in np.mean(arr,axis=(0,1)).astype(np.uint8))
   for name in ['relevant','irrelevant']:
    box=item[name+'_box'];assert 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height
    tmp=im.copy();tmp.paste(fill,box);path=run/'masks'/(sid+'_'+name+'.png');tmp.save(path)
    wi=rt.capture(dict(sample,image=str(path)),prefix,k-4)
    raw=(original['s']-original['h'])-(wi['s']-wi['h']);seq=[];checks=[]
    for hslot in hard:
     a,m=norm_match(raw,hslot,action);seq.append(a);checks.append(m)
    configs[name]=torch.stack(seq)
    q,rl=read_pair(rt,dict(sample,image=str(path)),plan,None,None)
    meta[name]=dict(box=box,fill=fill,mask_sha256=sha(path),norm_checks=checks,reference_fact_readout={z:q[z] for z in ['B','correct_logp','wrong_logp']},capture_norm=float(raw.norm()),capture_probability_target={c['role']:[float(wi['p'][v]) for v in c['token_ids']] for c in plan['candidates']})
    save_tensor(out/(name+'_capture.pt.gz'),wi)
  summaries={};rawsave={};naturals=[]
  for arm,acts in configs.items():
   q,raw=read_pair(rt,sample,plan,acts,bh)
   if arm in ('hard','full'):
    refarm='hard' if arm=='hard' else 'svdf'
    for j,c in enumerate(plan['candidates']):
     old=next(x['raw_logits'] for x in oldraw[refarm] if x['role']==c['role']);assert torch.equal(raw[j],old),('historical logits mismatch',sid,arm,c['role'])
    qm=next(x for x in ref['measurements'] if x['duration']==4 and x['arm']==refarm)
    for a,b in zip(q['readouts'],qm['readouts']):assert a['actual_input_hashes']==b['actual_input_hashes']
   natural=rt.natural(sample,prefix,acts,q['readouts'][0],raw[0]);natural.update(id=sid,arm=arm,stage=args.stage);naturals.append(natural)
   summaries[arm]=q;rawsave[arm]=raw
   print('BRANCH_DONE',args.stage,sid,arm,flush=True)
  if args.phase=='smoke':
   zq,zr=read_pair(rt,sample,plan,torch.zeros_like(full),bh)
   assert all(torch.equal(x,y) for x,y in zip(zr,rawsave['hard']))
   repeated,rr=read_pair(rt,sample,plan,full,bh)
   assert all(torch.equal(x,y) for x,y in zip(rr,rawsave['full']))
   again=rt.natural(sample,prefix,full,repeated['readouts'][0],rr[0])
   nn=next(x for x in naturals if x['arm']=='full');assert again['generated_token_ids']==nn['generated_token_ids'] and again['actual_input_hashes']==nn['actual_input_hashes']
  save_json(out/'readouts.json',summaries);save_json(out/'natural.json',naturals);save_json(out/'geometry.json',meta);save_tensor(out/'raw_logits.pt.gz',rawsave)
  save_json(out/'ACCEPTANCE.json',dict(passed=True,id=sid,stage=args.stage,zero_and_reorder_test=args.phase=='smoke',historical_hard_full_exact=True,natural_first_logits_exact=True,manifest_sha256=sha(E/'MANIFEST.json')))
  dest=folder/('smoke' if args.phase=='smoke' else 'completed');dest.mkdir(exist_ok=True);(dest/sid).symlink_to(out,target_is_directory=True)
  completed.append(sid);gc.collect();torch.cuda.empty_cache()
 if args.phase=='smoke':save_json(folder/'SMOKE_PASS.json',dict(passed=True,ids=completed,manifest_sha256=sha(E/'MANIFEST.json')))
 save_json(run/'COMPLETE.json',dict(passed=True,ids=completed));print('LANE_DONE',completed,flush=True)
if __name__=='__main__':main()
