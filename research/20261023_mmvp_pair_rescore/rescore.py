import os,json,hashlib,collections
b='/home/bingxing2/home/scx9ftv/gushuo/experiments/20261023_mmvp_pair_rescore'
results=[]; pairs=[]; issues=[]
def iter_rows(p):
 for ln in open(p):
  r=json.loads(ln)
  if isinstance(r.get('correct'),dict):
   for arm,c in r['correct'].items():
    z=dict(r);z['arm']=arm;z['correct']=c
    if isinstance(z.get('prediction'),dict):z['prediction']=z['prediction'].get(arm)
    yield z
  else:yield r
for line in open(b+'/sources.jsonl'):
 p=json.loads(line)
 if any(w in p.lower() for w in ['fixture','synthetic','/tests/']):
  issues.append({'source':p,'issue':'excluded_test_fixture'});continue
 if not os.path.isfile(p):
  issues.append({'source':p,'issue':'missing'});continue
 groups=collections.defaultdict(dict); conflicts=set(); duplicates=0
 for r in iter_rows(p):
  if str(r.get('dataset','')).lower()!='mmvp':continue
  model=r.get('model')
  if not model:
   model=next((s for s in ['openvl','qwen3b','rethinker','vision'] if s in p),'r1')
  arm=r.get('arm',r.get('method',r.get('branch','unknown')))
  for field in ['position_pct','offset','candidate_version']:
   if r.get(field) is not None:arm+='|'+field+'='+str(r[field])
  key=(model,arm)
  try:i=int(r['id'])
  except:issues.append({'source':p,'issue':'noninteger_id','id':r.get('id')});continue
  if type(r.get('correct'))!=bool:
   issues.append({'source':p,'model':model,'arm':arm,'id':i,'issue':'unscored_or_unavailable','value':r.get('correct')});continue
  if i in groups[key]:
   duplicates+=1
   if groups[key][i]['correct']!=r['correct']:conflicts.add(key)
  groups[key][i]=r
 for (model,arm),rows in groups.items():
  if (model,arm) in conflicts:
   issues.append({'source':p,'model':model,'arm':arm,'issue':'conflicting_duplicates'});continue
  if any(i<0 or i>=300 for i in rows):
   issues.append({'source':p,'model':model,'arm':arm,'issue':'ID_OUT_OF_RANGE'});continue
  counts=[0,0,0]; missing=[]
  for pid in range(150):
   ids=[2*pid,2*pid+1]
   if all(i in rows for i in ids):
    c=sum(rows[i]['correct'] for i in ids);counts[c]+=1
    pairs.append({'source':p,'model':model,'arm':arm,'pair_id':pid+1,'ids':ids,'correct':[rows[i]['correct'] for i in ids],'both_correct':c==2,'predictions':[rows[i].get('prediction') for i in ids]})
   elif any(i in rows for i in ids):missing.extend(i for i in ids if i not in rows)
  n=sum(counts); single=sum(r['correct'] for r in rows.values())
  assert single==2*counts[2]+counts[1]+sum(rows[i]['correct'] for i in rows if (i^1) not in rows)
  results.append({'source':p,'source_sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'model':model,'arm':arm,'questions':len(rows),'single_correct':single,'complete_pairs':n,'both_correct':counts[2],'one_correct':counts[1],'both_wrong':counts[0],'pair_accuracy':100.0*counts[2]/n if n else None,'missing_partner_ids':missing,'full300':len(rows)==300,'duplicate_rows_in_source':duplicates})
json.dump(results,open(b+'/SUMMARY.json','w'),indent=2)
json.dump(issues,open(b+'/ISSUES.json','w'),indent=2)
with open(b+'/PAIRS.jsonl','w') as f:
 for r in pairs:f.write(json.dumps(r)+'\n')
print('GROUPS',len(results),'FULL',sum(r['full300'] for r in results),'ISSUES',len(issues))
for r in results:
 if '20261019' in r['source'] or 'analysis_20261020' in r['source']:
  print(r['model'],r['arm'],r['single_correct'],r['questions'],r['both_correct'],r['complete_pairs'],round(r['pair_accuracy'] or 0,2))
