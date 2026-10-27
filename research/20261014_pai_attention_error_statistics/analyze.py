"""CPU-only summary: each frozen question has exactly one weight."""
from pathlib import Path
import json,math,csv,statistics,hashlib,datetime,argparse
E=Path(__file__).resolve().parent
EPS=1e-8
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def stats(values):
 v=list(values)
 return dict(n=len(v),mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,
  minimum=min(v) if v else None,maximum=max(v) if v else None,
  positive=sum(x>EPS for x in v),negative=sum(x< -EPS for x in v),unchanged=sum(abs(x)<=EPS for x in v))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 manifest_sha=sha(E/'MANIFEST.json');plans=rows(E/'positions.jsonl');selected=[]
 frozen=json.loads((E/'FROZEN_BASELINE_GROUPS.json').read_text())
 raw_sources=[]
 for plan in plans:
  sid=plan['id'];folder=E/'output/completed'/sid
  if not folder.exists():continue
  acc=json.loads((folder/'ACCEPTANCE.json').read_text())
  assert acc['passed'] and acc['manifest_sha256']==manifest_sha
  path=folder/'readouts.json';j=json.loads(path.read_text());base=j['results']['hard'];pai=j['results']['pai_attention']
  dc=pai['correct_logp']-base['correct_logp'];dw=pai['wrong_logp']-base['wrong_logp'];db=pai['B']-base['B']
  row=dict(id=sid,stage=plan['stage'],target_index=len(plan['prefix_token_ids']),correct_phrase=next(x['text'] for x in plan['candidates'] if x['role']=='correct'),
   wrong_phrase=next(x['text'] for x in plan['candidates'] if x['role']=='wrong'),
   base_correct_logp=base['correct_logp'],base_wrong_logp=base['wrong_logp'],base_B=base['B'],
   pai_correct_logp=pai['correct_logp'],pai_wrong_logp=pai['wrong_logp'],pai_B=pai['B'],
   delta_correct_logp=dc,delta_wrong_logp=dw,delta_B=db,
   delta_correct_probability_pp=100*(math.exp(pai['correct_logp'])-math.exp(base['correct_logp'])),
   delta_wrong_probability_pp=100*(math.exp(pai['wrong_logp'])-math.exp(base['wrong_logp'])),
   wrong_probability_up=dw>EPS,wrong_relative_advantage_up=db>EPS,
   both_wrong_up=(dw>EPS and db>EPS),base_B_positive=base['B']>0,
   image_support_wrong_dominant=j['image_support_wrong_dominant'])
  assert row['base_B_positive']==(sid in frozen['baseline_B_positive'])
  assert row['image_support_wrong_dominant']==(sid in frozen['image_support_wrong_dominant'])
  for arm in ['hard','pai_attention']:
   records=j['results'][arm]['attention']['records'];assert len(records)==4*26
   for name in ['before_fp32','after_fp32','before_bf16_weights','after_bf16_weights']:
    row[arm+'_'+name]=statistics.mean(r[name] for r in records)
  for precision in ['fp32','bf16_weights']:
   row['baseline_to_pai_mass_change_'+precision]=row['pai_attention_after_'+precision]-row['hard_before_'+precision]
   row['pai_same_query_local_mass_change_'+precision]=row['pai_attention_after_'+precision]-row['pai_attention_before_'+precision]
  selected.append(row);raw_sources.append(dict(path=str(path),sha256=sha(path)))
 if not args.partial:assert len(selected)==len(plans),('need all frozen cases',len(selected),len(plans))
 groups={}
 for name,pred in [('all_eligible',lambda x:True),('baseline_B_positive',lambda x:x['base_B_positive']),('image_support_wrong_dominant',lambda x:x['image_support_wrong_dominant'])]:
  subset=[x for x in selected if pred(x)]
  entry=dict(n=len(subset),ids=[x['id'] for x in subset],
   wrong_probability_up=sum(x['wrong_probability_up'] for x in subset),
   wrong_relative_advantage_up=sum(x['wrong_relative_advantage_up'] for x in subset),
   both_wrong_up=sum(x['both_wrong_up'] for x in subset))
  for key in ['delta_B','delta_correct_logp','delta_wrong_logp','delta_correct_probability_pp','delta_wrong_probability_pp',
   'hard_before_fp32','hard_before_bf16_weights','pai_attention_before_fp32','pai_attention_after_fp32',
   'pai_attention_before_bf16_weights','pai_attention_after_bf16_weights',
   'baseline_to_pai_mass_change_fp32','baseline_to_pai_mass_change_bf16_weights','pai_same_query_local_mass_change_fp32','pai_same_query_local_mass_change_bf16_weights']:
   entry[key]=stats(x[key] for x in subset)
  groups[name]=entry
 out=E/'analysis'/datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
 summary=dict(complete=len(selected)==len(plans),n=len(selected),manifest_sha256=manifest_sha,
  fullphrase=True,epsilon=EPS,groups=groups,source_files=raw_sources,
  claim_scope='Same-prefix attention-only local perturbation of frozen visual fact candidates; no decomposition of parameter/visual content, no final accuracy.',
  attention_scope='Per-case equal weight; one role only; baseline native path vs PAI path and PAI same-query local operation separately.')
 with (out/'SUMMARY.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
 if selected:
  with (out/'per_case.csv').open('x',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
 print(json.dumps(dict(path=str(out),complete=summary['complete'],n=len(selected),groups=groups),ensure_ascii=False))
if __name__=='__main__':main()
