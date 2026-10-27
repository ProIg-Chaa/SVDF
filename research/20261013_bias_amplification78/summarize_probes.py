"""Summarize fixed-prefix phrase preference; uses independently reconstructed complete logits only."""
from common import *
import statistics,csv,datetime

def stat(records,arm):
 vals=[r['delta_B'][arm] for r in records];return dict(n=len(vals),wrong_preference_increased=sum(v>0 for v in vals),wrong_preference_decreased=sum(v<0 for v in vals),exactly_unchanged=sum(v==0 for v in vals),mean_delta_B=statistics.mean(vals) if vals else None,median_delta_B=statistics.median(vals) if vals else None)

def main():
 pm=json.loads((E/'probe_manifest.json').read_text());ph=sha(E/'probe_manifest.json');out=E/'evaluation';recounts=[]
 for p in out.glob('LOGITS_RECOUNT_*.json'):
  x=json.loads(p.read_text())
  if x['complete'] and x['probe_manifest_sha256']==ph and x['n']==pm['eligible_count']:recounts.append((p,x))
 assert recounts,'independent complete logits reconstruction required first';rp,x=max(recounts,key=lambda z:z[0].name);rs=x['records'];reg={r['id']:r for r in rows(E/'probe_registry.jsonl')}
 arms=['attention2','attention4','raw_sI','real'];sub=[r for r in rs if r['image_helps_correct_but_still_wrong']]
 summary=dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),probe_manifest_sha256=ph,annotation_sha256=pm['annotation_sha256'],verified_recount=str(rp),verified_recount_sha256=sha(rp),n=len(rs),definition='B=logP(wrong phrase)-logP(correct phrase); deltaB=B_arm-B_original; lower supports correct relative to wrong; full vocabulary, unnormalized candidate pair, no final-answer selection.',sign_rule='Strict mathematical >0/<0/==0 on reconstructed FP32 log-softmax sums; no tuned threshold',all={a:stat(rs,a) for a in arms},image_helps_correct_but_still_wrong=dict(n=len(sub),ids=[r['id'] for r in sub],definition='logP_correct(original)>logP_correct(weakened) AND B_original>0, before any update at fixed baseline prefix',arms={a:stat(sub,a) for a in arms}),descriptive_only=len(rs)<20,analysis_script_sha256=sha(Path(__file__)))
 save(out/'PROBE_SUMMARY.json',summary);save(out/'PROBE_CASE_METRICS.json',rs)
 with (out/'PROBE_CASE_METRICS.csv').open('x',newline='') as f:
  cols=['id','event','probe_token_index','arm','B','delta_B','correct_phrase_logp','image_helps_correct_but_still_wrong'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in rs:
   for a,b in r['B'].items():w.writerow(dict(id=r['id'],event=reg[r['id']]['event'],probe_token_index=reg[r['id']]['probe_token_index'],arm=a,B=b,delta_B=r['delta_B'][a],correct_phrase_logp=r['correct_phrase_logp'][a],image_helps_correct_but_still_wrong=r['image_helps_correct_but_still_wrong']))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
