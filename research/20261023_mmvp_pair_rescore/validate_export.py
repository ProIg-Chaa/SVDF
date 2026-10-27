import json,collections,csv,os
b='/home/bingxing2/home/scx9ftv/gushuo/experiments/20261023_mmvp_pair_rescore'
rows=json.load(open(b+'/SUMMARY.json'))
agg=collections.defaultdict(list)
for s in open(b+'/PAIRS.jsonl'):
 r=json.loads(s);assert r['both_correct']==all(r['correct']);assert r['ids']==[2*(r['pair_id']-1),2*(r['pair_id']-1)+1]
 agg[(r['source'],r['model'],r['arm'])].append(r)
for r in rows:
 a=agg[(r['source'],r['model'],r['arm'])]
 assert len(a)==r['complete_pairs']
 assert len(set(x['pair_id'] for x in a))==len(a)
 assert sum(all(x['correct']) for x in a)==r['both_correct']
 assert sum(sum(x['correct'])==1 for x in a)==r['one_correct']
 assert sum(not any(x['correct']) for x in a)==r['both_wrong']
 if r['full300']:assert len(a)==150 and r['single_correct']==2*r['both_correct']+r['one_correct']
full=[r for r in rows if r['full300']]
partial=[r for r in rows if not r['full300']]
for name,data in [('FULL300',full),('PARTIAL',partial)]:
 with open(b+'/'+name+'.csv','w',newline='') as f:
  keys=list(rows[0]);w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
audit={'status':'PASS','source_files_discovered':sum(1 for s in open(b+'/sources.jsonl')),'mmvp_source_files':len(set(r['source'] for r in rows)),'groups_including_historical_versions':len(rows),'full300_groups':len(full),'partial_groups':len(partial),'pair_records':sum(len(a) for a in agg.values()),'issue_counts':dict(collections.Counter(r['issue'] for r in json.load(open(b+'/ISSUES.json')))),'scope':'Discovered scored-row archives in September and October experiment/report indexes; versions kept separate, not independent experiments','no_gpu_jobs_submitted':True,'single_item_grading':'Existing frozen verdicts unchanged'}
json.dump(audit,open(b+'/VALIDATION.json','w'),indent=2)
print(json.dumps(audit))
