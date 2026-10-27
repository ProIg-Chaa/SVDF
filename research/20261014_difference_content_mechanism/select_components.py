
from pathlib import Path
import sys,json,gzip,hashlib,random,re
E=Path(__file__).resolve().parent;P=E.parent/'20261013_same_position_fact_push';sys.path.insert(0,str(E/'source'))
import torch
from transformers import AutoTokenizer
torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(json.loads((E/'model.json').read_text())['path'],local_files_only=True)
positions=[json.loads(l) for l in (E/'positions.jsonl').read_text().splitlines() if json.loads(l)['cohort']=='first_fact']
stop={'a','an','the','is','are','was','were','be','being','been','it','its','this','that','these','those'}
selected=[];audit=[]
for plan in positions:
 stem=hashlib.sha256((plan['dataset']+':'+str(plan['id'])+':'+plan['position_id']).encode()).hexdigest()[:20]
 src=P/'output/completed'/stem
 with gzip.open(src/'captures_and_updates.pt.gz','rb') as f: cap=torch.load(f,map_location='cpu',weights_only=True)['captures']['4']
 d=cap['original']['p'].float()-cap['weakened']['p'].float()
 top=sorted(range(d.numel()),key=lambda v:(-abs(float(d[v])),v))[:64]
 sets=[set(c['token_ids'][:c['word_token_count']]) for c in plan['candidates']]
 uniq=sets[0]^sets[1]
 target=sorted(v for v in uniq if v in top and tok.decode([v]).strip().lower() not in stop and re.search('[A-Za-z0-9]',tok.decode([v])))
 row=dict(id=str(plan['id']),position_plan=plan,source=str(src),target_ids=target,target_tokens=[dict(id=v,text=tok.decode([v]),d=float(d[v])) for v in target],eligible=bool(target),top64=[dict(id=v,text=tok.decode([v]),d=float(d[v])) for v in top],candidate_ids=sorted(sets[0]|sets[1]),capture_slot=cap['geometry']['capture_slot'],capture_prefix=tok.decode(plan['prefix_token_ids'][:cap['geometry']['capture_slot']]),candidate_phrases=[c['text'] for c in plan['candidates']],source_capture_sha256=hashlib.sha256((src/'captures_and_updates.pt.gz').read_bytes()).hexdigest())
 audit.append(row)
 if target:selected.append(row)
 print('AUDITED',row['id'],target,flush=True)
random.Random(20261014).shuffle(selected);selected=selected[:24]
(E/'stage2/selection_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
(E/'stage2/selected_draft.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False))
print('SELECTED',len(selected),[(x['id'],x['target_tokens']) for x in selected],flush=True)
