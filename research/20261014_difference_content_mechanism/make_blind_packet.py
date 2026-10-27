import argparse, json, hashlib, random
from pathlib import Path
E=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument("stage",type=int);a=p.parse_args()
F=E/("stage%d"%a.stage)
items=json.loads((F/"selected.json").read_text())
exclusions=json.loads((F/"ENGINEERING_EXCLUSIONS.json").read_text()) if (F/"ENGINEERING_EXCLUSIONS.json").exists() else {}
items=[x for x in items if x["id"] not in exclusions];pack=[];mapping={}
public={str(x["id"]):x for x in [json.loads(l) for l in (E/"public.jsonl").read_text().splitlines()]}
for item in items:
 sid=item["id"];P=F/"completed"/sid
 assert json.loads((P/"ACCEPTANCE.json").read_text())["passed"]
 nat=json.loads((P/"natural.json").read_text())
 assert len(nat)==4
 for row in nat:
  key=hashlib.sha256((sid+"\n"+row["text"]).encode()).hexdigest()[:16]
  if key not in mapping:
   candidates=item["position_plan"]["candidates"]
   pack.append(dict(blind_id=key,id=sid,question=public[sid]["question"],options=public[sid]["options"],target_candidates=candidates,text=row["text"],suffix=row["suffix"],truncated=row["truncated"],fact_source="frozen position plan"))
   mapping[key]=[]
  mapping[key].append(dict(id=sid,arm=row["arm"],stage=a.stage))
random.Random(20261014+a.stage).shuffle(pack)
(F/"blind").mkdir(exist_ok=True)
for name,data in [("packet.json",pack),("mapping_private.json",mapping)]:
 with (F/"blind"/name).open("x") as f:json.dump(data,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(stage=a.stage,branches=sum(len(v) for v in mapping.values()),unique_texts=len(pack)),ensure_ascii=False))
