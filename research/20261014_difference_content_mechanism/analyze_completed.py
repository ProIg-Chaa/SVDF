import argparse,json,statistics,math
from pathlib import Path
E=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument("stage",type=int);args=a.parse_args();F=E/("stage%d"%args.stage)
items=json.loads((F/"selected.json").read_text())
exclusions=json.loads((F/"ENGINEERING_EXCLUSIONS.json").read_text()) if (F/"ENGINEERING_EXCLUSIONS.json").exists() else {}
items=[x for x in items if x["id"] not in exclusions]
mapping=json.loads((F/"blind"/"mapping_private.json").read_text())
annotations=json.loads((F/"blind"/"labels.json").read_text())
assert set(mapping)==set(annotations)
labs={}
for bid,links in mapping.items():
 assert annotations[bid]["label"] in ("correct","wrong","not_mentioned","conflict")
 assert annotations[bid]["reason"]
 for x in links:labs[(x["id"],x["arm"])]=dict(blind_id=bid,**annotations[bid])
out=[]
for it in items:
 sid=it["id"];P=F/"completed"/sid;q=json.loads((P/"readouts.json").read_text());geo=json.loads((P/"geometry.json").read_text())
 for arm,v in q.items():
  d=v["B"]-q["hard"]["B"];sgn="improved" if d < -1e-8 else ("worsened" if d > 1e-8 else "unchanged")
  out.append(dict(id=sid,arm=arm,B=v["B"],correct_logp=v["correct_logp"],wrong_logp=v["wrong_logp"],delta_B=d,movement=sgn,label=labs[(sid,arm)]["label"],blind_id=labs[(sid,arm)]["blind_id"],annotation=labs[(sid,arm)],geometry=geo.get(arm,{})))
arms=list(q);summary={}
for arm in arms:
 rs=[r for r in out if r["arm"]==arm]
 summary[arm]=dict(n=len(rs),mean_delta_B=statistics.mean(r["delta_B"] for r in rs),median_delta_B=statistics.median(r["delta_B"] for r in rs),movement={k:sum(r["movement"]==k for r in rs) for k in ["improved","worsened","unchanged"]},facts={k:sum(r["label"]==k for r in rs) for k in ["correct","wrong","not_mentioned","conflict"]},joint={k:{f:sum(r["movement"]==k and r["label"]==f for r in rs) for f in ["correct","wrong","not_mentioned","conflict"]} for k in ["improved","worsened","unchanged"]})
pairs=[];idx={(r["id"],r["arm"]):r for r in out}
left,right=("target_removed","unrelated_removed") if args.stage==2 else ("relevant","irrelevant")
for it in items:
 sid=it["id"];l=idx[sid,left];r=idx[sid,right];f=idx[sid,"full"]
 pairs.append(dict(id=sid,B_left_minus_right=l["B"]-r["B"],left_minus_full=l["B"]-f["B"],right_minus_full=r["B"]-f["B"],left_fact=l["label"],right_fact=r["label"],full_fact=f["label"]))
ds=[x["B_left_minus_right"] for x in pairs]
contrast=dict(left=left,right=right,mean=statistics.mean(ds),median=statistics.median(ds),positive=sum(x>1e-8 for x in ds),negative=sum(x < -1e-8 for x in ds),tie=sum(abs(x)<=1e-8 for x in ds))
for name,data in [("analysis_rows.json",out),("summary.json",dict(stage=args.stage,arms=summary,paired_contrast=contrast)),("paired_contrast.json",pairs)]:
 with (F/name).open("x") as f:json.dump(data,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(stage=args.stage,arms=summary,paired_contrast=contrast),ensure_ascii=False,indent=2))
