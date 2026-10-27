# Fixed ID1 engineering integration check; no new scientific sample or arm.
from pathlib import Path
import sys,hashlib,json
E=Path(__file__).resolve().parent
p=E/'run_precision_guard.py';src=p.read_text()
needle="  positions=[p for i,p in enumerate(positions) if i%2==a.lane]"
assert src.count(needle)==1
src=src.replace(needle,needle+"\n  positions=[p for p in positions if p['dataset']=='mmvp' and str(p['id'])=='1' and p['position_id']=='first_fact'];assert len(positions)==1")
sys.argv=[str(p),'--phase','full','--lane','0']
exec(compile(src,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
# Read out precisely this accepted existing frozen point, without overwriting full/latest.
p=E/'verify_logits_numeric.py';src=p.read_text();needle="plans=rows(E/'positions.jsonl')";assert src.count(needle)==1
src=src.replace(needle,needle+";plans=[p for p in plans if p['dataset']=='mmvp' and str(p['id'])=='1' and p['position_id']=='first_fact'];assert len(plans)==1")
src=src.replace("out=E/'evaluation'/a.scope","out=E/'evaluation'/'precision_guard_check'")
sys.argv=[str(p),'--scope','full']
exec(compile(src,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
