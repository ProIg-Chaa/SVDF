"""Independent CPU aggregation of frozen natural-trajectory token-rank observations.
No model loading, generation, intervention changes, or outcome-based sample selection.
"""
import argparse,collections,csv,datetime,gzip,hashlib,json,math,statistics,sys
from pathlib import Path
E=Path(__file__).resolve().parent
ARMS=('cot','once','svdf','pai'); ROLES=('correct','wrong'); STAGES=('early','middle','late')
LABELS={'cot':'CoT','once':'OneSoft','svdf':'SVDF-F4','pai':'PAI (image branch)'}
METRICS=('rank','log_probability','probability')
INPUT_SHA='f1b9813c20c1689133105d31628a3c93f9a5913b4f2f8bf4991504132441e844'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p):
 if str(p).endswith('.gz'):
  with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
 return json.loads(Path(p).read_text())
def avg(xs):
 xs=list(xs);assert xs
 return math.fsum(xs)/len(xs)
def close(a,b,atol=2e-9,rtol=1e-9):
 assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b)),(a,b)
def expected_points(n):
 assert n>0
 out=[];bounds=(0,n//3,2*n//3,n)
 for si,stage in enumerate(STAGES):
  lo,hi=bounds[si:si+2];count=min(8,hi-lo)
  if count==0:continue
  ss=[lo] if count==1 else [lo+((hi-1-lo)*i)//(count-1) for i in range(count)]
  assert len(set(ss))==count
  out.extend((s,stage) for s in ss)
 return out
def load_inputs():
 p=E/'input/facts_and_trajectories.jsonl';assert sha(p)==INPUT_SHA,'frozen input changed'
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 assert len(rows)==208 and len({r['case_id'] for r in rows})==208
 for r in rows:
  f=r['fact'];assert f['baseline_status'] in ('correct','incorrect','mixed')
  assert isinstance(f['rank_primary_eligible'],bool)
  assert bool(f['true_token_ids'][0]==f['false_token_ids'][0])==f['same_first_token']
  assert set(r['trajectories'])==set(ARMS)
 assert sum(r['fact']['rank_primary_eligible'] for r in rows)==201
 return rows

def validate_arm(o,source,arm):
 assert o['case_id']==source['case_id'] and o['arm']==arm
 assert o['replay_verified'] is True
 assert o['manifest_sha256']==sha(E/'manifest.json')
 assert o['token_ids_sha256']==hashlib.sha256(json.dumps(source['trajectories'][arm]['token_ids']).encode()).hexdigest()
 eng=o['engineering']
 for k in ('full_historical_tokens_exact','no_forced_tokens','final_layer_exact','final_norm_once','hook_cleanup_exact'):assert eng[k] is True,k
 assert eng['source_sha256']==source['trajectories'][arm]['source_sha256']
 if arm in ('once','svdf'):assert eng['actual_inputs_exact'] is True
 n=len(source['trajectories'][arm]['token_ids']);assert o['n_tokens']==n
 points=[(int(p['step']),p['stage']) for p in o['points']]
 assert sorted(points)==sorted(expected_points(n)),'sampling plan mismatch'
 pd=dict(points);seen=set();fact=source['fact']
 ids={'correct':fact['true_token_ids'][0],'wrong':fact['false_token_ids'][0]}
 for r in o['records']:
  step,li=int(r['step']),int(r['layer_index']);key=(step,li)
  assert key not in seen and step in pd and 0<=li<28
  seen.add(key);assert r['stage']==pd[step] and int(r['layer_number'])==li+1
  cs={x['role']:x for x in r['candidates']};assert set(cs)==set(ROLES) and len(cs)==len(r['candidates'])
  assert math.isfinite(r['logsumexp'])
  for role,c in cs.items():
   assert c['first_token_id']==ids[role]
   assert isinstance(c['rank'],int) and c['rank']>=1
   assert isinstance(c['ties'],int) and c['ties']>=1
   close(c['log_probability'],c['raw_logit']-r['logsumexp'])
   close(c['probability'],math.exp(c['log_probability']),atol=1e-15,rtol=1e-8)
  if fact['same_first_token']:
   for k in ('first_token_id','rank','ties','log_probability','probability','raw_logit'):assert cs['correct'][k]==cs['wrong'][k]
 assert seen=={(step,li) for step in pd for li in range(28)}
 prs=o.get('processed_readouts',[])
 if arm=='pai':
  assert len(prs)==len(points) and {(int(r['step']),r['stage']) for r in prs}==set(points)
  for r in prs:
   assert r.get('score_definition')
   cs={c['role']:c for c in r['candidates']};assert set(cs)==set(ROLES)
   for role,c in cs.items():
    assert c['first_token_id']==ids[role] and isinstance(c['allowed'],bool)
    assert c['rank']>=1 and c['ties']>=1
    assert 0<=c['probability']<=1 and 0<=c['actual_filtered_probability']<=1
    if c['allowed']:
     assert math.isfinite(c['score']) and math.isfinite(c['normalized_log_probability'])
     close(c['probability'],math.exp(c['normalized_log_probability']),atol=1e-12,rtol=1e-6)
    else:
     assert c['score'] is None and c['normalized_log_probability'] is None and c['probability']==0
 else:assert not prs
 return dict(case_id=source['case_id'],arm=arm,n_tokens=n,n_points=len(points),n_records=len(seen),replay_verified=True)

def case_rows(o,source):
 base=dict(case_id=source['case_id'],arm=o['arm'],baseline_status=source['fact']['baseline_status'],primary_eligible=source['fact']['rank_primary_eligible'],exclusion_reason=source['fact']['rank_exclusion_reason'])
 bucket=collections.defaultdict(list)
 for r in o['records']:
  for c in r['candidates']:bucket[(r['stage'],r['layer_number'],c['role'])].append(c)
 layer=[]
 for (stage,ln,role),cs in bucket.items():
  row=dict(base,stage=stage,layer=ln,role=role,n_points=len(cs))
  row.update({k:avg(c[k] for c in cs) for k in METRICS});row['mean_ties']=avg(c['ties'] for c in cs);layer.append(row)
 views=[]
 for stage in STAGES:
  for view,layers in [('last5',range(24,29)),('final',[28])]:
   for role in ROLES:
    rs=[r for r in layer if r['stage']==stage and r['role']==role and r['layer'] in layers]
    if not rs:continue
    assert len(rs)==len(layers)
    row=dict(base,stage=stage,view=view,role=role,n_layers=len(layers),n_points=rs[0]['n_points'])
    row.update({k:avg(r[k] for r in rs) for k in METRICS});views.append(row)
 processed=[]
 for stage in STAGES:
  for role in ROLES:
   cs=[c for r in o.get('processed_readouts',[]) if r['stage']==stage for c in r['candidates'] if c['role']==role]
   if not cs:continue
   row=dict(base,stage=stage,role=role,n_points=len(cs),score_space='PAI_CFG_before_sampling_filters')
   row.update(rank=avg(c['rank'] for c in cs),probability=avg(c['probability'] for c in cs),actual_filtered_probability=avg(c['actual_filtered_probability'] for c in cs),masked_fraction=avg(not c['allowed'] for c in cs))
   allowed=[c['normalized_log_probability'] for c in cs if c['allowed']]
   row['mean_log_probability_allowed_only']=avg(allowed) if allowed else None
   row['log_probability_denominator']=len(allowed);processed.append(row)
 return layer,views,processed

def summarize(rows,dimensions,metrics):
 out=[]
 for mask_name in ('main','excluded','all'):
  eligible=[r for r in rows if mask_name=='all' or (r['primary_eligible'] if mask_name=='main' else not r['primary_eligible'])]
  for group in ('all','correct','incorrect','mixed'):
   selected=[r for r in eligible if group=='all' or r['baseline_status']==group]
   buckets=collections.defaultdict(list)
   for r in selected:buckets[tuple(r[k] for k in dimensions)].append(r)
   for keys,rs in buckets.items():
    assert len({r['case_id'] for r in rs})==len(rs),'each sample must contribute once per cell'
    row=dict(zip(dimensions,keys));row.update(cohort=mask_name,baseline_group=group,n=len(rs))
    for met in metrics:
     vals=[r[met] for r in rs if r[met] is not None]
     row[met+'_n']=len(vals);row[met+'_mean']=avg(vals) if vals else None;row[met+'_median']=statistics.median(vals) if vals else None
    out.append(row)
 return out

def paired_rows(rows,dimensions,metrics):
 index={(r['case_id'],r['arm'])+tuple(r[d] for d in dimensions):r for r in rows}
 assert len(index)==len(rows)
 out=[]
 for r in rows:
  b=index.get((r['case_id'],'once')+tuple(r[d] for d in dimensions));assert b is not None
  row={k:v for k,v in r.items() if k not in metrics}
  row['reference_arm']='once'
  for met in metrics:row['delta_'+met]=r[met]-b[met]
  out.append(row)
 return out

def write_csv(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def draw(out,view_summary,layer_diff,view_diff):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
 colors={'cot':'#222222','once':'#888888','svdf':'#1976A3','pai':'#C87526'}
 def cell(rows,**kw):
  r=[r for r in rows if all(r.get(k)==v for k,v in kw.items())];assert len(r)==1,(kw,len(r));return r[0]
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
 for vi,view in enumerate(('last5','final')):
  for ri,role in enumerate(ROLES):
   ax=axes[vi,ri]
   for arm in ARMS:
    rs=[cell(view_summary,cohort='main',baseline_group='all',arm=arm,stage=s,view=view,role=role) for s in STAGES]
    ax.plot(range(3),[r['rank_mean'] for r in rs],marker='o',lw=2,color=colors[arm],label=LABELS[arm])
   ax.set_title(('Image-consistent' if role=='correct' else 'Image-inconsistent')+' target word | '+('last 5 layers' if view=='last5' else 'final layer'))
   ax.set_xticks(range(3),['Early','Middle','Late']);ax.set_ylabel('Mean vocabulary rank');ax.grid(alpha=.18)
 axes[0,0].legend(fontsize=9);n=cell(view_summary,cohort='main',baseline_group='all',arm='once',stage='early',view='last5',role='correct')['n']
 fig.suptitle(f'Natural-trajectory readouts | fixed {n} eligible facts',fontsize=15)
 fig.text(.02,.015,'Smaller rank = higher support. Each method follows its own text; fixed target words; equal sample weights. PAI: image-branch readout, not CFG scores.',fontsize=9)
 fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(out/'01_stage_rank.png',dpi=180);fig.savefig(out/'01_stage_rank.pdf');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,9),sharey=True)
 mats=[]
 for arm in ('svdf','pai'):
  for role in ROLES:
   mats.append(np.array([[cell(layer_diff,cohort='main',baseline_group='all',arm=arm,stage=s,layer=l,role=role)['delta_rank_mean'] for s in STAGES] for l in range(1,29)]))
 vmax=max(float(np.max(np.abs(a))) for a in mats);vmax=max(1,vmax)
 for ax,mat,(arm,role) in zip(axes.flat,mats,[(a,r) for a in ('svdf','pai') for r in ROLES]):
  im=ax.imshow(mat,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
  ax.set_title(LABELS[arm]+' minus OneSoft | '+('consistent' if role=='correct' else 'inconsistent')+' word')
  ax.set_xticks(range(3),['Early','Middle','Late']);ax.set_yticks([0,6,13,20,27],[1,7,14,21,28]);ax.set_ylabel('Decoder layer')
 fig.subplots_adjust(top=.91,bottom=.09,right=.86,hspace=.24,wspace=.17)
 cax=fig.add_axes([.89,.19,.018,.62]);fig.colorbar(im,cax=cax,label='Mean rank difference; blue = higher word support')
 fig.suptitle('Layer-wise differences along different natural trajectories',fontsize=15)
 fig.text(.02,.025,'Blue is favorable for consistent words; red is favorable for inconsistent words. Association across natural trajectories, not a fixed-prefix causal contrast.',fontsize=9)
 fig.savefig(out/'02_layer_rank_difference.png',dpi=180);fig.savefig(out/'02_layer_rank_difference.pdf');plt.close(fig)
 # Equal average of the three stage means for each case; no pooling of tokens across cases.
 bucket=collections.defaultdict(list)
 for r in view_diff:
  if r['primary_eligible'] and r['view']=='last5':bucket[(r['case_id'],r['arm'],r['role'])].append(r)
 collapsed=[]
 for (cid,arm,role),rs in bucket.items():
  assert {r['stage'] for r in rs}==set(STAGES)
  collapsed.append(dict(case_id=cid,arm=arm,role=role,baseline_status=rs[0]['baseline_status'],primary_eligible=True,delta_rank=avg(r['delta_rank'] for r in rs)))
 ss=summarize(collapsed,('arm','role'),('delta_rank',));table=[]
 for group in ('correct','incorrect','mixed'):
  if not any(r['cohort']=='main' and r['baseline_group']==group for r in ss):continue
  for arm in ARMS:
   c=cell(ss,cohort='main',baseline_group=group,arm=arm,role='correct');w=cell(ss,cohort='main',baseline_group=group,arm=arm,role='wrong')
   table.append([group,str(c['n']),LABELS[arm],f"{c['delta_rank_mean']:+,.1f}",f"{w['delta_rank_mean']:+,.1f}"])
 fig,ax=plt.subplots(figsize=(12,7));ax.axis('off')
 t=ax.table(cellText=table,colLabels=['Original CoT fact','N','Method','Consistent word: rank delta','Inconsistent word: rank delta'],loc='center',cellLoc='center',colWidths=[.16,.06,.24,.27,.27]);t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.9)
 for (row,col),ce in t.get_celld().items():
  ce.set_edgecolor('#DDDDDD')
  if row==0:ce.set_facecolor('#EEEEEE');ce.set_text_props(weight='bold')
 fig.suptitle('Original fact status | change from OneSoft along natural trajectories',fontsize=15,y=.96)
 fig.text(.02,.065,'Last 5 layers; each stage weighted equally within each sample. Negative consistent-word delta is favorable; positive inconsistent-word delta is favorable.',fontsize=9)
 fig.text(.02,.03,'Groups are frozen from original CoT factual text, not answer correctness. The mixed group is retained with its explicit small denominator.',fontsize=9)
 fig.tight_layout(rect=[0,.09,1,.92]);fig.savefig(out/'03_baseline_fact_groups.png',dpi=180);fig.savefig(out/'03_baseline_fact_groups.pdf');plt.close(fig)
 write_csv(out/'figure3_case_equal_stage_differences.csv',collapsed)

def self_test():
 assert len(expected_points(60))==24 and len(expected_points(10))==10
 assert len({s for s,_ in expected_points(1024)})==24
 assert expected_points(3)==[(0,'early'),(1,'middle'),(2,'late')]
 rows=[dict(case_id='a',arm='once',stage='early',role='correct',view='last5',primary_eligible=True,baseline_status='correct',rank=1),dict(case_id='b',arm='once',stage='early',role='correct',view='last5',primary_eligible=True,baseline_status='correct',rank=9)]
 s=summarize(rows,('arm','stage','view','role'),('rank',));assert s[0]['rank_mean']==5 and s[0]['n']==2
 rows += [dict(rows[0],arm='svdf',rank=2),dict(rows[1],arm='svdf',rank=7)]
 p=paired_rows(rows,('stage','view','role'),('rank',));assert [r['delta_rank'] for r in p]==[0,0,1,-2]
 o=dict(case_id='a',arm='once',records=[],processed_readouts=[])
 source=dict(case_id='a',fact=dict(baseline_status='correct',rank_primary_eligible=True,rank_exclusion_reason=None))
 for stage in STAGES:
  for step in (0,1):
   for li in range(28):o['records'].append(dict(step=step,stage=stage,layer_number=li+1,candidates=[dict(role=role,rank=li+step+1,log_probability=-2,probability=math.exp(-2),ties=1) for role in ROLES]))
 layers,views,pr=case_rows(o,source)
 v=[r for r in views if r['view']=='last5'];assert len(v)==6 and all(r['rank']==26.5 for r in v)
 return dict(status='passed',checks=['short_and_long_stage_plans','equal_sample_macro_mean','paired_difference_sign','last_five_layer_aggregation'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--no-plots',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--outputs',type=Path,default=E/'output/completed');ap.add_argument('--out',type=Path);ap.add_argument('--inventory',action='store_true');args=ap.parse_args()
 if args.self_test:
  print(json.dumps(self_test(),indent=2));return
 inputs=load_inputs();inventory=[];complete_cases=[];missing=[]
 for r in inputs:
  paths={a:args.outputs/r['case_id']/(a+'.json.gz') for a in ARMS}
  ms=[a for a,p in paths.items() if not p.exists()]
  if ms:missing.extend(dict(case_id=r['case_id'],arm=a) for a in ms)
  else:complete_cases.append(r)
 inv=dict(expected_cases=208,expected_arms=832,present_arms=832-len(missing),complete_cases=len(complete_cases),missing=missing)
 if args.inventory:print(json.dumps(inv,indent=2));return
 assert not missing or args.allow_partial,'Incomplete experiment: '+json.dumps({k:v for k,v in inv.items() if k!='missing'})
 assert complete_cases,'No complete four-arm cases yet'
 assert all(len(r['trajectories'][a]['token_ids'])>=3 for r in complete_cases for a in ARMS)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=args.out or E/'analysis'/('PARTIAL_' if missing else 'FULL_')/stamp
 out.mkdir(parents=True,exist_ok=False)
 audits=[];layer=[];views=[];processed=[];files=[]
 for r in complete_cases:
  for arm in ARMS:
   p=args.outputs/r['case_id']/(arm+'.json.gz');o=read_json(p)
   audits.append(validate_arm(o,r,arm));lr,vr,pr=case_rows(o,r);layer.extend(lr);views.extend(vr);processed.extend(pr)
   files.append(dict(path=str(p),sha256=sha(p)))
 vs=summarize(views,('arm','stage','view','role'),METRICS)
 ld=paired_rows(layer,('stage','layer','role'),METRICS);vd=paired_rows(views,('stage','view','role'),METRICS)
 lds=summarize(ld,('arm','stage','layer','role'),tuple('delta_'+k for k in METRICS))
 vds=summarize(vd,('arm','stage','view','role'),tuple('delta_'+k for k in METRICS))
 ps=summarize(processed,('arm','stage','role'),('rank','probability','actual_filtered_probability','masked_fraction')) if processed else []
 for name,rs in [('case_layer_stage.csv',layer),('case_stage_views.csv',views),('stage_summary.csv',vs),('case_paired_stage_differences.csv',vd),('paired_stage_summary.csv',vds),('paired_layer_summary.csv',lds),('pai_processed_case_stage.csv',processed),('pai_processed_summary.csv',ps),('arm_validation.csv',audits)]:write_csv(out/name,rs)
 primary=[r for r in complete_cases if r['fact']['rank_primary_eligible']]
 info=dict(complete=not missing,scope='Each method on its own natural trajectory; no fixed-prefix causal interpretation',inventory=inv,input_sha256=INPUT_SHA,analyzer_sha256=sha(__file__),primary_n=len(primary),excluded_n=len(complete_cases)-len(primary),baseline_fact_groups=dict(collections.Counter(r['fact']['baseline_status'] for r in complete_cases)),primary_fact_groups=dict(collections.Counter(r['fact']['baseline_status'] for r in primary)),input_path=str(E/'input/facts_and_trajectories.jsonl'),source_files=files,unit_check=self_test(),validation_rows=len(audits),stage_summary=vs,paired_summary=vds,pai_processed_summary=ps)
 (out/'SUMMARY.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
 if not args.no_plots:draw(out,vs,lds,vd)
 report=['# Natural-trajectory token ranking','',f"Status: {'COMPLETE' if not missing else 'PARTIAL ENGINEERING SNAPSHOT'}; {len(complete_cases)}/208 cases with all four arms; {len(primary)} main-rank eligible.",'','Frozen target words are shared across methods, but each method follows its own generated text. These are trajectory-associated readouts, not a causal fixed-prefix contrast.','', 'Primary measure: vocabulary rank = 1 + count(logit > target logit); lower rank means greater relative support. All labels refer to the original CoT target fact, not answer correctness.','', 'Ranks use final RMSNorm and the output head on each decoder block. The native final-layer normalization is applied once. PAI hidden-state ranks describe its image branch; its actual CFG score readouts and sampling-filter probabilities are in separate tables.','',f'Whole-cohort factual groups: {info["baseline_fact_groups"]}. Main-rank groups: {info["primary_fact_groups"]}.','', 'Main-rank exclusions are frozen before new results: shared first token or context-bound suffix; all 208 cases remain saved and excluded cases have separate summary rows.','', 'Three stages are defined from each trajectory including its recorded terminal token. Up to eight predetermined positions per stage; first average positions and layers per sample, then give samples equal weight. Figures are descriptive; token rank does not establish that a complete fact was corrected.','', '| Method | Stage | Consistent rank | Inconsistent rank | N |','|---|---|---:|---:|---:|']
 for a in ARMS:
  for st in STAGES:
   rr=[r for r in vs if r['cohort']=='main' and r['baseline_group']=='all' and r['arm']==a and r['stage']==st and r['view']=='last5']
   if len(rr)==2:
    rr={r['role']:r for r in rr};report.append(f"| {LABELS[a]} | {st} | {rr['correct']['rank_mean']:.2f} | {rr['wrong']['rank_mean']:.2f} | {rr['correct']['n']} |")
 report += ['', 'Artifacts: `01_stage_rank`, `02_layer_rank_difference`, `03_baseline_fact_groups`; raw per-case tables and PAI processed-score tables are provided alongside this report.','', 'No strength tuning, sample replacement, or favorable-case filtering is performed by this analysis.']
 (out/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 receipt=dict(out=str(out),complete=not missing,summary_sha256=sha(out/'SUMMARY.json'),analyzer_sha256=sha(__file__),files=[dict(name=p.name,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
 (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 print(json.dumps(dict(out=str(out),complete=not missing,n_cases=len(complete_cases),n_primary=len(primary),receipt_sha256=sha(out/'RECEIPT.json')),indent=2))
if __name__=='__main__':main()
