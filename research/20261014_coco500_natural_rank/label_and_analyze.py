#!/usr/bin/env python3
"""CPU-only VISTA-inspired COCO80 object-mention labeling and natural-rank analysis.
Not GPT-4o annotation; not replication of VISTA's exact 500 images.
No model generation, GPU access, oracle-dependent intervention, or parameter search.
"""
import argparse,collections,csv,datetime,gzip,hashlib,importlib.util,json,math,os,re,sys
from pathlib import Path
E=Path(__file__).resolve().parent
R=E.parent.parent
sys.path[:0]=[str(E/'analysis_deps'),str(R/'experiments/20260915_internvl14b_core_h2/deps'),str(R/'container_python/py310')]
os.environ['NLTK_DATA']=str(E/'analysis_nltk_data')
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
ARMS=('cot','once','svdf','pai')
CLASSES=('decoded_genuine','hidden_genuine','hallucinated')
CLASS_NAMES={'decoded_genuine':'GT-present and mentioned','hidden_genuine':'GT-present, unmentioned','hallucinated':'Mentioned, GT-absent'}
STAGES=('early','middle','late')
REFERENCE_SHA='fb9aa4c18432a6c8b6dadebcd0727298b2e6c1f371e53307f6bed82c5ddeb49a'
RULES={'scope':'COCO80 object mentions only; no attribute or relation labels','ground_truth':'Union of official instance categories and official captions extracted by the frozen CHAIR rules','classification':'CoT full natural generated text only; control tags replaced by spaces; think body retained; all methods share frozen targets','surface':'Decoded and GT-absent targets use actual CoT surface and actual token IDs covering its first observed occurrence; hidden uses leading-space canonical object name','deduplication':'Within each image and class, unique canonical plus actual first-token ID; repeated mentions do not increase rank weight','exclusions':['token alignment failure','ambiguous lexical/canonical class mapping','first token shared by different classes within one image'],'no_semantic_filter':'Do not exclude negated or uncertain mentions; this is an object-mention CHAIR-based measure, not affirmative factual annotation','aggregation':'Per image/class average target words, selected timesteps and selected layers first; then equal-weight images with eligible class. Missing classes do not become zero.','positions':'Each arm own natural trajectory, equal thirds, up to eight deterministic equal-spaced points per third; 28 layers; final RMSNorm exactly once','interpretation':'Different natural trajectories are compared; rank shifts are associated with method generation, not a common-prefix causal estimate.'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else open(p,encoding='utf-8')) as f:return json.load(f)
def rows(p):
 with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else open(p,encoding='utf-8')) as f:return [json.loads(l) for l in f if l.strip()]
def write_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def write_jsonl(p,items):
 with Path(p).open('x',encoding='utf-8') as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def mean(xs):return sum(xs)/len(xs) if xs else None
def write_csv(p,items):
 if not items:return
 with (gzip.open(p,'xt',encoding='utf-8',newline='') if str(p).endswith('.gz') else open(p,'x',encoding='utf-8',newline='')) as f:
  writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
def clean_controls(text):return re.sub(r'<\|[^<>]*\|>|</?think\s*>',lambda m:' '*len(m.group()),text,flags=re.I)
def point_plan(n):
 out=[];edges=(0,n//3,2*n//3,n)
 for stage,lo,hi in zip(STAGES,edges[:-1],edges[1:]):
  count=min(8,hi-lo)
  out.extend((lo if count==1 else lo+(hi-1-lo)*i//(count-1),stage) for i in range(count))
 return out

def make_extractor():
 assert sha(E/'analysis_reference/chair_ans.py')==REFERENCE_SHA
 spec=importlib.util.spec_from_file_location('frozen_chair',E/'analysis_reference/chair_ans.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 class NoLoadCHAIR(m.CHAIR):
  def get_annotations(self):pass
 obj=NoLoadCHAIR('unused')
 # Empty synonym rows in upstream are not COCO categories.
 obj.mscoco_objects=[w for w in obj.mscoco_objects if w]
 obj.inverse_synonym_dict={k:v for k,v in obj.inverse_synonym_dict.items() if k and v}
 assert len(set(obj.inverse_synonym_dict.values()))==80
 return obj

def extract_mentions(text,chair):
 """Official CHAIR normalization/phrase rules, plus retained raw text spans.
 Each result is checked against the original caption_to_words canonical output.
 """
 import nltk
 from nltk.stem import WordNetLemmatizer
 cleaned=clean_controls(text);words=nltk.word_tokenize(cleaned)
 tagged=nltk.pos_tag([w.lower() for w in words]);wnl=WordNetLemmatizer()
 lemmas=[wnl.lemmatize(w,pos=chair.get_wordnet_pos(tag) or 'n') for w,tag in tagged]
 spans=[];cursor=0
 for word in words:
  alternatives=[word]
  if word in ('``',"''"):alternatives += ['"','“','”']
  found=[(cleaned.find(a,cursor),a) for a in alternatives if cleaned.find(a,cursor)>=0]
  if found:
   start,actual=min(found);end=start+len(actual);cursor=end;spans.append((start,end))
  else:spans.append(None)
 combined=[];i=0
 while i<len(lemmas):
  pair=' '.join(lemmas[i:i+2]);size=2 if pair in chair.double_word_dict else 1
  normalized=chair.double_word_dict[pair] if size==2 else lemmas[i]
  word_indices=list(range(i,i+size))
  # Upstream collapses qualifiers (baby bird -> bird, toilet seat -> toilet).
  if size==2 and normalized!=pair:
   matches=[j for j in word_indices if lemmas[j]==normalized]
   if len(matches)==1:word_indices=matches
  valid=all(spans[j] is not None for j in word_indices)
  start=spans[word_indices[0]][0] if valid else None;end=spans[word_indices[-1]][1] if valid else None
  combined.append({'normalized':normalized,'canonical':chair.inverse_synonym_dict.get(normalized),'surface':text[start:end] if valid else normalized,'start':start,'end':end,'word_indices':word_indices})
  i+=size
 normalized=[x['normalized'] for x in combined]
 if 'toilet' in normalized and 'seat' in normalized:combined=[x for x in combined if x['normalized']!='seat']
 result=[x for x in combined if x['canonical'] is not None]
 official=chair.caption_to_words(cleaned)
 assert [x['canonical'] for x in result]==official[1],('CHAIR canonical mismatch',text)
 assert [x['normalized'] for x in result]==official[0],('CHAIR lexical mismatch',text)
 return result

def build_gt(public,chair):
 cp=E/'input/gt/captions_val2014_selected500.json.gz';ip=E/'input/gt/instances_val2014_selected500.json.gz'
 caps,ins=load(cp),load(ip);wanted={int(s['id']) for s in public};gt={i:set() for i in wanted}
 assert {x['id'] for x in caps['images']}==wanted=={x['id'] for x in ins['images']}
 categories={x['id']:x['name'] for x in ins['categories']};assert len(categories)==80
 for ann in ins['annotations']:
  assert ann['image_id'] in wanted
  gt[ann['image_id']].add(chair.inverse_synonym_dict[categories[ann['category_id']]])
 for ann in caps['annotations']:
  assert ann['image_id'] in wanted
  gt[ann['image_id']].update(chair.caption_to_words(ann['caption'])[1])
 return gt,{str(cp.relative_to(E)):sha(cp),str(ip.relative_to(E)):sha(ip)}

def add_exclusions(targets):
 by_first=collections.defaultdict(set)
 for q in targets:
  if q['token_ids']:by_first[q['token_ids'][0]].add(q['class'])
 for q in targets:
  if not q['token_ids']:q['exclusion_reasons'].append('token_alignment_failure')
  elif len(by_first[q['token_ids'][0]])>1:q['exclusion_reasons'].append('cross_class_first_token_collision')
  q['rank_primary_eligible']=not q['exclusion_reasons']
 return targets

def targets_for_case(record,ground_truth,chair,tokenizer):
 text=record['text_with_special'];mentions=extract_mentions(text,chair)
 encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 aligned=encoded['input_ids']==record['generated_token_ids']
 # On a failed decode/encode roundtrip, never use newly tokenized IDs as actual IDs.
 # Keep the image and labels, exclude only observed targets from rank; hidden canonical targets remain.
 record['_label_roundtrip_exact']=aligned
 seen=set();targets=[];mentioned={m['canonical'] for m in mentions}
 for m in mentions:
  reasons=[];ids=[]
  if not aligned:reasons.append('full_text_token_roundtrip_mismatch')
  if aligned and m['start'] is not None:
   indices=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>m['start'] and a<m['end']]
   if indices:
    ids=[encoded['input_ids'][i] for i in indices]
    if indices!=list(range(min(indices),max(indices)+1)):reasons.append('noncontiguous_token_alignment')
  cls='decoded_genuine' if m['canonical'] in ground_truth else 'hallucinated'
  key=(m['canonical'],ids[0] if ids else m['surface'])
  if key in seen:continue
  seen.add(key)
  targets.append(dict(class_name=CLASS_NAMES[cls],**{'class':cls},surface=m['surface'],text=m['surface'],canonical=m['canonical'],token_ids=ids,origin='actual_cot_mention',first_occurrence_char_span=[m['start'],m['end']],exclusion_reasons=reasons))
 for canonical in sorted(ground_truth-mentioned):
  ids=tokenizer.encode(' '+canonical,add_special_tokens=False)
  targets.append(dict(class_name=CLASS_NAMES['hidden_genuine'],**{'class':'hidden_genuine'},surface=canonical,text=canonical,canonical=canonical,token_ids=ids,origin='leading_space_gt_canonical',first_occurrence_char_span=None,exclusion_reasons=[]))
 for i,q in enumerate(targets):q.update(target_id='t%03d'%i,role='t%03d'%i)
 add_exclusions(targets)
 return targets,mentions

def mention_metrics(record,gt,chair):
 ms=extract_mentions(record['text_with_special'],chair);canon=[m['canonical'] for m in ms];bad=[x for x in canon if x not in gt]
 return dict(case_id=record['case_id'],arm=record['arm'],n_tokens=record['n_tokens'],truncated=record['truncated'],n_mentions=len(canon),n_gt_absent_mentions=len(bad),has_gt_absent_mention=bool(bad),n_unique_mentioned=len(set(canon)),n_unique_gt_mentioned=len(set(canon)&gt),n_gt=len(gt),chair_i=(len(bad)/len(canon) if canon else None),chair_s=int(bool(bad)),gt_recall=(len(set(canon)&gt)/len(gt) if gt else None),mentions=ms)

def label(a):
 public=rows(E/'input/public.jsonl');assert len(public)==500 and len({r['case_id'] for r in public})==500
 gm=E/'GENERATION_MANIFEST.json';gm_sha=sha(gm)
 paths=[E/'output/generation/completed'/s['case_id']/(arm+'.json.gz') for s in public for arm in ARMS]
 missing=[str(p.relative_to(E)) for p in paths if not p.exists()]
 if missing:raise RuntimeError('Label freezing requires all 2000 natural arms; missing %d'%len(missing))
 out=E/'labels'
 if (out/'LABEL_FREEZE.json').exists():
  frozen=load(out/'LABEL_FREEZE.json')
  assert frozen['generation_manifest_sha256']==gm_sha and frozen['label_file_sha256']==sha(out/'candidates.jsonl')
  print(json.dumps({'status':'already_frozen','label_sha256':frozen['label_file_sha256']}));return
 out.mkdir(exist_ok=True)
 chair=make_extractor();gt,gt_hashes=build_gt(public,chair)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(load(E/'model.json')['path'],local_files_only=True,use_fast=True);assert tok.is_fast
 result=[];mention_rows=[];source_hashes={};counts=collections.Counter();eligible_counts=collections.Counter();class_images=collections.Counter();excluded=collections.Counter()
 for sample in public:
  case=sample['case_id'];case_gt=gt[int(sample['id'])];records={}
  for arm in ARMS:
   p=E/'output/generation/completed'/case/(arm+'.json.gz');r=load(p)
   assert r['case_id']==case and r['arm']==arm and r['generation_manifest_sha256']==gm_sha
   assert r['n_tokens']==len(r['generated_token_ids']) and 0<r['n_tokens']<=512 and r['oracle_read'] is False
   source_hashes[str(p.relative_to(E))]=sha(p);records[arm]=r
   mention_rows.append(mention_metrics(r,case_gt,chair))
  targets,mentions=targets_for_case(records['cot'],case_gt,chair,tok)
  primary=[q for q in targets if q['rank_primary_eligible']];ex=[q for q in targets if not q['rank_primary_eligible']]
  for q in targets:counts[q['class']]+=1
  for q in primary:eligible_counts[q['class']]+=1
  for cls in {q['class'] for q in primary}:class_images[cls]+=1
  for q in ex:excluded.update(q['exclusion_reasons'])
  result.append(dict(case_id=case,image_id=int(sample['id']),gt_canonical=sorted(case_gt),candidates=primary,excluded_candidates=ex,status='eligible' if primary else 'no_eligible_candidate',cot_source_sha256=source_hashes[str((E/'output/generation/completed'/case/'cot.json.gz').relative_to(E))],cot_token_roundtrip_exact=records['cot'].get('_label_roundtrip_exact'),cot_generated_token_ids=records['cot']['generated_token_ids'],cot_n_tokens=records['cot']['n_tokens'],cot_truncated=records['cot']['truncated'],cot_text_with_special=records['cot']['text_with_special'],cot_mentions=mentions))
 write_jsonl(out/'candidates.jsonl',result)
 with gzip.open(out/'all_arm_mentions.jsonl.gz','xt',encoding='utf-8') as f:
  for row in mention_rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
 write_json(out/'GENERATION_SOURCES.json',source_hashes)
 dependencies={str(p.relative_to(E)):sha(p) for p in sorted((E/'analysis_nltk_data').rglob('*.zip'))}
 freeze=dict(created_utc=now(),n_images=500,n_arms=2000,generation_manifest_sha256=gm_sha,public_sha256=sha(E/'input/public.jsonl'),label_file_sha256=sha(out/'candidates.jsonl'),all_arm_mentions_sha256=sha(out/'all_arm_mentions.jsonl.gz'),generation_sources_sha256=sha(out/'GENERATION_SOURCES.json'),gt_files=gt_hashes,chair_reference_sha256=REFERENCE_SHA,nltk_version='3.9.1',nltk_resources=dependencies,label_script_sha256=sha(__file__),rules=RULES,all_target_counts=dict(counts),eligible_target_counts=dict(eligible_counts),eligible_image_counts=dict(class_images),excluded_reasons=dict(excluded),no_eligible_candidate_images=sum(not r['candidates'] for r in result),interpretation='VISTA-inspired object-only operational labels; not GPT-4o verified visual facts and not the original VISTA 500 sample.')
 write_json(out/'LABEL_FREEZE.json',freeze)
 print(json.dumps(freeze,ensure_ascii=False,indent=2))

def validate_replay(record,label_row,generation,manifest_sha):
 assert record['case_id']==label_row['case_id'] and record['arm'] in ARMS
 assert record['generated_token_ids']==generation['generated_token_ids']
 assert record['actual_input_hashes']==generation['actual_input_hashes']
 if 'replay_manifest_sha256' in record:assert record['replay_manifest_sha256']==manifest_sha
 if not label_row['candidates']:
  assert record.get('engineering',{}).get('replay_skipped') and not record.get('records')
  return
 assert record['replay_verified'] is True
 expected=point_plan(record['n_tokens']);assert [(x['step'],x['stage']) for x in record['points']]==expected
 byrole={q['role']:q for q in label_row['candidates']};assert len(byrole)==len(label_row['candidates'])
 assert {q['role']:q['token_ids'] for q in record['candidates']}=={k:v['token_ids'] for k,v in byrole.items()}
 seen=set()
 for r in record['records']:
  key=(r['step'],r['layer_index']);assert key not in seen;seen.add(key)
  assert (r['step'],r['stage']) in expected and r['layer_number']==r['layer_index']+1
  assert {q['role'] for q in r['candidates']}==set(byrole)
  for q in r['candidates']:
   assert q['first_token_id']==byrole[q['role']]['token_ids'][0]
   assert 1<=q['rank']<=152064 and 1<=q['ties']<=152064
   assert math.isfinite(q['raw_logit']) and math.isfinite(q['log_probability'])
   assert abs(q['log_probability']-(q['raw_logit']-r['logsumexp']))<1e-8
   assert abs(q['probability']-math.exp(q['log_probability']))<1e-10
 assert seen=={(step,l) for step,stage in expected for l in range(28)}
 if record['arm']=='pai':
  assert {(x['step'],x['stage']) for x in record['processed_readouts']}==set(expected)
  for rr in record['processed_readouts']:
   assert {q['role'] for q in rr['candidates']}==set(byrole)
   for q in rr['candidates']:
    assert q['first_token_id']==byrole[q['role']]['token_ids'][0]
    if not q['allowed']:assert q['score'] is None and q['normalized_log_probability'] is None and q['probability']==0

def mean_metrics(candidates):
 return {k:mean([q[k] for q in candidates]) for k in ('rank','ties','log_probability','probability')}

def reduce_case(record,label_row):
 roles={q['role']:q['class'] for q in label_row['candidates']};nwords=collections.Counter(roles.values())
 output=[];layers=[];processed=[]
 for cls in CLASSES:
  if not nwords[cls]:continue
  for stage in STAGES:
   for view,ls in [('last5',set(range(23,28))),('final',set([27]))]:
    values=[q for r in record['records'] if r['stage']==stage and r['layer_index'] in ls for q in r['candidates'] if roles[q['role']]==cls]
    if values:output.append(dict(case_id=record['case_id'],arm=record['arm'],class_name=cls,stage=stage,view=view,n_words=nwords[cls],**mean_metrics(values)))
  for layer in range(28):
   values=[q for r in record['records'] if r['layer_index']==layer for q in r['candidates'] if roles[q['role']]==cls]
   if values:layers.append(dict(case_id=record['case_id'],arm=record['arm'],class_name=cls,layer=layer+1,n_words=nwords[cls],**mean_metrics(values)))
  if record['arm']=='pai':
   for stage in STAGES:
    values=[q for r in record['processed_readouts'] if r['stage']==stage for q in r['candidates'] if roles[q['role']]==cls]
    if values:
     allowed=[q for q in values if q['allowed']]
     processed.append(dict(case_id=record['case_id'],arm='pai',class_name=cls,stage=stage,n_words=nwords[cls],rank=mean([q['rank'] for q in values]),probability=mean([q['probability'] for q in values]),allowed_fraction=mean([int(q['allowed']) for q in values]),allowed_only_log_probability=mean([q['normalized_log_probability'] for q in allowed]),score_space='PAI final CFG and plausibility mask, separate from main-image hidden state'))
 return output,layers,processed

def aggregate(items,keys,metrics):
 groups=collections.defaultdict(list)
 for item in items:groups[tuple(item[k] for k in keys)].append(item)
 out=[]
 for key,group in sorted(groups.items()):
  assert len({r['case_id'] for r in group})==len(group),'Not one row per image before macro averaging'
  row=dict(zip(keys,key));row['n_images']=len(group);row['sum_targets']=sum(x['n_words'] for x in group)
  for metric in metrics:row[metric]=mean([r[metric] for r in group if r[metric] is not None])
  out.append(row)
 return out

def paired_differences(percase):
 lookup={(r['case_id'],r['arm'],r['class_name'],r['stage'],r['view']):r for r in percase if r['view']=='last5'}
 details=[]
 for r in percase:
  if r['view']!='last5':continue
  for baseline in ('cot','once'):
   if r['arm']==baseline:continue
   b=lookup.get((r['case_id'],baseline,r['class_name'],r['stage'],'last5'))
   if b is None:continue
   details.append(dict(case_id=r['case_id'],arm=r['arm'],baseline=baseline,class_name=r['class_name'],stage=r['stage'],n_words=r['n_words'],rank_difference=r['rank']-b['rank'],log_probability_difference=r['log_probability']-b['log_probability'],probability_difference=r['probability']-b['probability']))
 groups=collections.defaultdict(list)
 for r in details:groups[tuple(r[k] for k in ('arm','baseline','class_name','stage'))].append(r)
 summaries=[]
 for key,group in sorted(groups.items()):
  ds=[r['rank_difference'] for r in group];truth=key[2]!='hallucinated'
  summaries.append(dict(zip(('arm','baseline','class_name','stage'),key),n_images=len(group),mean_rank_difference=mean(ds),median_rank_difference=__import__('statistics').median(ds),rank_lower_images=sum(d<0 for d in ds),rank_higher_images=sum(d>0 for d in ds),unchanged_images=sum(d==0 for d in ds),desired_direction_images=sum(d<0 if truth else d>0 for d in ds),opposite_direction_images=sum(d>0 if truth else d<0 for d in ds),desired_direction='lower rank for GT-present; higher rank for GT-absent',mean_log_probability_difference=mean([r['log_probability_difference'] for r in group]),mean_probability_difference=mean([r['probability_difference'] for r in group])))
 return details,summaries

def chair_summary(mention_rows):
 output=[]
 for arm in ARMS:
  xs=[r for r in mention_rows if r['arm']==arm];assert len(xs)==500
  mentions=sum(r['n_mentions'] for r in xs);bad=sum(r['n_gt_absent_mentions'] for r in xs);gt=sum(r['n_gt'] for r in xs);recognized=sum(r['n_unique_gt_mentioned'] for r in xs)
  output.append(dict(arm=arm,n_images=500,n_truncated=sum(r['truncated'] for r in xs),n_no_object_mentions=sum(r['n_mentions']==0 for r in xs),total_mentions=mentions,gt_absent_mentions=bad,chair_s=sum(r['chair_s'] for r in xs)/500,chair_i=bad/mentions if mentions else None,n_images_with_gt=sum(r['n_gt']>0 for r in xs),gt_recall_micro=recognized/gt if gt else None,gt_recall_macro=mean([r['gt_recall'] for r in xs if r['gt_recall'] is not None]),description='CHAIR object mentions over full generated text including reasoning; not affirmative-claim verification'))
 return output

def analyze(a):
 frozen=load(E/'labels/LABEL_FREEZE.json');assert sha(E/'labels/candidates.jsonl')==frozen['label_file_sha256'];assert sha(E/'labels/all_arm_mentions.jsonl.gz')==frozen['all_arm_mentions_sha256']
 labels=rows(E/'labels/candidates.jsonl');assert len(labels)==500
 missing=[(r['case_id'],arm) for r in labels for arm in ARMS if not (E/'output/replay/completed'/r['case_id']/(arm+'.json.gz')).exists()]
 if missing:raise RuntimeError('Full analysis requires 2000 replay records, missing %d'%len(missing))
 out=Path(a.out) if a.out else E/'analysis'
 if out.exists() and list(out.iterdir()):raise RuntimeError('Refuse overwrite nonempty output; use --out unique path')
 out.mkdir(parents=True,exist_ok=True);mh=sha(E/'REPLAY_MANIFEST.json');percase=[];layers=[];cfg=[];verified=0;skipped=0
 for row in labels:
  for arm in ARMS:
   record=load(E/'output/replay/completed'/row['case_id']/(arm+'.json.gz'));generation=load(E/'output/generation/completed'/row['case_id']/(arm+'.json.gz'))
   validate_replay(record,row,generation,mh)
   if not row['candidates']:skipped+=1;continue
   x,y,z=reduce_case(record,row);percase.extend(x);layers.extend(y);cfg.extend(z);verified+=1
 summary=aggregate(percase,('arm','class_name','stage','view'),('rank','ties','log_probability','probability'))
 layer_summary=aggregate(layers,('arm','class_name','layer'),('rank','ties','log_probability','probability'))
 cfg_summary=aggregate(cfg,('arm','class_name','stage'),('rank','probability','allowed_fraction','allowed_only_log_probability')) if cfg else []
 chair=chair_summary(rows(E/'labels/all_arm_mentions.jsonl.gz'))
 paired_rows,paired_summary=paired_differences(percase)
 write_csv(out/'paired_image_differences.csv.gz',paired_rows);write_csv(out/'paired_summary.csv',paired_summary)
 write_csv(out/'per_image_stage_metrics.csv.gz',percase);write_csv(out/'per_image_layer_metrics.csv.gz',layers);write_csv(out/'stage_summary.csv',summary);write_csv(out/'layer_summary.csv',layer_summary);write_csv(out/'pai_cfg_per_image.csv.gz',cfg);write_csv(out/'pai_cfg_summary.csv',cfg_summary);write_csv(out/'chair_mentions.csv',chair)
 payload=dict(created_utc=now(),status='FULL',n_images=500,n_arms=2000,verified_nonempty_rank_arms=verified,no_eligible_skipped_arms=skipped,label_freeze_sha256=sha(E/'labels/LABEL_FREEZE.json'),replay_manifest_sha256=mh,analysis_script_sha256=sha(__file__),eligible_image_counts=frozen['eligible_image_counts'],rules=RULES,stage_summary=summary,layer_summary=layer_summary,pai_cfg_summary=cfg_summary,chair_mentions=chair,paired_summary=paired_summary)
 write_json(out/'SUMMARY.json',payload)
 if not a.no_plots:plots(out,summary,layer_summary,chair)
 report=['# COCO500 natural trajectory: object-mention ranking','', 'This is a VISTA-inspired object-only study on a new frozen COCO500 subset, not a replication of the original exact 500 images or GPT-4o labels.','', 'Four methods follow their own natural generation. Common targets were frozen from CoT and official COCO GT before rank replay. Rank changes here describe different resulting trajectories; they are not same-prefix causal effects.','', 'Classes: GT-present and mentioned / GT-present but unmentioned / mentioned but GT-absent. CHAIR mention rules include uncertain and negated mentions; do not call every GT-absent mention an affirmative hallucination.','', 'Full 500 x 4 records are present. Empty target classes are excluded only from that class denominator, never counted as zero.','', '| Method | CHAIR-s mention fraction | CHAIR-i mention fraction | Truncated |','|---|---:|---:|---:|']
 for q in chair:report.append('|%s|%.4f|%.4f|%d|'%(q['arm'],q['chair_s'],q['chair_i'] or 0,q['n_truncated']))
 report+=['','Main hidden-state figures use the original-image branch. PAI final CFG scores are separately reported in pai_cfg_summary.csv.','', 'Sources: [VISTA official paper](https://proceedings.mlr.press/v267/li25ca.html), [VISTA code](https://github.com/LzVv123456/VISTA), [original CHAIR vocabulary](https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt).']
 (out/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 write_json(out/'RECEIPT.json',{'utc':now(),'status':'complete','files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
 print(json.dumps({'output':str(out),'verified_rank_arms':verified,'skipped_empty_rank_arms':skipped,'class_denominators':frozen['eligible_image_counts']}))

def plots(out,summary,layers,chair):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 colors=dict(zip(ARMS,['#222222','#777777','#126b93','#d28032']))
 fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
 for row,view in enumerate(('last5','final')):
  for col,cls in enumerate(CLASSES):
   ax=axes[row,col]
   for arm in ARMS:
    xs=[next((r for r in summary if r['arm']==arm and r['class_name']==cls and r['stage']==stage and r['view']==view),None) for stage in STAGES]
    ax.plot(range(3),[r['rank'] if r else np.nan for r in xs],marker='o',label=arm,color=colors[arm])
   n=next((r['n_images'] for r in summary if r['class_name']==cls and r['view']==view),0)
   ax.set_title('%s\n%s; n=%d images'%(CLASS_NAMES[cls],view,n));ax.set_xticks(range(3),STAGES);ax.set_ylabel('Mean vocabulary rank (lower = higher score)');ax.grid(alpha=.15)
 axes[0,0].legend();fig.suptitle('Shared CoT object targets on each method own natural trajectory',fontsize=14)
 fig.savefig(out/'01_three_stage_rank.png',dpi=180);fig.savefig(out/'01_three_stage_rank.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
 for ax,cls in zip(axes,CLASSES):
  for arm in ARMS:
   rs=sorted([r for r in layers if r['arm']==arm and r['class_name']==cls],key=lambda r:r['layer'])
   ax.plot([r['layer'] for r in rs],[r['rank'] for r in rs],label=arm,color=colors[arm])
  ax.set_title(CLASS_NAMES[cls]);ax.set_xlabel('Layer');ax.set_ylabel('Mean vocabulary rank');ax.grid(alpha=.15)
 axes[0].legend();fig.suptitle('28-layer readout: original-image hidden states; natural trajectories')
 fig.savefig(out/'02_all_layers_rank.png',dpi=180);fig.savefig(out/'02_all_layers_rank.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
 for ax,metric,title in zip(axes,('chair_s','chair_i'),('Captions with GT-absent mentions','GT-absent / all object mentions')):
  rs=[next(r for r in chair if r['arm']==arm) for arm in ARMS];values=[r[metric]*100 if r[metric] is not None else 0 for r in rs]
  ax.bar(ARMS,values,color=[colors[x] for x in ARMS]);ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,max(values+[1])*1.22)
  for i,v in enumerate(values):ax.text(i,v,'%.1f'%v,ha='center',va='bottom')
 fig.suptitle('CHAIR object-mention statistics: full generated text, 500 images')
 fig.savefig(out/'03_object_mentions.png',dpi=180);fig.savefig(out/'03_object_mentions.pdf');plt.close(fig)

def self_test(a):
 tests=[]
 chair=make_extractor()
 for text,expected in [('A baby bird near a toilet seat.',['bird','toilet']),('Two men and dogs beside a traffic light.',['dog','traffic light']),('<think>No dogs, perhaps a cat.</think>A cat sits.', ['dog','cat','cat']),('A laptop computer and a passenger jet.',['laptop','airplane'])]:
  ms=extract_mentions(text,chair);assert [m['canonical'] for m in ms]==expected;assert all(m['surface']==text[m['start']:m['end']] for m in ms);tests.append(text)
 for n in (1,2,3,8,24,512):
  p=point_plan(n);assert len(set(p))==len(p) and all(0<=s<n for s,stage in p)
 ts=[{'class':'decoded_genuine','token_ids':[1],'exclusion_reasons':[]},{'class':'hallucinated','token_ids':[1],'exclusion_reasons':[]},{'class':'hidden_genuine','token_ids':[2],'exclusion_reasons':[]}];add_exclusions(ts);assert [t['rank_primary_eligible'] for t in ts]==[False,False,True]
 avg=aggregate([dict(case_id='a',group='x',n_words=100,v=1),dict(case_id='b',group='x',n_words=1,v=9)],('group',),('v',));assert avg[0]['v']==5
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(load(E/'model.json')['path'],local_files_only=True,use_fast=True)
 text='<think>Two dogs stand near a car.</think> A cat is beside the car.'
 record=dict(text_with_special=text,generated_token_ids=tok.encode(text,add_special_tokens=False),case_id='synthetic')
 targets,ms=targets_for_case(record,{'dog','car','person'},chair,tok)
 assert {q['canonical']:q['class'] for q in targets}=={'dog':'decoded_genuine','car':'decoded_genuine','cat':'hallucinated','person':'hidden_genuine'}
 assert next(q for q in targets if q['canonical']=='dog')['surface']=='dogs'
 record['generated_token_ids']=record['generated_token_ids']+[1]
 bad_targets,_=targets_for_case(record,{'dog','car','person'},chair,tok)
 assert all(not q['rank_primary_eligible'] for q in bad_targets if q['origin']=='actual_cot_mention')
 assert next(q for q in bad_targets if q['canonical']=='person')['rank_primary_eligible']
 assert record['_label_roundtrip_exact'] is False
 p=Path(a.out) if a.out else E/'self_audit/vista_attention_audit'/('CPU_TEST_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json')
 write_json(p,{'utc':now(),'passed':True,'scope':'CPU synthetic token alignment, CHAIR parity, qualifier phrases, control tags, collision exclusions, sample equal-weight mean, point plans','test_texts':tests,'synthetic_targets':targets,'script_sha256':sha(__file__),'reference_sha256':REFERENCE_SHA,'real_scientific_results':False});print(str(p))

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['labels','analyze','self-test']);p.add_argument('--out');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
 {'labels':label,'analyze':analyze,'self-test':self_test}[a.command](a)
if __name__=='__main__':main()
