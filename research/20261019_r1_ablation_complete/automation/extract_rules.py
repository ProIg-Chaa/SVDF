"""Gold-blind COW extraction; upstream sources unchanged."""
import re,json,hashlib,importlib.util
from pathlib import Path
R=Path('/home/bingxing2/home/scx9ftv/gushuo')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
PA=load(R/'experiments/20260930_repeat_visual_full3/evaluation/unified_parser.py','run_parser')
TAG=load(R/'experiments/20261005_openvl_output_once_full3/evaluation/answer_tag_r2_20261005/protocol_apply.py','run_tag')
OLD_OPTIONS=PA.parse_options
def line_options(options,question=None):
 for text in [options or '',question or '']:
  ms=list(re.finditer(r'(?m)^\s*(?:Options:\s*)?(?:\(([A-Ha-h])\)|([A-Ha-h])[.:])\s+',text))
  if len(ms)>=2:
   result={}
   for i,m in enumerate(ms):
    k=(m.group(1) or m.group(2)).upper();v=text[m.end():ms[i+1].start() if i+1<len(ms) else len(text)].strip()
    if k in result or not v:break
    result[k]=v
   else:return result
 return OLD_OPTIONS(options,question)
PA.parse_options=line_options;TAG.PARSER=PA
def canonical(q,o,p):
 if p is None:return None
 opts=line_options(o,q);v=str(p).strip()
 if re.fullmatch('[A-Ha-h]+',v) and all(a.upper() in opts for a in v):return ''.join(sorted(set(v.upper())))
 return TAG.canonical_prediction(q,o,v)
def clean(t):
 t=re.sub(r'<\|[^>]*\|>','',t).replace('**','').replace('__','').replace(chr(96),'')
 t=re.sub(r'</?think\s*>','',t,flags=re.I)
 for _ in range(8):t=re.sub(r'\\(?:text|mathrm|mathbf|textbf|operatorname)\s*\{([^{}]*)\}',r'\1',t)
 return t.strip(' \t\r\n$\\().,:;!{}[]')
def selected_body(body,opts):
 z=clean(body)
 if re.fullmatch(r'[A-Ha-h]',z) and z.upper() in opts:return z.upper()
 if re.fullmatch(r'[A-Ha-h](?:\s*(?:,|and|&|、|和|与)\s*[A-Ha-h])+',z):
  aa=re.findall('[A-Ha-h]',z.replace('and',''));return ''.join(sorted(set(a.upper() for a in aa))) if all(a.upper() in opts for a in aa) else None
 matches=[k for k,v in opts.items() if PA.normalize(z)==PA.normalize(v)]
 if len(matches)==1:return matches[0]
 if re.fullmatch(r'[A-H]{2,8}',z) and all(a.upper() in opts for a in z):return ''.join(sorted(set(z.upper())))
 if re.search(r'(?i)\b(?:or|versus|either)\b',z):return None
 m=re.match(r'^(?:\(([A-Ha-h])\)|([A-Ha-h])[.):])\s+',z)
 if m and (m.group(1) or m.group(2)).upper() in opts:return (m.group(1) or m.group(2)).upper()
 matches=[k for k,v in opts.items() if PA.normalize(z)==PA.normalize(v)]
 return matches[0] if len(matches)==1 else None
def balanced_boxes(t):
 out=[]
 for m in re.finditer(r'\\boxed\s*\{',t):
  i=m.end();start=i;depth=1
  while i<len(t) and depth:
   depth+=int(t[i]=='{')-int(t[i]=='}');i+=1
  if depth==0:out.append((m.start(),i,t[start:i-1]))
 return out
def result(pred,why,evidence=None):return dict(prediction=pred,reason=why,evidence=evidence,status='resolved')
def resolve(x):
 t=x['text'];q=x['question'];o=x.get('options','') or '';opts=line_options(o,q)
 if not opts and x['dataset'] not in ['vstar','mmvp','realworldqa','visulogic'] and not o.strip():
  o='\n'.join('(%s) IMAGE_OPTION_%s'%(a,a) for a in 'ABCDE');opts=line_options(o,q)
 tags=list(re.finditer(r'<answer\s*>(.*?)</answer\s*>',t,re.I|re.S))
 if tags:
  final=tags[-1];suffix=re.sub(r'<\|[^>]*\|>','',t[final.end():]).strip()
  closes=list(re.finditer(r'</think\s*>',t,re.I));start=closes[-1].end() if closes else 0
  if final.start()>=start and not re.search(r'<think\s*>',t[start:final.start()],re.I) and not suffix.strip('.!'):
   outside=[m for m in tags if m.start()>=start];vals=[]
   for m in outside:
    value=selected_body(m.group(1),opts) if opts else None
    if value is None and not opts:
     body=clean(m.group(1))
     if re.fullmatch(r'(?i)yes|no',body):value=body.lower()
     elif re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:\s*%)?',body):value=body
     elif PA.answer_type(q)!='yesno' and len(body)<120 and not re.search(r'(?i)\b(?:or|cannot|unable|uncertain)\b',body):value=body
    vals.append(value)
   if vals and all(v is not None for v in vals) and len(set(vals))==1:return result(canonical(q,o,vals[-1]),'cow_explicit_terminal_answer_tag',final.group(1))
 boxes=balanced_boxes(t)
 if boxes:
  st,en,body=boxes[-1];suffix=clean(t[en:])
  if not suffix:
   value=selected_body(body,opts)
   if value is not None:return result(canonical(q,o,value),'cow_explicit_terminal_box',body)
   if not opts and len(clean(body))<120:return result(canonical(q,o,clean(body)),'cow_explicit_terminal_open_box',body)
 z=clean(t);cue=re.compile(r'(?i)\b(?:final|correct|best|most suitable|most appropriate)?\s*(?:answer|choice|option|statement)\s*(?:should\s+be|would\s+be|is|:|=)\s*[:=]?\s*|(?:最终答案|正确答案|答案)\s*(?:是|为|[:：])\s*')
 cues=list(cue.finditer(z))
 if cues:
  tail=clean(z[cues[-1].end():])
  if len(tail)<350:
   m=re.match(r'^(?i:options?\s+)?((?:[A-H]|\([A-Ha-h]\)|[A-Ha-h](?=[).:]))(?:\s*(?:,|(?i:and)|&|、|和|与)\s*(?:[A-H]|\([A-Ha-h]\)|[A-Ha-h](?=[).:])))*)((?!\w))',tail)
   if m:
    value=selected_body(m.group(1).replace('(','').replace(')',''),opts);remain=tail[m.end():]
    if value is not None and not re.match(r'\s*(?:or|/|versus)\s*[A-H]\b',remain,re.I):return result(canonical(q,o,value),'cow_terminal_answer_cue',tail)
 blocks=[b.strip() for b in re.split(r'\n\s*\n',t) if b.strip()]
 for j in range(max(0,len(blocks)-2),len(blocks)):
  block=clean(blocks[j]);value=selected_body(block,opts)
  if value is None:continue
  rest=clean('\n\n'.join(blocks[j+1:]));before='\n\n'.join(blocks[:j])
  if len(block)<220 and (not rest or len(rest)<220 and not re.search(r'(?i)\b(?:answer|option|choice|instead)\b',rest)) and re.search(r'(?i)(?:answer|matches|choose|choice|likely|would be|is:|analysis|given)',before[-900:]):return result(canonical(q,o,value),'cow_terminal_selected_option_paragraph',block)
 parsed=PA.parse(q,o,t);fallback=parsed.get('prediction') if parsed.get('status')=='ok' else None
 tag=TAG.apply_answer(q,o,t,fallback)
 if tag['eligible'] or tag.get('exact_text_manual_review') or not tag['actual_final_tag'] and fallback is not None:return result(canonical(q,o,tag['prediction']),'upstream_r2_with_line_option_adapter',tag.get('tag'))
 end=clean(t[-700:])
 if re.search(r'(?i)\b(?:cannot|can.not|unable to|not possible to|impossible to)\s+(?:determine|answer|identify|tell)|(?:无法确定|无法判断|无法回答)',end) and not re.search(r'(?i)(?:answer|choose|select|choice)\s*(?:is|:|would be)?\s*\(?[A-H]\b',end):return result(None,'cow_explicit_terminal_abstention',end)
 # A final conclusion region can be parsed without collecting earlier explored alternatives.
 region=re.split(r'</think\s*>',t,flags=re.I)[-1]
 if not region.strip():region=t
 paras=[b.strip() for b in re.split(r'\n\s*\n',region) if b.strip()]
 if paras:
  final=clean(paras[-1])
  if len(final)<650 and re.search(r'(?i)(?:therefore|thus|hence|given|based|conclu|answer|choice|option|^yes\b|^no\b)',final):
   z=PA.parse(q,o,final)
   if z.get('status')=='ok' and z.get('prediction') is not None:
    return result(canonical(q,o,z['prediction']),'cow_final_conclusion_region',final)
 return dict(prediction=None,reason='pending_fulltext_review',status='pending',evidence=None)
