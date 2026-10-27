"""CPU only. Freeze token endpoints from baseline and pre-treatment fact annotations."""
from common import *
import datetime

def main():
 from transformers import AutoTokenizer
 m,mh=check_manifest();tok=AutoTokenizer.from_pretrained(m['checkpoint']['path'],local_files_only=True)
 facts=rows(E/'annotation/facts_frozen.jsonl');bases={str(x['id']):x for x in rows(E/'annotation/baseline78.jsonl')}
 assert len(facts)==len(bases)==78 and {str(x['id']) for x in facts}==set(bases)
 out=[]
 for a in facts:
  b=bases[str(a['id'])];bt=b['token_ids'];text=tok.decode(bt,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert text==b['text'];r=dict(dataset='mmvp',id=str(a['id']),event=b['event'],annotation=a,eligible=False,excluded_reason=None)
  reason=None
  if not a.get('probe_requested',a.get('probe_eligible',False)):reason='not_requested_by_blind_fact_review'
  elif a.get('earlier_same_error',False):reason='same_error_already_in_prefix'
  else:
   p=a['prefix_text'];wrong=a['wrong_candidate'];correct=a['correct_candidate'];boundaries={tok.decode(bt[:k],skip_special_tokens=False,clean_up_tokenization_spaces=False):k for k in range(len(bt)+1)}
   shift=''
   if p not in boundaries:
    while p and p[-1].isspace() and p not in boundaries:shift=p[-1]+shift;p=p[:-1]
   wrong=shift+wrong;correct=shift+correct
   if p not in boundaries:reason='prefix_not_exact_token_boundary'
   elif not text.startswith(p+wrong):reason='wrong_candidate_not_baseline_continuation'
   elif a.get('baseline_quote') and a['baseline_quote'] not in text:reason='quote_not_in_baseline'
   else:
    endpoint=boundaries[p];candidates=[]
    for role,word in [('wrong',wrong),('correct',correct)]:
     ids=tok.encode(word,add_special_tokens=False)
     assert tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==word,('candidate_decode_mismatch',a['id'],role)
     candidates.append(dict(role=role,text=word,word_text=word,token_ids=ids,word_token_count=len(ids)))
    if endpoint<b['event']+4:reason='four_feedbacks_not_yet_consumed'
    elif any(endpoint+len(x['token_ids'])>1024 for x in candidates):reason='candidate_exceeds_horizon'
    elif not candidates[0]['token_ids'] or not candidates[1]['token_ids']:reason='empty_candidate'
    elif bt[endpoint:endpoint+len(candidates[0]['token_ids'])]!=candidates[0]['token_ids']:reason='wrong_candidate_tokenization_not_baseline_exact'
    elif tok.eos_token_id in bt[:endpoint]:reason='endpoint_after_eos'
    else:r.update(eligible=True,prefix_text=p,prefix_token_ids=bt[:endpoint],probe_token_index=endpoint,candidates=candidates,whitespace_moved_to_candidates=shift)
  r['excluded_reason']=reason;out.append(r)
 eligible=[x for x in out if x['eligible']];single=next((x['id'] for x in eligible if all(len(c['token_ids'])==1 for c in x['candidates'])),None);multi=next((x['id'] for x in eligible if any(len(c['token_ids'])>1 for c in x['candidates']) and x['id']!=single),None)
 with (E/'probe_registry.jsonl').open('x') as f:
  for x in out:f.write(json.dumps(x)+'\n')
 pm=dict(prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_manifest_sha256=mh,annotation_sha256=sha(E/'annotation/facts_frozen.jsonl'),baseline_annotation_sha256=sha(E/'annotation/baseline78.jsonl'),probe_registry_sha256=sha(E/'probe_registry.jsonl'),eligible_ids=[x['id'] for x in eligible],eligible_count=len(eligible),smoke_ids=[single,multi],code_sha256={n:sha(E/n) for n in ['prepare_probes.py','run_probes.py','verify_probe_logits.py']},arms=['original','weakened','attention2','attention4','raw_sI','real'],endpoint='Before first explicitly wrong fact on unchanged baseline text; no auxiliary suffix; event unchanged; after four feedbacks.',selection='Annotation-only; no intervention outcomes used',probability='Full-vocabulary phrase joint log probability, no candidate renormalization',minimum_population_claim_n=20)
 save(E/'probe_manifest.json',pm);print(json.dumps(pm,indent=2))
 if single is None or multi is None:print('SMOKE_SETUP_MISSING: cannot launch probe smoke without both single-token and multi-token candidates; no invented target.')
if __name__=='__main__':main()
