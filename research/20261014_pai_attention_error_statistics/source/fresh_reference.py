from compact_vectors import save_vectors
"""Smoke-only uninterrupted native B/F reference; never an online decision input."""
import hashlib,time
import torch,base,v2_core
from online_runtime import CostMeter,SlimTrace,legacy_geometry
from state_isolation import StateIsolation,tensor_sha
from multi_device import synchronize,reset_peak_memory,peak_memory

class ReferenceSink:
    def __init__(self,event,vocab):self.event=event;self.vocab=vocab;self.prob=None;self.native_sha=None;self.aux=0
    def append(self,row):
        if row.get('kind')=='step0_geometry':
            assert row['step']==0 and self.aux==0;self.aux+=1;return
        assert row['kind']=='logits' and row['step']==self.event+2 and self.prob is None
        assert row['probs'].dtype==torch.float32 and row['probs'].shape==(self.vocab,)
        self.prob=row['probs'].half().contiguous();self.native_sha=tensor_sha(row['probs'])

def run_reference(model,processor,tokenizer,math_ids,threshold,sample,folder):
    assert base.sha(__import__('pathlib').Path(sample['image']))==sample['image_sha256']
    synchronize();reset_peak_memory();started=time.perf_counter();meter=CostMeter(model)
    try:
        meter.main_stage='discovery';event,prefix,trace,tape=v2_core.discover(model,processor,tokenizer,sample,math_ids,threshold)
        geometry={};branches={};vectors={};native_hashes={}
        if event<0:
            for name in ('talr','true_1p0x4'):branches[name]=dict(generated_token_ids=prefix,text=tokenizer.decode(prefix,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(),h2_readout_available=False)
        else:
            actions,geometry=legacy_geometry(model,processor,tokenizer,sample,math_ids,threshold,event,prefix,folder,meter)
            isolation=StateIsolation(model);state=isolation.snapshot()
            for name in ('talr','true_1p0x4'):
                isolation.restore_checked(state);meter.main_stage=name+'_uninterrupted_full'
                inputs,length=base.prepare(processor,sample,sample['image'],model.device);slim=SlimTrace();sink=ReferenceSink(event,model.config.vocab_size)
                kw=base.sweep.talr_kwargs(math_ids,1024,**base.controller(threshold))
                kw.update(forced_prefix_ids=prefix,token_trace=slim,trace_topk=0,capture_logits_steps=[event+2],capture_logits_sink=sink)
                if name!='talr':kw.update(trace_route_override_step=event,trace_route_override_kind='method_plus_external_residual',trace_route_override_mix_lambda=1.,trace_route_override_duration=4,
                    trace_external_route_vector=actions['true'],trace_external_route_source='matched_visual_action')
                with torch.no_grad():out=base.generate_lead(model,tokenizer,**inputs,**kw)[0]
                tokens=[int(x) for x in out[length:].tolist()];assert tokens[:event+1]==prefix and [x['token_id'] for x in slim]==tokens
                branches[name]=dict(generated_token_ids=tokens,text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(),h2_readout_available=sink.prob is not None)
                vectors[name]=sink.prob;native_hashes[name]=sink.native_sha
            stem=hashlib.sha256((sample['dataset']+':'+str(sample['id'])).encode()).hexdigest()[:20]
            save_vectors(dict(event=event,horizon=2,prefix=prefix,probs=vectors,native_sha256=native_hashes),folder/'vectors'/(stem+'_h2.pt'))
        synchronize()
        base.append(folder/'results.jsonl',dict(dataset=sample['dataset'],id=sample['id'],pair_id=sample['pair_id'],image_sha256=sample['image_sha256'],event=event,prefix=prefix if event>=0 else [],
            profile='fresh_reference',branches=branches,geometry=geometry,decision=None,no_stop_or_resume=True,smoke_only=True,runtime_error=None,replay_mismatch=False,
            cost=dict(stages=meter.records,wall_seconds=time.perf_counter()-started,per_device_peak_memory=peak_memory()),logical_horizon=1024))
    finally:meter.close()
