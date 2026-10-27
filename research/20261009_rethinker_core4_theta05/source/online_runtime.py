from compact_vectors import save_vectors
"""Frozen faithful h2, four native short frames and exactly one selected tail."""
import gc,hashlib,sys,threading,time
from pathlib import Path
import torch
import base,legacy_action,v2_core,h2_rules
from multi_device import synchronize,reset_peak_memory,peak_memory,peak_allocated,peak_reserved
from parked_probability import ProbabilityWorker
from state_isolation import StateIsolation,cache_signature,assert_disjoint,tensor_sha

ARMS={'talr':(0.,None),'true_0p5x4':(.5,'true'),'true_1p0x4':(1.,'true'),'random_1p0x4':(1.,'random')}


class SlimTrace(list):
    def append(self,row):
        super().append({k:row[k] for k in ('step','token_id','lead_refinement_active','route_override_active') if k in row})


class ProbabilitySink:
    def __init__(self,event,trace,pause,vocab_size):
        self.event=event;self.trace=trace;self.pause=pause;self.vocab_size=vocab_size;self.prob=None;self.native_sha=None;self.prefix=None;self.aux=0
    def append(self,row):
        if row.get('kind')=='step0_geometry':
            assert row['step']==0 and row['sample_index']==0 and self.aux==0
            self.aux+=1;return
        assert row['kind']=='logits' and row['step']==self.event+2 and row['sample_index']==0 and self.prob is None
        prefix=[x['token_id'] for x in self.trace]
        assert len(prefix)==self.event+2,'Pause must precede current token choice'
        raw=row['probs'];assert raw.dtype==torch.float32 and raw.shape==(self.vocab_size,)
        self.native_sha=tensor_sha(raw);self.prob=raw.half().contiguous();self.prefix=prefix
        frame=sys._getframe(1);live=frame.f_locals
        assert live['step']==self.event+2 and live['past_key_values'] is not None
        assert torch.equal(live['probs_original'][0].detach().float().cpu(),raw)
        assert [len(x)-n for x,n in zip(live['all_generated'],live['prompt_lens'])]==[self.event+2]
        self.pause(dict(event=self.event,step=self.event+2,prefix=prefix,fp16_sha256=tensor_sha(self.prob)),frame)


class CostMeter:
    def __init__(self,model):
        self.model=model;self.stages={};self.records={};self.main_stage='setup';self.handles=[]
        self.handles.append(model.register_forward_pre_hook(self.observe,with_kwargs=True))
    def observe(self,module,args,kwargs):
        label=self.stages.get(threading.current_thread().name,self.main_stage)
        r=self.records.setdefault(label,dict(forwards=0,prefill_forwards=0,decoder_forwards=0,prefill_input_positions=0,decoder_positions=0,raw_image_prefills=0))
        r['forwards']+=1
        if kwargs.get('past_key_values') is None:
            r['prefill_forwards']+=1;ids=kwargs.get('input_ids')
            assert ids is not None
            r['prefill_input_positions']+=int(ids.numel())
            r['raw_image_prefills']+=int(kwargs.get('pixel_values') is not None and kwargs['pixel_values'].numel()>0)
            if kwargs.get('image_grid_thw') is not None:
                grid=kwargs['image_grid_thw'].detach().cpu()
                r['image_grid_thw']=grid.tolist()
                r['image_pixel_equivalent']=int(grid.prod(dim=-1).sum())*14*14
        else:
            r['decoder_forwards']+=1
            r['decoder_positions']+=int(kwargs['inputs_embeds'].shape[0]*kwargs['inputs_embeds'].shape[1])
    def close(self):
        for h in self.handles:h.remove()


def legacy_geometry(model,processor,tokenizer,sample,math_ids,threshold,event,prefix,folder,meter):
    key=sample['dataset']+':'+str(sample['id']);stem=hashlib.sha256(key.encode()).hexdigest()[:20]
    started=time.perf_counter()
    mask=base.sweep.create_mean_mask(sample['image'],folder/'masks'/(stem+'.png'))
    meter.main_stage='capture_true'
    true=legacy_action.capture_talr_vector(model,processor,tokenizer,sample,sample['image'],prefix,event,math_ids,refinement_entropy_threshold=threshold)
    meter.main_stage='capture_mask'
    masked=legacy_action.capture_talr_vector(model,processor,tokenizer,sample,mask,prefix,event,math_ids,refinement_entropy_threshold=threshold)
    assert torch.equal(true['hard_embedding'],masked['hard_embedding'])
    assert true['selected_token_id']==masked['selected_token_id']==prefix[event]
    direction,rho,raw,hard,soft=legacy_action.matched_visual_direction(true,masked)
    seed=legacy_action.stable_seed(key,42);random=legacy_action.random_direction_like(direction,rho,seed)
    assert torch.isfinite(direction).all() and torch.isfinite(random).all()
    if raw>1e-12:assert abs(float(direction.norm())-rho)<=max(1e-5,1e-4*max(rho,1.))
    assert abs(float(random.norm())-rho)<=max(1e-5,1e-4*max(rho,1.))
    synchronize()
    return {'true':direction,'random':random},dict(target_norm=rho,raw_norm=raw,hard_norm=hard,soft_norm=soft,true_norm=float(direction.norm()),
        random_norm=float(random.norm()),random_seed=seed,true_fp32_sha=tensor_sha(direction),random_fp32_sha=tensor_sha(random),
        true_actual_cast_sha=tensor_sha(direction.to(device=model.device,dtype=model.dtype)),
        random_actual_cast_sha=tensor_sha(random.to(device=model.device,dtype=model.dtype)),shared_hard_anchor=True,finite=True,
        capture_horizon=event+1,continuation_horizon=1024,nominal_rho_random_even_if_true_zero=True,seconds=time.perf_counter()-started)


def run_sample(model,processor,tokenizer,math_ids,threshold,sample,profile,folder):
    started=time.perf_counter();synchronize();reset_peak_memory()
    assert base.sha(Path(sample['image']))==sample['image_sha256']
    meter=CostMeter(model);isolation=StateIsolation(model);workers=[]
    initial_hooks=tuple(model._forward_pre_hooks.values());geometry={};checks={};tails=0;decision=None;event=-1;prefix=[];capture_seconds=0.
    def plain():
        meter.main_stage='default_off'
        inputs,length=base.prepare(processor,sample,sample['image'],model.device)
        with torch.no_grad():out=base.generate_lead(model,tokenizer,**inputs,**base.sweep.talr_kwargs(math_ids,1024,**base.controller(threshold)))[0]
        return [int(x) for x in out[length:].tolist()]
    try:
        if profile=='default_off':
            tokens=plain();chosen='talr';states={};probs=None;shorts={};decision=dict(faithful_accept=None,diagnostic_profile='default_off')
        else:
            meter.main_stage='discovery';ds=time.perf_counter()
            event,discovered,trace,tape=v2_core.discover(model,processor,tokenizer,sample,math_ids,threshold)
            discovery_seconds=time.perf_counter()-ds
            if event<0:
                tokens=discovered;chosen='talr';states={};probs=None;shorts={};decision=h2_rules.decide(None,None,None,None)
                checks['noevent_no_counterfactual_calls']=True
            else:
                prefix=discovered;assert len(prefix)==event+1
                if profile=='zero':
                    actions={'true':torch.zeros_like(model.get_input_embeddings().weight[0].detach().float()),'random':None}
                    names=['true_1p0x4']
                else:
                    actions,geometry=legacy_geometry(model,processor,tokenizer,sample,math_ids,threshold,event,prefix,folder,meter)
                    capture_seconds=geometry['seconds'];names=list(ARMS)
                    if profile=='order_reverse':names.reverse()
                base.append(folder/'action_states.jsonl',dict(dataset=sample['dataset'],id=sample['id'],event=event,prefix=prefix,tape_hash=tape.fingerprint(),geometry=geometry,
                    no_old_event_or_prefix_loaded=True,complete_reference_outputs_available=False,profile=profile))
                initial_state=isolation.snapshot();byname={};signatures={}
                for name in names:
                    strength,kind=ARMS[name];action=None if kind is None else actions[kind]
                    slim=SlimTrace();item={'sink':None,'trace':slim,'consumed':[],'count':0}
                    def target(pause,name=name,action=action,strength=strength,item=item):
                        assert not torch.is_inference_mode_enabled(),'Frozen runtime uses no_grad, not inference_mode'
                        inputs,length=base.prepare(processor,sample,sample['image'],model.device)
                        sink=ProbabilitySink(event,item['trace'],pause,model.config.vocab_size);item['sink']=sink
                        kw=base.sweep.talr_kwargs(math_ids,1024,**base.controller(threshold))
                        kw.update(forced_prefix_ids=prefix,token_trace=item['trace'],trace_topk=0,capture_logits_steps=[event+2],capture_logits_sink=sink)
                        if action is not None:
                            kw.update(trace_route_override_step=event,trace_route_override_kind='method_plus_external_residual',trace_route_override_mix_lambda=strength,
                                trace_route_override_duration=4,trace_external_route_vector=action,trace_external_route_source='matched_visual_action')
                        def route(module,args,kwargs):
                            if threading.current_thread().name!='h2_'+name:return
                            step=item['count']-1;item['count']+=1
                            if action is not None and event<=step<event+4 and kwargs.get('inputs_embeds') is not None:
                                x=kwargs['inputs_embeds'];assert torch.isfinite(x).all()
                                item['consumed'].append(dict(step=step,norm=float(x.float().norm()),dtype=str(x.dtype),shape=list(x.shape)))
                        handle=model.register_forward_pre_hook(route,with_kwargs=True)
                        try:
                            with torch.no_grad():out=base.generate_lead(model,tokenizer,**inputs,**kw)[0]
                        finally:handle.remove()
                        return [int(x) for x in out[length:].tolist()]
                    worker=ProbabilityWorker('h2_'+name,target,isolation.snapshot,isolation.restore_checked,initial_state)
                    workers.append(worker);byname[name]=(worker,item);meter.stages[worker.name]=name+'_short'
                    status=worker.start();assert status in ('paused','complete')
                    if status=='paused':
                        signature=cache_signature(worker.frame)
                        assert signature['step']==event+2 and signature['generated_lengths']==[event+2]
                        assert [x['token_id'] for x in item['trace']][:event+1]==prefix
                        assert item['count']==event+3
                        signatures[name]=signature
                    else:
                        assert len(worker.result)<=event+2 and worker.result[-1]==tokenizer.eos_token_id
                    # Each previously parked native cache must be untouched by later branches.
                    for previous,signature in signatures.items():assert cache_signature(byname[previous][0].frame)==signature
                assert_disjoint(list(signatures.values()));checks['native_KV_disjoint_and_parked_unchanged']=True
                checks['RNG_RoPE_restored_before_each_active_branch']=True
                if profile=='zero':
                    chosen='true_1p0x4';decision=dict(faithful_accept=None,diagnostic_profile='zero_action');probs=None
                else:
                    probs={name:byname[name][1]['sink'].prob for name in ARMS}
                    ds=time.perf_counter();decision=h2_rules.decide(*(probs[name] for name in ARMS));decision_seconds=time.perf_counter()-ds
                    chosen='true_1p0x4' if decision['faithful_accept'] else 'talr'
                    assert chosen in ('talr','true_1p0x4')
                    stem=hashlib.sha256((sample['dataset']+':'+str(sample['id'])).encode()).hexdigest()[:20]
                    save_vectors(dict(event=event,horizon=2,prefix=prefix,probs=probs,
                        conditioning_prefix={n:byname[n][1]['sink'].prefix for n in ARMS},
                        native_sha256={n:byname[n][1]['sink'].native_sha for n in ARMS}),folder/'vectors'/(stem+'_h2.pt'))
                    base.append(folder/'decisions_before_tail.jsonl',dict(dataset=sample['dataset'],id=sample['id'],event=event,decision=decision,chosen=chosen,
                        chosen_tail_started=False,horizon=2,decision_step=event+2,before_current_token=True,
                        fp16_sha256={n:tensor_sha(p) if p is not None else None for n,p in probs.items()},seconds=decision_seconds))
                shorts={name:dict(status=w.status,tokens=[x['token_id'] for x in item['trace']],forward_calls=item['count'],
                    consumed_routes=list(item['consumed']),pauses=w.pauses,active_seconds=w.short_active_seconds if w.status=='paused' else w.active_seconds) for name,(w,item) in byname.items()}
                states={name:signature for name,signature in signatures.items()}
                selected,selected_item=byname[chosen]
                for name,(worker,item) in byname.items():
                    if name!=chosen:
                        worker.discard();assert item['count']==shorts[name]['forward_calls']
                        assert len(item['trace'])<=event+2
                tail_started=time.perf_counter();meter.stages[selected.name]=chosen+'_selected_tail'
                if selected.status=='paused':
                    tails=1;assert selected.resume()=='complete'
                tokens=selected.result;tail_seconds=time.perf_counter()-tail_started
                assert tokens[:event+1]==prefix and [x['token_id'] for x in selected_item['trace']]==tokens
                checks['only_selected_BF_or_zero_diagnostic_resumed']=True
                checks['no_current_token_or_forward_repeated_on_resume']=True
                checks['selected_consumed_route_slots_not_restarted']=all(event<=x['step']<event+4 for x in selected_item['consumed'])
                checks['selected_frame_forward_matches_token_count']=selected_item['count']==len(tokens)
                assert checks['selected_frame_forward_matches_token_count']
        assert tuple(model._forward_pre_hooks.values())==initial_hooks
        synchronize()
        result=dict(dataset=sample['dataset'],id=sample['id'],pair_id=sample['pair_id'],image_sha256=sample['image_sha256'],profile=profile,
            event=event,prefix=prefix,chosen_branch=chosen,decision=decision,generated_token_ids=tokens,
            text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(),runtime_error=None,replay_mismatch=False,
            geometry=geometry,checks=checks,short_branches=shorts,cache_signatures=states,resumed_branch_count=tails,
            cost=dict(stages=meter.records,wall_seconds=time.perf_counter()-started,capture_seconds=capture_seconds,
                discovery_seconds=locals().get('discovery_seconds',0.),selected_tail_seconds=locals().get('tail_seconds',0.),
                decision_seconds=locals().get('decision_seconds',0.),peak_gpu_allocated_bytes=peak_allocated(),peak_gpu_reserved_bytes=peak_reserved(),per_device_peak_memory=peak_memory()),
            logical_horizon=1024,legacy_geometry_horizon=event+1 if event>=0 else None,configured_h2_step=event+2 if event>=0 else None,
            actual_stop_step=event+2 if any(w.pauses for w in workers) else None,h2_readout_available=bool(decision.get('available',False)),
            native_active_frame_resume=bool(tails),shared_KV_prefill_optimization=False,full_HN_tails_generated=False)
        base.append(folder/'results.jsonl',result)
        print('ONLINE_H2_COMPLETE '+sample['dataset']+':'+str(sample['id'])+' '+profile,flush=True)
    finally:
        fatal_timeout=any(worker.status=='timed_out' for worker in workers)
        for worker in workers:
            if worker.status=='paused':worker.discard()
            if worker.status!='timed_out':assert not worker.thread.is_alive()
        # The exception escapes entry and ends the process; do not touch a timed-out model.
        if not fatal_timeout:
            meter.close();workers.clear();gc.collect()
