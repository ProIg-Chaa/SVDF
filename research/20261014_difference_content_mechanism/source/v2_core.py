import gc,hashlib,time
from pathlib import Path
import torch
import base
from action_kernel import bounded_reweight,matched_random
from route_tape import RouteTape
VERSION='state_matched_bounded_reweight_v2_boundary_integration'

def discover(model,processor,tokenizer,sample,math_ids,threshold):
    with RouteTape(model) as tape:
        event,prefix,trace = base.discover(model,processor,tokenizer,sample,math_ids,threshold)
    if event>=0:
        assert len(tape.states)==event and tape.forward_calls==event+1, 'capture/event input count'
    return event,prefix,trace,tape



def locked_capture(model,processor,tokenizer,sample,image,prefix,event,math_ids,threshold,tape):
    with RouteTape(model,tape) as replay:
        value = base.capture(model,processor,tokenizer,sample,image,prefix,event,math_ids,threshold)
    assert replay.fingerprint()==tape.fingerprint(), 'replayed inputs differ'
    assert replay.forward_calls==event+1, 'unexpected model call at capture'
    return value,replay



def run_sample(model,processor,tokenizer,math_ids,sample,threshold,folder,smoke):
    started = time.perf_counter()
    hook_ids = set(model._forward_pre_hooks)
    assert base.sha(Path(sample['image']))==sample['image_sha256']
    event,prefix,trace,tape = discover(model,processor,tokenizer,sample,math_ids,threshold)
    key = sample['dataset']+':'+str(sample['id'])
    checks = dict(current_prefix_only=event>=0,raw_image_present=True,source_image_hash=True)
    geometry, actions = {}, {}
    if event>=0:
        mask = base.sweep.create_mean_mask(sample['image'],folder/'masks'/(key.replace(':','_')+'.png'))
        args = (model,processor,tokenizer,sample)
        true = base.capture(*args,sample['image'],prefix,event,math_ids,threshold)
        natural_mask = base.capture(*args,mask,prefix,event,math_ids,threshold)
        locked, replay = locked_capture(*args,mask,prefix,event,math_ids,threshold,tape)
        assert torch.equal(true['hard_embedding'],natural_mask['hard_embedding'])
        assert torch.equal(true['hard_embedding'],locked['hard_embedding'])
        old,target,raw_norm,_,_ = base.sweep.matched_visual_direction(true,natural_mask)
        new,geometry = bounded_reweight(true['probs'],locked['probs'],model.get_input_embeddings().weight,target)
        actions = dict(existing_residual=old,bounded_reweight=new,
                       norm_matched_random=matched_random(new,target,key))
        geometry.update(existing_raw_norm=raw_norm,existing_norm=old.norm().item(),
            candidate_existing_cosine=torch.nn.functional.cosine_similarity(new.cpu(),old,dim=0).item(),
            candidate_cast_norm=new.to(model.dtype).float().norm().item(),
            mask_natural_locked_probability_l1=(natural_mask['probs']-locked['probs']).abs().sum().item(),
            mask_input_drift_norm_mean=sum(replay.input_drift_norms)/max(1,len(replay.input_drift_norms)),
            mask_input_drift_norm_max=max(replay.input_drift_norms,default=0.),
            taped_generated_inputs=len(tape.states),capture_forward_calls=replay.forward_calls)
        if smoke:
            true_locked,true_replay = locked_capture(*args,sample['image'],prefix,event,math_ids,threshold,tape)
            base.assert_capture_equal(true,true_locked)
            assert true_replay.prefill==tape.prefill
            assert all(n==0 for n in true_replay.input_drift_norms)
            blank = dict(sample,answer='',gold='',native_full_answer='',future_trace=[])
            assert base.public_input(blank)==sample
            e2,p2,_,tape2 = discover(model,processor,tokenizer,blank,math_ids,threshold)
            assert (e2,p2)==(event,prefix) and tape2.fingerprint()==tape.fingerprint()
            t2 = base.capture(model,processor,tokenizer,blank,sample['image'],prefix,event,math_ids,threshold)
            m2,_ = locked_capture(model,processor,tokenizer,blank,mask,prefix,event,math_ids,threshold,tape2)
            base.assert_capture_equal(true,t2)
            base.assert_capture_equal(locked,m2)
            n2,_ = bounded_reweight(t2['probs'],m2['probs'],model.get_input_embeddings().weight,target)
            assert torch.equal(new,n2), 'blank-answer action mismatch'
            checks.update(true_tape_capture_exact=True,blank_answers_decisions_actions_equal=True)
    assert set(model._forward_pre_hooks)==hook_ids, 'capture hook leaked'
    decision = dict(id=sample['id'],dataset=sample['dataset'],pair_id=sample['pair_id'],event=event,
        candidate_version=VERSION,prefix=prefix if event>=0 else [],geometry=geometry,checks=checks,
        tape_hash=tape.fingerprint() if event>=0 else None,
        action_hashes={name:hashlib.sha256(v.detach().cpu().float().numpy().tobytes()).hexdigest() for name,v in actions.items()},
        full_answers_available_at_decision=(event<0),action_constructed=(event>=0),
        no_event_boundary='Discovery completes TALR; no action or selection' if event<0 else None)
    base.append(folder/'decisions.jsonl',decision)
    baseline = base.continuation(model,processor,tokenizer,sample,math_ids,threshold)
    tokens = baseline[1]
    if event>=0:
        assert tokens[:len(prefix)]==prefix, 'discovery prefix mismatch'
        replay = base.continuation(model,processor,tokenizer,sample,math_ids,threshold,prefix)
        assert replay[1]==tokens, 'forced baseline replay mismatch'
        assert next(r['step'] for r in baseline[2] if r['lead_refinement_active'])==event
    else:
        assert tokens==prefix and not any(r['lead_refinement_active'] for r in baseline[2])
    if smoke:
        reference = base.sweep.run_talr(model,processor,tokenizer,sample,math_ids,1024,**base.controller(threshold))
        assert reference[1]==tokens, 'default-off token regression'
        inputs,length = base.prepare(processor,sample,sample['image'],model.device)
        with torch.no_grad():
            untraced = base.generate_lead(model,tokenizer,**inputs,**base.sweep.talr_kwargs(math_ids,1024,**base.controller(threshold)))[0]
        assert untraced[length:].tolist()==tokens, 'uninstrumented regression'
        checks.update(default_off_token_regression=True,uninstrumented_token_regression=True)
        if event>=0:
            zero = base.continuation(model,processor,tokenizer,sample,math_ids,threshold,prefix,event,torch.zeros_like(actions['bounded_reweight']))
            assert zero[1]==tokens, 'zero-action regression'
            checks['zero_action_token_regression']=True
    results = {'talr':baseline}
    for branch in base.BRANCHES[1:]:
        if event<0:
            results[branch]=baseline
            continue
        value = base.continuation(model,processor,tokenizer,sample,math_ids,threshold,prefix,event,actions[branch])
        assert value[1][:len(prefix)]==prefix, 'shared prefix mismatch'
        applied = [r['step'] for r in value[2] if r['route_override_active']]
        assert applied==list(range(event,min(event+4,len(value[1])))), 'injection timing mismatch'
        results[branch]=value
    assert set(model._forward_pre_hooks)==hook_ids
    for branch,value in results.items():
        _,generated,trace,elapsed = value
        base.append(folder/'results.jsonl',dict(id=sample['id'],dataset=sample['dataset'],pair_id=sample['pair_id'],
            branch=branch,candidate_version=VERSION,event=event,generated_token_ids=generated,
            text=tokenizer.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            continuation_seconds=elapsed if branch=='talr' or event>=0 else 0.,
            no_event_reused=event<0 and branch!='talr',runtime_error=None,replay_mismatch=False,
            image_sha256=sample['image_sha256'],token_equal_talr=generated==tokens,
            effective_action_norm=0. if branch=='talr' or event<0 else actions[branch].norm().item()))
        base.append(folder/'traces.jsonl',dict(id=sample['id'],dataset=sample['dataset'],branch=branch,trace=trace))
    elapsed = time.perf_counter()-started
    base.append(folder/'sample_acceptance.jsonl',dict(id=sample['id'],dataset=sample['dataset'],event=event,
        checks=checks,baseline_forced_replay_equal=True,hook_removed=True,seconds=elapsed,
        static_generator_invocations=(16 if smoke else 9) if event>=0 else (4 if smoke else 2),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
    print('SAMPLE_COMPLETE %s event=%d %.1fs'%(key,event,elapsed),flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return event,elapsed


