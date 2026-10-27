import contextlib,hashlib,json,math,time
class ProbeDone(Exception):
    pass

def read_candidate(model, engine, sample, prefix, candidate, event, parent_hashes, action=None, multiplier=None):
    import torch
    import torch.nn.functional as F
    from route_tape import tensor_hash
    from attention_boost import VisualAttentionBoost

    ids = candidate["token_ids"]
    first = len(prefix)
    final = first + len(ids) - 1
    forced = prefix + ids[:-1]
    assert final < 1024 and model.config.image_token_id not in forced
    assert first >= event + 4
    assert model.config.eos_token_id not in forced
    weight = model.get_input_embeddings().weight
    records, raw_logits, input_hashes, geometry = [], [], [], []
    step = -1
    initial_sdpa = F.scaled_dot_product_attention
    modules = [model] + [layer.self_attn for layer in model.model.layers]
    hooks_before = [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in modules]
    applied = None if action is None else action.to(device=weight.device, dtype=weight.dtype).reshape(1, 1, -1)

    def before(module, positional, keyword):
        nonlocal step
        step += 1
        value = keyword.get("inputs_embeds")
        if value is None:
            value = keyword.get("input_ids")
        assert value is not None
        input_hashes.append(tensor_hash(value))
        if step <= event:
            assert input_hashes[-1] == parent_hashes[step], ("pre-event actual input changed", step)
        if step >= 2:
            previous_id = forced[step - 1]
            hard = weight[previous_id].detach().reshape(1, 1, -1)
            active = applied is not None and event + 1 <= step <= event + 4
            expected = hard + applied if active else hard
            assert torch.equal(value, expected), ("forced visible token did not determine actual input", step, previous_id)
            if active:
                actual = value.float() - hard.float()
                geometry.append({"forward_step": step, "feedback_slot": step - 1,
                                 "token_id": previous_id,
                                 "hard_input_sha256": tensor_hash(hard),
                                 "actual_input_sha256": input_hashes[-1],
                                 "cast_action_norm": float(applied.float().norm()),
                                 "actual_delta_norm": float(actual.norm())})

    def after(module, positional, keyword, output):
        if step < first:
            return
        assert first <= step <= final
        raw = output.logits[0, -1, :].detach().contiguous().cpu()
        assert raw.ndim == 1 and raw.numel() == model.config.vocab_size
        assert torch.isfinite(raw).all()
        index = step - first
        token_id = ids[index]
        lp = torch.log_softmax(raw.float(), dim=-1)
        records.append({"forward_step": step, "predicted_token_index": step,
                        "target_token_id": token_id, "candidate_offset": index,
                        "raw_logits_dtype": str(raw.dtype),
                        "raw_logit": float(raw[token_id]),
                        "log_probability": float(lp[token_id]),
                        "probability": float(lp[token_id].exp()),
                        "rank": int((raw > raw[token_id]).sum()) + 1,
                        "top1_token_id": int(raw.argmax()),
                        "top1_probability": float(lp.max().exp())})
        raw_logits.append(raw)
        if step == final:
            raise ProbeDone()

    guard = None if multiplier is None else VisualAttentionBoost(model, event, multiplier)
    interrupted = False
    handles = [model.register_forward_pre_hook(before, with_kwargs=True),
               model.register_forward_hook(after, with_kwargs=True)]
    started = time.monotonic()
    try:
        with guard if guard is not None else contextlib.nullcontext():
            try:
                engine["generate"](sample, cap=0, lam=.95, prefix=forced,
                                   event=event, action=action)
            except ProbeDone:
                interrupted = True
    finally:
        for handle in reversed(handles):
            handle.remove()
    assert interrupted, "diagnostic did not reach its fixed endpoint"
    assert step == final and len(input_hashes) == final + 1
    assert [x["forward_step"] for x in records] == list(range(first, final + 1))
    assert F.scaled_dot_product_attention is initial_sdpa
    assert [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in modules] == hooks_before
    if action is not None:
        assert [x["forward_step"] for x in geometry] == list(range(event + 1, event + 5))
    observation = None if guard is None else guard.validate(final + 1)
    joint = sum(x["log_probability"] for x in records)
    word = sum(x["log_probability"] for x in records[:candidate["word_token_count"]])
    return {"text": candidate["text"], "word_text": candidate["word_text"],
            "role": candidate["role"], "token_ids": ids,
            "word_token_count": candidate["word_token_count"],
            "first_token_probability": records[0]["probability"],
            "word_probability": math.exp(word), "word_log_probability": word,
            "phrase_probability": math.exp(joint), "phrase_log_probability": joint,
            "per_token": records, "actual_input_hashes": input_hashes,
            "residual_geometry": geometry, "attention_observation": observation,
            "fixed_prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            "forced_candidate_history_sha256": hashlib.sha256(json.dumps(forced).encode()).hexdigest(),
            "hook_cleanup_exact": True, "seconds": time.monotonic() - started}, raw_logits
