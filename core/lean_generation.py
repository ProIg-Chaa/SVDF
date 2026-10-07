"""Lean execution of the frozen R1 fixed4 input-feedback protocol.

Only step0 and the actual capture step form p@E. Entropy is evaluated while
waiting in the unchanged step2..9 window. Afterwards there are no soft vectors,
LEAD switches, transition geometry, or controller-history updates.
The original filtered greedy choice is retained bit-for-bit; this is execution
cleanup, not a change to selection, precision, thresholds, or the visual action.
Weak replay and state isolation remain the frozen engine implementation.
"""
import torch
import torch.nn.functional as F
from lead.generation_utils import apply_sampling_filter

def generate_lean_fixed(model, tokenizer, **kwargs):
    expected = dict(
        do_sample=False, max_new_tokens=1024,
        lead_initial_transition_with_refinement=True,
        lead_force_initial_transition_step1=True,
        lead_disable_step0_linebreak_mix=True,
        lead_disable_to_normal_transition=True,
        lead_disable_answer_zone_lock=True,
        lead_refinement_window=8, lead_refinement_soft_cap=0,
        lead_refinement_start_offset=0, lead_refinement_threshold_mode="fixed",
        lead_refinement_entropy_threshold=kwargs["lead_refinement_entropy_threshold"],
        lead_refinement_soft_mix_lambda=0.95,
        max_switch_count=5, window_size=128,
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
    )
    for name, value in expected.items():
        if kwargs.get(name) != value:
            raise ValueError("Lean pilot accepts only frozen %s=%r" % (name, value))
    if kwargs.get("math_ids_tensor") is not None:
        raise ValueError("Math-token routing is not part of the frozen method")
    for name in (
        "lead_force_normal", "lead_initial_soft_only", "lead_initial_transition_only",
        "lead_soft_veto_on_diffuse", "lead_format_cooldown", "lead_soft_quota_ratio",
        "lead_early_visual_anchor", "trace_event_geometry", "trace_forced_answer_probe",
    ):
        if kwargs.get(name, False):
            raise ValueError("Unsupported active legacy control: "+name)
    if kwargs.get("forced_prefix_ids") is not None:
        raise ValueError("Forced-prefix weak replay stays on the frozen reference engine")
    if kwargs.get("trace_route_override_kind", "none") != "none":
        raise ValueError("Only the frozen online provider can supply a visual action")
    if kwargs.get("trace_soft_vector_collector") is not None:
        raise ValueError("Weak capture uses the unmodified reference generator")
    if kwargs.get("capture_logits_sink") is not None:
        raise ValueError("No additional logit probes in the lean pilot")
    input_ids = kwargs["input_ids"]
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    if input_ids.shape[0] != 1:
        raise ValueError("Lean fixed pilot requires batch size one")
    vision_inputs = {
        key: value for key, value in kwargs.items()
        if any(tag in key for tag in ("pixel", "image", "video")) and value is not None
    }
    token_trace = kwargs.get("token_trace")
    stream_callback = kwargs.get("stream_callback")
    provider = kwargs.get("trace_online_residual_provider")
    audit_sink = kwargs.get("lean_compute_audit_sink")
    threshold = kwargs["lead_refinement_entropy_threshold"]
    reference_ratio = kwargs.get("lead_refinement_reference_ratio", 1.0)
    if reference_ratio != 1.0:
        raise ValueError("Frozen reference-entropy ratio must be one")
    temperature, top_p, top_k, min_p = (
        kwargs["temperature"], kwargs["top_p"], kwargs["top_k"], kwargs["min_p"]
    )
    max_new_tokens = kwargs["max_new_tokens"]
    device = input_ids.device
    E = model.get_input_embeddings().weight
    past_key_values = None
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    all_generated = [input_ids[0].clone().tolist()]
    prompt_lens = [len(all_generated[0])]
    mode = torch.zeros(1, dtype=torch.long, device=device)
    cur_ref_entropy = None
    waiting = provider is not None
    event, vector = None, None
    soft_steps, entropy_steps = [], []
    window_close_reason = None

    for step in range(max_new_tokens):
        if past_key_values is None:
            model_inputs = {"input_ids": input_ids.clone(),
                            "attention_mask": attention_mask,
                            "cache_position": cache_position}
            model_inputs.update(vision_inputs)
        else:
            attention_mask_new = torch.ones(
                (1, 1), dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {"inputs_embeds": last_emb.unsqueeze(1),
                            "attention_mask": attention_mask,
                            "past_key_values": past_key_values,
                            "cache_position": cache_position}
        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True,
                            output_hidden_states=False, return_dict=True)
        past_key_values = outputs.past_key_values
        if vision_inputs:
            vision_inputs = {}
        cache_position = cache_position[-1:] + 1
        logits_original = outputs.logits[:, -1, :]
        if waiting and step > 9:
            waiting = False
            window_close_reason = "fixed_window_exhausted"
        need_raw = (step == 0 or (waiting and 1 <= step <= 9)
                    or (step == 1 and token_trace is not None))
        # Where needed, retain the reference raw-softmax-before-filter order.
        probs_original = F.softmax(logits_original, dim=-1) if need_raw else None
        # Preserve the frozen BF16 temperature/filter/softmax argmax, including ties.
        logits = logits_original / temperature
        logits_filtered = apply_sampling_filter(
            logits, top_k=top_k, top_p=top_p, min_p=min_p
        )
        probs = F.softmax(logits_filtered, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)

        normal_emb = E[next_tokens]
        mode_before = "soft" if step <= 1 else "normal"
        if step == 1:
            mode.fill_(1)
        cur_entropy = None
        entropy_proposed = False
        if (step == 0 and token_trace is not None) or (need_raw and step >= 1):
            cur_entropy = -(
                probs_original * probs_original.clamp(min=1e-8).log()
            ).sum(dim=-1)
            entropy_steps.append(step)
        if step == 0 and cur_entropy is not None:
            cur_ref_entropy = cur_entropy.clone()
        if step == 1:
            # Needed for provider decisions; trace also retains this reference.
            if cur_entropy is None and token_trace is not None:
                probs_original = F.softmax(logits_original, dim=-1)
                cur_entropy = -(
                    probs_original * probs_original.clamp(min=1e-8).log()
                ).sum(dim=-1)
                entropy_steps.append(step)
            if cur_entropy is not None:
                cur_ref_entropy = cur_entropy.clone()
        if waiting and 2 <= step <= 9:
            entropy_proposed = bool(
                ((cur_entropy > reference_ratio*cur_ref_entropy)
                 & (cur_entropy > threshold))[0].item()
            )
        need_soft = step == 0 or (waiting and entropy_proposed)
        raw_soft_emb = torch.matmul(probs_original, E) if need_soft else None
        if need_soft:
            soft_steps.append(step)
        last_emb = raw_soft_emb if step == 0 else normal_emb
        route_override_active = False

        if waiting and entropy_proposed:
            # Provider reads live prefix/cache locals, exactly as in the reference.
            request = provider(
                step=int(step), selected_token_id=int(next_tokens[0].item()),
                raw_soft=raw_soft_emb[0], hard=normal_emb[0],
                transition_step=1, entropy_proposed=True,
                locked_normal=False, controller_mode=1,
            )
            if request is None or int(request["event"]) != step:
                raise RuntimeError("Frozen provider rejected an eligible capture")
            event, vector = int(request["event"]), request["vector"]
            waiting = False
            window_close_reason = "captured"
        if event is not None and event <= step < event+4:
            external_emb = torch.as_tensor(vector, device=device, dtype=E.dtype)
            if external_emb.ndim == 1:
                external_emb = external_emb.unsqueeze(0)
            if external_emb.shape != last_emb.shape:
                raise ValueError("Frozen visual action shape mismatch")
            route_origin = last_emb
            # Retain the reference operation order, including its multiply by one.
            last_emb = route_origin + 1.0*external_emb
            route_override_active = True

        token_id = int(next_tokens[0].item())
        all_generated[0].append(token_id)
        if token_trace is not None:
            record = {
                "step": int(step), "token_id": token_id,
                "raw_entropy": float(cur_entropy[0].item()) if cur_entropy is not None else None,
                "raw_top1_prob": float(probs_original[0].max().item()) if probs_original is not None else None,
                "cur_ref_entropy": float(cur_ref_entropy[0].item()) if cur_ref_entropy is not None else None,
                "mode": "soft" if step == 0 else "normal",
                "mode_before": mode_before,
                "to_soft": False, "to_normal": step == 1,
                "locked_normal": False, "route_suppressed_by": [],
                "lead_refinement_active": False,
                "lead_refinement_candidate": False,
                "lead_refinement_entropy_proposed": entropy_proposed,
                "lead_refinement_count_before": 0,
                "lead_initial_transition_step": None if step == 0 else 1,
                "lead_refinement_elapsed": None if step == 0 else step-1,
                "lead_refinement_start_offset": 0,
                "lead_refinement_window": 8, "lead_refinement_soft_cap": 0,
                "lead_refinement_soft_mix_lambda": 0.95,
                "lead_answer_zone_lock_enabled": False,
                "lead_refinement_reference_ratio": 1.0,
                "lead_refinement_entropy_threshold": threshold,
                "route_override_active": route_override_active,
                "lean_diagnostic_not_computed": not need_raw,
            }
            token_trace.append(record)
        if stream_callback is not None:
            stream_callback(token_id)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        # Keep reference identity batch-selection semantics in this first pilot.
        keep_idx = torch.tensor([0], dtype=torch.long, device=device)
        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        mode = mode[keep_idx]
        if cur_ref_entropy is not None:
            cur_ref_entropy = cur_ref_entropy[keep_idx]
        if hasattr(past_key_values, "batch_select_indices"):
            past_key_values.batch_select_indices(keep_idx)

    if audit_sink is not None:
        audit_sink.append({
            "soft_matmul_steps": soft_steps, "soft_matmul_count": len(soft_steps),
            "entropy_steps": entropy_steps, "entropy_count": len(entropy_steps),
            "capture_event": event, "window_close_reason": window_close_reason,
            "generated_tokens": len(all_generated[0])-prompt_lens[0],
            "ordinary_phase_soft_matmul_count": 0,
            "legacy_switch_geometry_executed": False,
            "filtered_selection_preserved": True,
        })
    out = torch.full(
        (1, len(all_generated[0])), tokenizer.pad_token_id or 0,
        dtype=torch.long, device=device
    )
    out[0] = torch.tensor(all_generated[0], dtype=torch.long, device=device)
    return out
