import os
import re
import copy
import torch
import torch.nn.functional as F
import random
import numpy as np
import math

from .refinement_controller import nearest_rank_quantile

a  = 1.0   # 锚点加权缩放：1.0=原样；0.0=禁用锚点加权
b1 = 1.0   # 触发 normal→soft 所需的“额外熵阈值” (越大越难切到 soft)
b2 = 0.2   # 触发 soft→normal 所需的“额外熵阈值” (越大越难切回 normal)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers
        transformers.set_seed(seed)
    except Exception:
        pass


def apply_sampling_filter(logits, top_k=0, top_p=1.0, min_p=0.0):
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float('-inf'), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, float('-inf'), logits)
    return logits


def _topk_trace_record(tokenizer, probs, batch_index, top_k):
    if top_k is None or int(top_k) <= 0:
        return None
    cur_top_k = min(int(top_k), int(probs.shape[-1]))
    vals, ids = torch.topk(probs[batch_index], k=cur_top_k, dim=-1)
    vals = vals.detach().float().cpu().tolist()
    ids = ids.detach().long().cpu().tolist()
    texts = [
        tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in ids
    ]
    return [
        {
            "token_id": int(token_id),
            "prob": float(prob),
            "token_text": text,
        }
        for token_id, prob, text in zip(ids, vals, texts)
    ]


def _add_topk_trace_fields(record, tokenizer, raw_probs, filtered_probs, batch_index, trace_topk):
    if trace_topk is None or int(trace_topk) <= 0:
        return
    raw_topk = _topk_trace_record(tokenizer, raw_probs, batch_index, trace_topk)
    filtered_topk = _topk_trace_record(tokenizer, filtered_probs, batch_index, trace_topk)
    if raw_topk:
        record["raw_topk"] = raw_topk
        record["raw_top1_prob"] = raw_topk[0]["prob"]
        record["raw_margin"] = (
            raw_topk[0]["prob"] - raw_topk[1]["prob"]
            if len(raw_topk) > 1
            else raw_topk[0]["prob"]
        )
    if filtered_topk:
        record["filtered_topk"] = filtered_topk
        record["filtered_top1_prob"] = filtered_topk[0]["prob"]
        record["filtered_margin"] = (
            filtered_topk[0]["prob"] - filtered_topk[1]["prob"]
            if len(filtered_topk) > 1
            else filtered_topk[0]["prob"]
        )


def _embedding_geometry_record(hard_emb, soft_emb, route_emb, batch_index, visual_anchor=None):
    """Return compact scalar geometry without retaining hidden tensors in the trace."""
    hard = hard_emb[batch_index].detach().float()
    soft = soft_emb[batch_index].detach().float()
    route = route_emb[batch_index].detach().float()

    def cosine(left, right):
        return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item())

    record = {
        "hard_emb_norm": float(hard.norm().item()),
        "soft_emb_norm": float(soft.norm().item()),
        "route_emb_norm": float(route.norm().item()),
        "soft_hard_cosine": cosine(soft, hard),
        "route_hard_cosine": cosine(route, hard),
        "route_soft_cosine": cosine(route, soft),
    }
    if visual_anchor is not None:
        anchor = visual_anchor.detach().float()
        record["route_visual_anchor_cosine"] = cosine(route, anchor)
        record["soft_visual_anchor_cosine"] = cosine(soft, anchor)
    else:
        record["route_visual_anchor_cosine"] = None
        record["soft_visual_anchor_cosine"] = None
    return record


def _forced_answer_probe_from_route(
    model,
    tokenizer,
    route_emb,
    attention_mask,
    past_key_values,
    cache_position,
    gold_choice,
    choice_case="upper",
    prompt_hidden_states=None,
    visual_token_mask=None,
    prompt_len=None,
    visual_attn_last_k=4,
):
    """Probe A-E after a forced answer marker on an isolated cache copy."""
    result = {
        "available": False,
        "interpretation": "forced-answer diagnostic; not natural-generation confidence",
        "gold_choice": gold_choice,
    }
    if route_emb.shape[0] != 1 or past_key_values is None:
        result["reason"] = "probe_requires_batch1_and_cache"
        return result
    try:
        marker = "\n</think>\nAnswer: ("
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        choice_ids = {}
        choice_symbols = "abcde" if choice_case == "lower" else "ABCDE"
        for normalized_choice, choice_symbol in zip("ABCDE", choice_symbols):
            encoded = tokenizer.encode(choice_symbol, add_special_tokens=False)
            if len(encoded) != 1:
                result["reason"] = f"choice_{choice_symbol}_is_not_single_token"
                return result
            choice_ids[normalized_choice] = int(encoded[0])

        probe_cache = copy.deepcopy(past_key_values)
        probe_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device),
        ], dim=1)
        probe_position = cache_position.clone()
        with torch.no_grad():
            outputs = model(
                inputs_embeds=route_emb.unsqueeze(1),
                attention_mask=probe_mask,
                past_key_values=probe_cache,
                cache_position=probe_position,
                use_cache=True,
                output_attentions=visual_token_mask is not None,
                output_hidden_states=prompt_hidden_states is not None,
                return_dict=True,
            )
        try:
            attn_summary = None
            if visual_token_mask is not None and getattr(outputs, "attentions", None):
                attn_summary = _summarize_visual_attention(
                    attn_layers=outputs.attentions,
                    visual_token_mask=visual_token_mask,
                    prompt_len=int(prompt_len or visual_token_mask.shape[1]),
                    attn_last_k=int(visual_attn_last_k),
                )
            if attn_summary is not None:
                available = bool(attn_summary["available"][0].item())
                result.update({
                    "event_visual_attn_available": available,
                    "event_visual_attn_last_k": int(visual_attn_last_k),
                    "event_visual_attn_mass": float(attn_summary["mass"][0].item()),
                    "event_visual_attn_top1": float(attn_summary["top1"][0].item()),
                    "event_visual_attn_top4_sum": float(attn_summary["top4_sum"][0].item()),
                    "event_visual_attn_entropy": (
                        float(attn_summary["entropy"][0].item()) if available else None
                    ),
                })
            else:
                result["event_visual_attn_available"] = False
                result["event_visual_attn_reason"] = "model_attention_not_exposed"

            if prompt_hidden_states is not None and getattr(outputs, "hidden_states", None):
                alignment = _summarize_hidden_visual_alignment(
                    current_hidden=outputs.hidden_states[-1][:, -1, :],
                    prompt_hidden_states=prompt_hidden_states,
                    visual_token_mask=visual_token_mask,
                    top_k=4,
                )
                align_available = bool(alignment["available"][0].item())
                result.update({
                    "event_hidden_visual_align_available": align_available,
                    "event_hidden_visual_align_max": (
                        float(alignment["max"][0].item()) if align_available else None
                    ),
                    "event_hidden_visual_align_top4_mean": (
                        float(alignment["topk_mean"][0].item()) if align_available else None
                    ),
                    "event_hidden_visual_align_token_count": int(alignment["token_count"][0].item()),
                })
            else:
                result["event_hidden_visual_align_available"] = False
                result["event_hidden_visual_align_reason"] = "prompt_or_current_hidden_state_missing"
        except Exception as diagnostic_exc:
            result.update({
                "event_visual_attn_available": False,
                "event_hidden_visual_align_available": False,
                "event_visual_diagnostic_error_type": type(diagnostic_exc).__name__,
                "event_visual_diagnostic_error_message": str(diagnostic_exc),
            })
        probe_cache = outputs.past_key_values
        probe_position = probe_position[-1:] + 1
        for marker_id in marker_ids:
            probe_mask = torch.cat([
                probe_mask,
                torch.ones((1, 1), dtype=probe_mask.dtype, device=probe_mask.device),
            ], dim=1)
            token = torch.tensor([[marker_id]], dtype=torch.long, device=route_emb.device)
            with torch.no_grad():
                outputs = model(
                    input_ids=token,
                    attention_mask=probe_mask,
                    past_key_values=probe_cache,
                    cache_position=probe_position,
                    use_cache=True,
                )
            probe_cache = outputs.past_key_values
            probe_position = probe_position[-1:] + 1

        probs = F.softmax(outputs.logits[:, -1, :].float(), dim=-1)[0]
        choice_probs = {choice: float(probs[token_id].item()) for choice, token_id in choice_ids.items()}
        gold = str(gold_choice or "").upper()[:1]
        if gold not in choice_probs:
            result["reason"] = "gold_choice_not_in_A_to_E"
            result["choice_probs"] = choice_probs
            return result
        best_other = max(prob for choice, prob in choice_probs.items() if choice != gold)
        result.update({
            "available": True,
            "marker": marker,
            "choice_case": choice_case,
            "choice_token_ids": choice_ids,
            "choice_probs": choice_probs,
            "gold_margin": choice_probs[gold] - best_other,
            "predicted_choice": max(choice_probs, key=choice_probs.get),
        })
        return result
    except Exception as exc:
        result["reason"] = "probe_exception"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        return result


def _entropy_spike_mask(raw_entropy, entropy_history, window, alpha, min_history, min_entropy):
    batch_size = raw_entropy.shape[0]
    mask = torch.zeros(batch_size, dtype=torch.bool, device=raw_entropy.device)
    deltas = torch.zeros(batch_size, dtype=raw_entropy.dtype, device=raw_entropy.device)
    history_counts = torch.zeros(batch_size, dtype=torch.long, device=raw_entropy.device)
    window = max(1, int(window))
    for bi in range(batch_size):
        hist = entropy_history[bi][-window:]
        history_counts[bi] = len(hist)
        if len(hist) < int(min_history):
            continue
        hist_tensor = raw_entropy.new_tensor(hist)
        mean = hist_tensor.mean()
        std = hist_tensor.std(unbiased=False)
        threshold = mean + float(alpha) * std
        deltas[bi] = raw_entropy[bi] - mean
        mask[bi] = (
            raw_entropy[bi] >= float(min_entropy)
            and raw_entropy[bi] > threshold
        )
    return mask, deltas, history_counts


def _talr_refinement_eligible(
    step,
    transition_step,
    refinement_count,
    window,
    soft_cap,
    entropy_proposed,
    locked_normal=False,
    start_offset=0,
):
    """Return whether one post-transition soft refinement may run."""
    if transition_step < 0 or locked_normal or not entropy_proposed:
        return False
    elapsed = int(step) - int(transition_step)
    return (
        int(start_offset) < elapsed <= int(start_offset) + int(window)
        and int(refinement_count) < int(soft_cap)
    )


_FORMAT_TOKEN_TEXTS = {
    "\n", "\n\n", ".", ",", ":", ";", "-", "(", ")", "[", "]", "{", "}",
    "<", ">", "</", "<think", "think", "answer", "option", "**", "*",
}

_HIGH_RISK_FORMAT_TOKEN_TEXTS = {
    ":", "(", ")", "[", "]", "{", "}", "<", ">", "</", "<think", "think",
    "answer", "option", "**",
}


def _is_format_token_text(text):
    raw = text or ""
    stripped = raw.strip()
    lowered = stripped.lower().replace("▁", " ")
    if not stripped:
        return True
    if raw in _FORMAT_TOKEN_TEXTS or stripped in _FORMAT_TOKEN_TEXTS or lowered in _FORMAT_TOKEN_TEXTS:
        return True
    if all((not ch.isalnum()) for ch in stripped):
        return True
    return False


def _is_high_risk_format_token_text(text):
    raw = text or ""
    stripped = raw.strip()
    lowered = stripped.lower().replace("▁", " ")
    if not stripped:
        return False
    if raw in _HIGH_RISK_FORMAT_TOKEN_TEXTS or stripped in _HIGH_RISK_FORMAT_TOKEN_TEXTS or lowered in _HIGH_RISK_FORMAT_TOKEN_TEXTS:
        return True
    if "answer" in lowered or "think" in lowered or "option" in lowered:
        return True
    return False


def get_math_symbols_ids(tokenizer):
    math_symbols = [
        "+", "-", "*", "/", "^", "=", "<", ">", "\\leq", "\\geq", "\\neq", "\\approx", "\\sim", "\\equiv", "\\to", "\\implies", "\\iff",
        "(", ")", "[", "]", "{", "}", "\\left(", "\\right)", "\\left[", "\\right]", "\\left\\{", "\\right\\}",
        "\\begin{pmatrix}", "\\end{pmatrix}",
        "\\frac", "\\dfrac", "\\sqrt", "\\sqrt[]",
        "\\in", "\\notin", "\\subset", "\\supset", "\\subseteq", "\\supseteq", "\\cup", "\\cap", "\\emptyset", "\\varnothing",
        "\\pi", "\\theta", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\lambda", "\\mu", "\\nu",
        "\\sin", "\\cos", "\\tan", "\\arcsin", "\\arccos", "\\arctan", "\\log", "\\ln", "\\exp",
        "_", "\\binom", "\\choose", "\\cdot", "\\dots", "\\ldots", "\\cdots", "\\vdots", "\\ddots",
        "\\mathbb", "\\mathbf", "\\mathrm", "\\text", "\\mbox",
        "\\infty", "\\circ", "\\prime", "\\ast", "\\star", "\\triangle", "\\triangleleft", "\\triangleright", "\\perp", "\\parallel", "\\angle",
        "\\boxed", "\\overline", "\\underline", "\\lceil", "\\rceil", "\\lfloor", "\\rfloor", "\\left", "\\right", "\\mid", "|", "\\vert", "\\Vert",
        "\\because", "\\therefore", "\\forall", "\\exists", "\\wedge", "\\vee", "\\neg",
        "\\sum", "\\prod", "\\int", "\\lim", "\\min", "\\max", "\\arg", "\\deg", "\\gcd", "\\operatorname",
        "\\cot", 
        "\\cotg", "\\sec", "\\csc",
    ]
    math_symbols += [chr(c) for c in range(ord('0'), ord('9')+1)]
    math_symbols += [chr(c) for c in range(ord('a'), ord('z')+1)]
    math_symbols += [chr(c) for c in range(ord('A'), ord('Z')+1)]
    math_token_ids = set()
    for symbol in math_symbols:
        math_token_ids.update(tokenizer.encode(symbol, add_special_tokens=False))
    return math_token_ids
    

def generate_cot(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)

    stream_callback = kwargs.pop("stream_callback", None)
    token_trace = kwargs.pop("token_trace", None)
    trace_topk = kwargs.pop("trace_topk", 0)
    trace_route_override_step = int(kwargs.pop("trace_route_override_step", -1))
    trace_route_override_kind = str(kwargs.pop("trace_route_override_kind", "none"))
    trace_route_override_mix_lambda = float(
        kwargs.pop("trace_route_override_mix_lambda", 0.95)
    )
    trace_route_override_lambda_max = (
        2.0
        if trace_route_override_kind == "method_plus_external_residual"
        else 1.0
    )
    if not 0.0 <= trace_route_override_mix_lambda <= trace_route_override_lambda_max:
        raise ValueError(
            "trace_route_override_mix_lambda must be in "
            f"[0, {trace_route_override_lambda_max:g}] for "
            f"kind={trace_route_override_kind}"
        )
    trace_route_override_duration = int(
        kwargs.pop("trace_route_override_duration", 1)
    )
    if trace_route_override_duration < 1:
        raise ValueError("trace_route_override_duration must be >= 1")
    trace_external_route_vector = kwargs.pop("trace_external_route_vector", None)
    trace_external_route_source = kwargs.pop("trace_external_route_source", None)
    trace_soft_vector_collector = kwargs.pop("trace_soft_vector_collector", None)
    trace_capture_soft_vector_step = int(
        kwargs.pop("trace_capture_soft_vector_step", -1)
    )
    forced_prefix_ids = kwargs.pop("forced_prefix_ids", None)
    if forced_prefix_ids is not None:
        forced_prefix_ids = [int(x) for x in forced_prefix_ids]
    log_visual_attn_summary = kwargs.pop("log_visual_attn_summary", False)
    visual_attn_summary_last_k = kwargs.pop("visual_attn_summary_last_k", 4)
    sidecar_attn_on_entropy = kwargs.pop("sidecar_attn_on_entropy", False)
    sidecar_attn_entropy_threshold = kwargs.pop("sidecar_attn_entropy_threshold", 2.0)
    sidecar_attn_last_k = kwargs.pop("sidecar_attn_last_k", 4)

    # ============================================

    batch_size = input_ids.shape[0]
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    visual_token_mask = (
        _build_visual_token_mask(input_ids, tokenizer)
        if log_visual_attn_summary or sidecar_attn_on_entropy
        else None
    )
    sidecar_vision_inputs = dict(vision_inputs)

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    generated = input_ids.clone()
    attn_mask = attention_mask.clone() if attention_mask is not None else None
    past_key_values = None
    next_inputs_embeds = None
    cache_position = torch.arange(generated.shape[1], device=device, dtype=torch.long)
    attn_config_values = None
    if log_visual_attn_summary:
        attn_config_values = _get_text_attn_implementation(model)
        _set_text_attn_implementation(attn_config_values, "eager")

    try:
        for step in range(max_new_tokens):
            cur_batch = generated.shape[0]
            if cur_batch == 0:
                break

            if past_key_values is None:
                model_inputs = {"input_ids": generated}
                if attn_mask is not None:
                    model_inputs["attention_mask"] = attn_mask
                if vision_inputs:
                    model_inputs.update(vision_inputs)
                model_inputs["cache_position"] = cache_position
            else:
                if attn_mask is not None:
                    attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
                    attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
                model_inputs = {"past_key_values": past_key_values}
                if next_inputs_embeds is None:
                    model_inputs["input_ids"] = next_tokens.unsqueeze(1)
                else:
                    model_inputs["inputs_embeds"] = next_inputs_embeds.unsqueeze(1)
                if attn_mask is not None:
                    model_inputs["attention_mask"] = attn_mask
                model_inputs["cache_position"] = cache_position

            need_visual_attn_summary = log_visual_attn_summary and (past_key_values is not None)

            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    use_cache=True,
                    output_attentions=need_visual_attn_summary,
                )
            past_key_values = outputs.past_key_values
            if vision_inputs:
                vision_inputs = {}
            cache_position = cache_position[-1:] + 1

            visual_attn_summary = None
            if need_visual_attn_summary:
                visual_attn_summary = _summarize_visual_attention(
                    attn_layers=outputs.attentions,
                    visual_token_mask=visual_token_mask,
                    prompt_len=prompt_len,
                    attn_last_k=visual_attn_summary_last_k,
                )

            next_token_logits = outputs.logits[:, -1, :]  # [cur_batch, vocab]
            raw_probs = F.softmax(next_token_logits, dim=-1)
            raw_entropy = -(
                raw_probs * raw_probs.clamp(min=1e-8).log()
            ).sum(dim=-1)
            logits = next_token_logits / temperature
            logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)

            probs = F.softmax(logits, dim=-1)
            filtered_entropy = -(
                probs * probs.clamp(min=1e-8).log()
            ).sum(dim=-1)
            if forced_prefix_ids is not None and step < len(forced_prefix_ids):
                next_tokens = torch.full(
                    (cur_batch,),
                    int(forced_prefix_ids[step]),
                    dtype=torch.long,
                    device=device,
                )
            elif do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            embedding_weight = model.get_input_embeddings().weight
            normal_emb = embedding_weight[next_tokens]
            soft_emb = torch.matmul(
                raw_probs.to(embedding_weight.dtype), embedding_weight
            )
            if (
                trace_soft_vector_collector is not None
                and step == trace_capture_soft_vector_step
            ):
                trace_soft_vector_collector.append(
                    {
                        "step": int(step),
                        "soft_embedding": soft_emb[0].detach().float().cpu(),
                        "hard_embedding": normal_emb[0].detach().float().cpu(),
                        "raw_entropy": float(raw_entropy[0].item()),
                        "selected_token_id": int(next_tokens[0].item()),
                        "selected_token_prob": float(
                            raw_probs[0, next_tokens[0]].item()
                        ),
                    }
                )
            raw_top_values = torch.topk(
                raw_probs, k=min(5, raw_probs.shape[-1]), dim=-1
            ).values
            raw_top1_prob = raw_top_values[:, 0]
            raw_margin = (
                raw_top_values[:, 0] - raw_top_values[:, 1]
                if raw_top_values.shape[-1] > 1
                else raw_top_values[:, 0]
            )
            route_override_active = (
                trace_route_override_step <= step
                < trace_route_override_step + trace_route_override_duration
                and trace_route_override_kind != "none"
            )
            next_inputs_embeds = None
            external_norm = None
            external_hard_cosine = None
            if route_override_active:
                if trace_route_override_kind == "hard":
                    next_inputs_embeds = normal_emb
                elif trace_route_override_kind in {"raw_soft", "method_soft"}:
                    next_inputs_embeds = soft_emb
                elif trace_route_override_kind == "contracted_soft":
                    next_inputs_embeds = (
                        trace_route_override_mix_lambda * soft_emb
                        + (1.0 - trace_route_override_mix_lambda) * normal_emb
                    )
                elif trace_route_override_kind in {
                    "external_soft",
                    "external_contracted",
                    "external_residual",
                }:
                    if trace_external_route_vector is None:
                        raise ValueError(
                            f"{trace_route_override_kind} requires "
                            "trace_external_route_vector"
                        )
                    external_emb = torch.as_tensor(
                        trace_external_route_vector,
                        device=device,
                        dtype=embedding_weight.dtype,
                    )
                    if external_emb.ndim == 1:
                        external_emb = external_emb.unsqueeze(0)
                    if external_emb.shape != normal_emb.shape:
                        if external_emb.shape[0] == 1 and cur_batch > 1:
                            external_emb = external_emb.expand(cur_batch, -1)
                        else:
                            raise ValueError(
                                "External route vector shape mismatch: "
                                f"{tuple(external_emb.shape)} vs "
                                f"{tuple(normal_emb.shape)}"
                            )
                    if trace_route_override_kind == "external_soft":
                        next_inputs_embeds = external_emb
                    elif trace_route_override_kind == "external_residual":
                        # The supplied tensor is a direction, not an old-token
                        # embedding. Re-anchor it at the current hard token so
                        # multi-step interventions do not pull toward the token
                        # produced at the original checkpoint.
                        next_inputs_embeds = (
                            normal_emb
                            + trace_route_override_mix_lambda * external_emb
                        )
                    else:
                        next_inputs_embeds = (
                            trace_route_override_mix_lambda * external_emb
                            + (1.0 - trace_route_override_mix_lambda) * normal_emb
                        )
                    external_norm = torch.linalg.vector_norm(
                        external_emb.float(), dim=-1
                    )
                    external_hard_cosine = F.cosine_similarity(
                        external_emb.float(), normal_emb.float(), dim=-1
                    )
                else:
                    raise ValueError(
                        f"Unsupported trace_route_override_kind={trace_route_override_kind}"
                    )

            hard_norm = torch.linalg.vector_norm(normal_emb.float(), dim=-1)
            soft_norm = torch.linalg.vector_norm(soft_emb.float(), dim=-1)
            soft_hard_delta = torch.linalg.vector_norm(
                (soft_emb - normal_emb).float(), dim=-1
            )
            soft_hard_cosine = F.cosine_similarity(
                soft_emb.float(), normal_emb.float(), dim=-1
            )

            for bi, orig in enumerate(unfinished_idx):
                token_id = next_tokens[bi].item()
                token_text = tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                all_generated[orig].append(token_id)
                if token_trace is not None:
                    sidecar_record = None
                    if (
                        sidecar_attn_on_entropy
                        and raw_entropy[bi].item() >= float(sidecar_attn_entropy_threshold)
                        and visual_token_mask is not None
                    ):
                        try:
                            sidecar_record = _observe_sidecar_visual_attention(
                                model=model,
                                tokenizer=tokenizer,
                                generated_ids=all_generated[orig],
                                prompt_len=prompt_len,
                                visual_token_mask=visual_token_mask[bi : bi + 1],
                                vision_inputs=sidecar_vision_inputs,
                                device=device,
                                attn_last_k=sidecar_attn_last_k,
                            )
                        except Exception as exc:
                            sidecar_record = {
                                "sidecar_attn_observed": False,
                                "sidecar_attn_error_type": type(exc).__name__,
                                "sidecar_attn_error_message": str(exc),
                            }
                            torch.cuda.empty_cache()
                    record = {
                        "step": int(step),
                        "batch_index": int(orig),
                        "token_id": int(token_id),
                        "token_text": token_text,
                        "token_is_newline": "\n" in token_text,
                        "token_is_whitespace": not token_text.strip(),
                        "token_is_punctuation": bool(
                            token_text.strip()
                            and all(not char.isalnum() for char in token_text.strip())
                        ),
                        "token_is_answer_marker": (
                            "answer" in token_text.lower()
                            or "</think" in token_text.lower()
                        ),
                        "raw_entropy": float(raw_entropy[bi].item()),
                        "filtered_entropy": float(filtered_entropy[bi].item()),
                        "selected_prob": float(probs[bi, next_tokens[bi]].item()),
                        "mode": "normal",
                        "raw_top1_prob": float(raw_top1_prob[bi].item()),
                        "raw_margin": float(raw_margin[bi].item()),
                        "raw_top2_mass": float(
                            raw_top_values[bi, : min(2, raw_top_values.shape[-1])]
                            .sum()
                            .item()
                        ),
                        "raw_top5_mass": float(raw_top_values[bi].sum().item()),
                        "hard_embedding_norm": float(hard_norm[bi].item()),
                        "soft_embedding_norm": float(soft_norm[bi].item()),
                        "soft_hard_l2": float(soft_hard_delta[bi].item()),
                        "soft_hard_relative_l2": float(
                            (
                                soft_hard_delta[bi]
                                / hard_norm[bi].clamp_min(1e-8)
                            ).item()
                        ),
                        "soft_hard_cosine": float(soft_hard_cosine[bi].item()),
                        "route_override_active": bool(route_override_active),
                        "route_override_kind": (
                            trace_route_override_kind if route_override_active else None
                        ),
                        "route_override_mix_lambda": (
                            float(trace_route_override_mix_lambda)
                            if route_override_active
                            else None
                        ),
                        "route_override_duration": (
                            int(trace_route_override_duration)
                            if route_override_active
                            else None
                        ),
                        "route_override_offset": (
                            int(step - trace_route_override_step)
                            if route_override_active
                            else None
                        ),
                        "external_route_source": (
                            trace_external_route_source
                            if route_override_active
                            and trace_route_override_kind.startswith("external_")
                            else None
                        ),
                        "external_embedding_norm": (
                            float(external_norm[bi].item())
                            if external_norm is not None
                            else None
                        ),
                        "external_hard_cosine": (
                            float(external_hard_cosine[bi].item())
                            if external_hard_cosine is not None
                            else None
                        ),
                    }
                    _add_topk_trace_fields(
                        record,
                        tokenizer,
                        raw_probs,
                        probs,
                        bi,
                        trace_topk,
                    )
                    if sidecar_attn_on_entropy:
                        if sidecar_record is None:
                            record.update({
                                "sidecar_attn_observed": False,
                                "sidecar_attn_skipped": True,
                            })
                        else:
                            record.update(sidecar_record)
                    if log_visual_attn_summary and visual_attn_summary is None:
                        record.update({
                            "visual_attn_available": False,
                            "visual_attn_mass": 0.0,
                            "visual_attn_top1": 0.0,
                            "visual_attn_top4_sum": 0.0,
                            "visual_attn_entropy": None,
                            "visual_attn_token_count": 0,
                        })
                    elif visual_attn_summary is not None:
                        record.update({
                            "visual_attn_available": bool(visual_attn_summary["available"][bi].item()),
                            "visual_attn_mass": float(visual_attn_summary["mass"][bi].item()),
                            "visual_attn_top1": float(visual_attn_summary["top1"][bi].item()),
                            "visual_attn_top4_sum": float(visual_attn_summary["top4_sum"][bi].item()),
                            "visual_attn_entropy": (
                                float(visual_attn_summary["entropy"][bi].item())
                                if visual_attn_summary["available"][bi].item()
                                else None
                            ),
                            "visual_attn_token_count": int(visual_attn_summary["token_count"][bi].item()),
                        })
                    token_trace.append(record)
                if stream_callback is not None:
                    stream_callback(all_generated[orig][-1])

            if tokenizer.eos_token_id is not None:
                cur_finished = (next_tokens == tokenizer.eos_token_id)
            else:
                cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)
            keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
            unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

            if len(unfinished_idx) == 0:
                break
            generated = generated[keep_idx]
            next_tokens = next_tokens[keep_idx]
            if next_inputs_embeds is not None:
                next_inputs_embeds = next_inputs_embeds[keep_idx]
            if attention_mask is not None:
                attention_mask = attention_mask[keep_idx]
            if attn_mask is not None:
                attn_mask = attn_mask[keep_idx]
            if visual_token_mask is not None:
                visual_token_mask = visual_token_mask[keep_idx]
            keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=generated.device)
            if hasattr(past_key_values, "batch_select_indices"):
                past_key_values.batch_select_indices(keep_idx_tensor)
    finally:
        if attn_config_values is not None:
            _restore_text_attn_implementation(attn_config_values)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


def generate_pure_soft(model, tokenizer, **kwargs):
    """Generate with probability-weighted token embeddings at every decode step."""
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    image_pad_bias = kwargs.pop("image_pad_bias", False)
    image_pad_bias_lambda = kwargs.pop("image_pad_bias_lambda", 0.0)
    image_pad_bias_min_step = kwargs.pop("image_pad_bias_min_step", 0)
    image_pad_bias_max_step = kwargs.pop("image_pad_bias_max_step", None)
    image_pad_bias_entropy_min = kwargs.pop("image_pad_bias_entropy_min", None)
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    temperature = kwargs.get("temperature", 1.0)
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 0)
    min_p = kwargs.get("min_p", 0)
    max_new_tokens = kwargs.get("max_new_tokens", 32768)
    do_sample = kwargs.get("do_sample", True)

    stream_callback = kwargs.pop("stream_callback", None)
    token_trace = kwargs.pop("token_trace", None)
    trace_topk = kwargs.pop("trace_topk", 0)
    trace_event_geometry = bool(kwargs.pop("trace_event_geometry", False))
    trace_event_steps = {int(value) for value in kwargs.pop("trace_event_steps", [0, 1, 2, 4, 8, 16, 32])}
    trace_route_override_step = int(kwargs.pop("trace_route_override_step", -1))
    trace_route_override_kind = str(kwargs.pop("trace_route_override_kind", "none"))
    trace_forced_answer_probe = bool(kwargs.pop("trace_forced_answer_probe", False))
    trace_probe_gold_choice = kwargs.pop("trace_probe_gold_choice", None)
    trace_probe_choice_case = kwargs.pop("trace_probe_choice_case", "upper")
    collapse_on_diffuse = kwargs.pop("collapse_on_diffuse", False)
    collapse_entropy_window = kwargs.pop("collapse_entropy_window", 16)
    collapse_entropy_alpha = kwargs.pop("collapse_entropy_alpha", 2.0)
    collapse_min_history = kwargs.pop("collapse_min_history", 4)
    collapse_min_entropy = kwargs.pop("collapse_min_entropy", 1.0)
    collapse_low_conf_tau = kwargs.pop("collapse_low_conf_tau", 0.20)
    collapse_low_margin_tau = kwargs.pop("collapse_low_margin_tau", 0.05)
    collapse_min_step = kwargs.pop("collapse_min_step", 0)
    collapse_patience = kwargs.pop("collapse_patience", 1)
    collapse_patience_window = kwargs.pop("collapse_patience_window", 16)
    collapse_require_repeat_degen = kwargs.pop("collapse_require_repeat_degen", False)
    collapse_repeat_ngram = kwargs.pop("collapse_repeat_ngram", 0)
    collapse_recent_repeat_window = kwargs.pop("collapse_recent_repeat_window", 32)
    collapse_recent_repeat_tau = kwargs.pop("collapse_recent_repeat_tau", 0.0)
    format_cooldown = kwargs.pop("format_cooldown", False)
    format_cooldown_steps = kwargs.pop("format_cooldown_steps", 0)
    format_cooldown_min_step = kwargs.pop("format_cooldown_min_step", 0)
    format_cooldown_highrisk_only = kwargs.pop("format_cooldown_highrisk_only", False)
    format_cooldown_normal_steps = kwargs.pop("format_cooldown_normal_steps", None)
    format_cooldown_highrisk_steps = kwargs.pop("format_cooldown_highrisk_steps", None)
    format_cooldown_mix_lambda = kwargs.pop("format_cooldown_mix_lambda", 1.0)
    format_cooldown_max_active = kwargs.pop("format_cooldown_max_active", 0)
    format_cooldown_entropy_min = kwargs.pop("format_cooldown_entropy_min", None)
    format_cooldown_top1_max = kwargs.pop("format_cooldown_top1_max", None)
    format_cooldown_margin_max = kwargs.pop("format_cooldown_margin_max", None)
    answer_zone_discrete = kwargs.pop("answer_zone_discrete", False)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    batch_size, device = input_ids.shape[0], input_ids.device
    prompt_len = input_ids.shape[1]
    visual_token_mask = _build_visual_token_mask(input_ids, tokenizer)
    prompt_hidden_states = None
    E = model.get_input_embeddings().weight
    image_pad_emb = None
    if image_pad_bias and float(image_pad_bias_lambda) > 0.0:
        imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if isinstance(imgpad_id, int) and imgpad_id >= 0:
            image_pad_emb = E[imgpad_id]
    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    prompt_lens = [len(ids) for ids in all_generated]
    unfinished_idx = list(range(batch_size))
    past_key_values = None
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    last_emb = None
    entropy_history = [[] for _ in range(batch_size)]
    collapse_candidate_history = [[] for _ in range(batch_size)]
    format_cooldowns = [0 for _ in range(batch_size)]
    format_cooldown_active_counts = [0 for _ in range(batch_size)]
    answer_zone_active = [False for _ in range(batch_size)]

    for step in range(max_new_tokens):
        cur_batch = attention_mask.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {"input_ids": input_ids.clone()}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if vision_inputs:
                model_inputs.update(vision_inputs)
            model_inputs["cache_position"] = cache_position
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "inputs_embeds": last_emb.unsqueeze(1),
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "cache_position": cache_position,
            }

        need_prompt_hidden = trace_event_geometry and past_key_values is None
        with torch.no_grad():
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=need_prompt_hidden,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        if need_prompt_hidden and getattr(outputs, "hidden_states", None):
            prompt_hidden_states = outputs.hidden_states[-1][:, :prompt_len, :].detach()
        if vision_inputs:
            vision_inputs = {}
        cache_position = cache_position[-1:] + 1

        logits_original = outputs.logits[:, -1, :]
        probs_original = F.softmax(logits_original, dim=-1)
        raw_entropy = -(
            probs_original * probs_original.clamp(min=1e-8).log()
        ).sum(dim=-1)

        logits = logits_original / temperature
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
        probs = F.softmax(logits_filtered, dim=-1)
        filtered_entropy = -(
            probs * probs.clamp(min=1e-8).log()
        ).sum(dim=-1)

        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        selected_prob = probs[torch.arange(cur_batch, device=device), next_tokens]
        raw_selected_prob = probs_original[torch.arange(cur_batch, device=device), next_tokens]
        soft_emb = torch.matmul(probs_original, E)
        normal_emb = E[next_tokens]
        raw_top2 = torch.topk(probs_original, k=min(2, probs_original.shape[-1]), dim=-1).values
        raw_top1_prob = raw_top2[:, 0]
        raw_margin = (
            raw_top2[:, 0] - raw_top2[:, 1]
            if raw_top2.shape[-1] > 1
            else raw_top2[:, 0]
        )
        spike_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        diffuse_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        repeat_degen_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        collapse_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        format_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        format_token_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        highrisk_format_token_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        image_pad_bias_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        answer_zone_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        answer_zone_trigger_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        entropy_delta = torch.zeros(cur_batch, dtype=raw_entropy.dtype, device=device)
        entropy_history_count = torch.zeros(cur_batch, dtype=torch.long, device=device)
        if collapse_on_diffuse:
            spike_mask, entropy_delta, entropy_history_count = _entropy_spike_mask(
                raw_entropy=raw_entropy,
                entropy_history=entropy_history,
                window=collapse_entropy_window,
                alpha=collapse_entropy_alpha,
                min_history=collapse_min_history,
                min_entropy=collapse_min_entropy,
            )
            diffuse_mask = (
                (raw_top1_prob < float(collapse_low_conf_tau))
                | (raw_margin < float(collapse_low_margin_tau))
            )
            candidate_mask = spike_mask & diffuse_mask
            if collapse_min_step > 0:
                candidate_mask = candidate_mask & (step >= int(collapse_min_step))

            refined_mask = torch.zeros_like(candidate_mask)
            for bi, orig in enumerate(unfinished_idx):
                if not bool(candidate_mask[bi].item()):
                    continue

                recent_candidates = (
                    collapse_candidate_history[orig][-int(collapse_patience_window):]
                    if collapse_patience_window > 0
                    else collapse_candidate_history[orig]
                )
                enough_patience = (
                    sum(recent_candidates) + 1 >= max(1, int(collapse_patience))
                )
                if not enough_patience:
                    continue

                generated_only = all_generated[orig][prompt_lens[orig]:]
                repeat_degen = False
                ngram = int(collapse_repeat_ngram)
                if ngram > 0 and len(generated_only) >= ngram * 2:
                    last_ngram = tuple(generated_only[-ngram:])
                    prior = generated_only[:-ngram]
                    repeat_degen = any(
                        tuple(prior[i:i + ngram]) == last_ngram
                        for i in range(0, len(prior) - ngram + 1)
                    )

                repeat_tau = float(collapse_recent_repeat_tau)
                if repeat_tau > 0.0 and generated_only:
                    window = max(1, int(collapse_recent_repeat_window))
                    recent = generated_only[-window:]
                    duplicate_ratio = 1.0 - (len(set(recent)) / max(1, len(recent)))
                    repeat_degen = repeat_degen or (duplicate_ratio >= repeat_tau)

                repeat_degen_mask[bi] = repeat_degen
                if collapse_require_repeat_degen and not repeat_degen:
                    continue

                refined_mask[bi] = True
            collapse_mask = refined_mask
        if format_cooldown and int(format_cooldown_steps) > 0 and step >= int(format_cooldown_min_step):
            for bi, orig in enumerate(unfinished_idx):
                token_text = tokenizer.decode(
                    [int(next_tokens[bi].item())],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                is_highrisk = _is_high_risk_format_token_text(token_text)
                is_format = is_highrisk if format_cooldown_highrisk_only else _is_format_token_text(token_text)
                highrisk_format_token_mask[bi] = is_highrisk
                if is_format and (
                    format_cooldown_entropy_min is not None
                    or format_cooldown_top1_max is not None
                    or format_cooldown_margin_max is not None
                ):
                    unstable = False
                    if format_cooldown_entropy_min is not None:
                        unstable = unstable or (
                            float(raw_entropy[bi].item()) >= float(format_cooldown_entropy_min)
                        )
                    if format_cooldown_top1_max is not None:
                        unstable = unstable or (
                            float(raw_top1_prob[bi].item()) <= float(format_cooldown_top1_max)
                        )
                    if format_cooldown_margin_max is not None:
                        unstable = unstable or (
                            float(raw_margin[bi].item()) <= float(format_cooldown_margin_max)
                        )
                    is_format = unstable
                format_token_mask[bi] = is_format
                format_mask[bi] = is_format or format_cooldowns[orig] > 0
                if int(format_cooldown_max_active) > 0 and format_cooldown_active_counts[orig] >= int(format_cooldown_max_active):
                    format_mask[bi] = False
        if image_pad_emb is not None:
            image_pad_bias_mask = torch.ones(cur_batch, dtype=torch.bool, device=device)
            if int(image_pad_bias_min_step) > 0:
                image_pad_bias_mask = image_pad_bias_mask & (step >= int(image_pad_bias_min_step))
            if image_pad_bias_max_step is not None:
                image_pad_bias_mask = image_pad_bias_mask & (step <= int(image_pad_bias_max_step))
            if image_pad_bias_entropy_min is not None:
                image_pad_bias_mask = image_pad_bias_mask & (
                    raw_entropy >= float(image_pad_bias_entropy_min)
                )
        if answer_zone_discrete:
            for bi, orig in enumerate(unfinished_idx):
                recent_ids = all_generated[orig][prompt_lens[orig]:] + [int(next_tokens[bi].item())]
                recent_text = tokenizer.decode(
                    recent_ids[-16:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ).lower()
                triggered = ("</think" in recent_text) or ("answer" in recent_text)
                answer_zone_trigger_mask[bi] = triggered
                answer_zone_mask[bi] = answer_zone_active[orig] or triggered

        format_lambda = max(0.0, min(1.0, float(format_cooldown_mix_lambda)))
        biased_soft_emb = soft_emb
        if image_pad_emb is not None:
            bias_lambda = max(0.0, min(1.0, float(image_pad_bias_lambda)))
            mixed_soft_emb = (1.0 - bias_lambda) * soft_emb + bias_lambda * image_pad_emb.to(soft_emb.device)
            biased_soft_emb = torch.where(image_pad_bias_mask[:, None], mixed_soft_emb, soft_emb)
        format_emb = format_lambda * normal_emb + (1.0 - format_lambda) * biased_soft_emb
        last_emb = biased_soft_emb
        last_emb = torch.where(format_mask[:, None], format_emb, last_emb)
        last_emb = torch.where((collapse_mask | answer_zone_mask)[:, None], normal_emb, last_emb)
        route_override_active = step == trace_route_override_step and trace_route_override_kind != "none"
        if route_override_active:
            if trace_route_override_kind == "hard":
                last_emb = normal_emb
            elif trace_route_override_kind == "raw_soft":
                last_emb = soft_emb
            elif trace_route_override_kind == "method_soft":
                last_emb = biased_soft_emb
            else:
                raise ValueError(f"Unsupported trace_route_override_kind={trace_route_override_kind}")
        forced_answer_probe = None
        if trace_forced_answer_probe and step == trace_route_override_step:
            forced_answer_probe = _forced_answer_probe_from_route(
                model, tokenizer, last_emb, attention_mask, past_key_values,
                cache_position, trace_probe_gold_choice,
                choice_case=trace_probe_choice_case,
                prompt_hidden_states=prompt_hidden_states,
                visual_token_mask=visual_token_mask,
                prompt_len=prompt_len,
            )

        for bi, orig in enumerate(unfinished_idx):
            token_id = next_tokens[bi].item()
            all_generated[orig].append(token_id)
            if token_trace is not None:
                phase = (
                    "early"
                    if step <= 128
                    else "mid" if step <= 512 else "late"
                )
                image_candidate = bool(image_pad_bias_mask[bi].item())
                answer_active = bool(answer_zone_mask[bi].item())
                collapse_active = bool(collapse_mask[bi].item())
                format_active = bool(format_mask[bi].item())
                if answer_active:
                    route_signal = "answer_zone"
                    route_action = "hard_discrete"
                    route_priority = 100
                elif collapse_active:
                    route_signal = (
                        "diffuse_repeat_degen"
                        if bool(repeat_degen_mask[bi].item())
                        else "diffuse_low_conf_spike"
                    )
                    route_action = "hard_discrete"
                    route_priority = 90
                elif format_active:
                    route_signal = (
                        "highrisk_format_uncertain"
                        if bool(highrisk_format_token_mask[bi].item())
                        else "format_uncertain"
                    )
                    route_action = "format_cooldown"
                    route_priority = 80
                elif image_candidate:
                    route_signal = f"{phase}_visual_bias"
                    route_action = "image_pad_bias"
                    route_priority = 50
                else:
                    route_signal = "default"
                    route_action = "pure_soft"
                    route_priority = 0

                suppressed = []
                if image_candidate and answer_active:
                    suppressed.append("image_pad_bias_by_answer_zone")
                if image_candidate and collapse_active:
                    suppressed.append("image_pad_bias_by_collapse")
                if image_candidate and format_active:
                    suppressed.append("image_pad_bias_by_format")

                record = {
                    "step": int(step),
                    "batch_index": int(orig),
                    "token_id": int(token_id),
                    "raw_entropy": float(raw_entropy[bi].item()),
                    "filtered_entropy": float(filtered_entropy[bi].item()),
                    "selected_prob": float(selected_prob[bi].item()),
                    "raw_selected_prob": float(raw_selected_prob[bi].item()),
                    "confidence": float(selected_prob[bi].item()),
                    "mode": (
                        "collapsed"
                        if bool(collapse_mask[bi].item())
                        else ("format_cooldown" if bool(format_mask[bi].item()) else "pure_soft")
                    ),
                        "generation_phase": phase,
                        "route_signal": route_signal,
                        "route_action": route_action,
                        "route_priority": int(route_priority),
                        "route_suppressed_by": suppressed,
                        "collapse_on_diffuse": bool(collapse_mask[bi].item()),
                        "format_cooldown_active": bool(format_mask[bi].item()),
                        "format_token": bool(format_token_mask[bi].item()),
                        "is_highrisk_format_token": bool(highrisk_format_token_mask[bi].item()),
                        "format_cooldown_highrisk_only": bool(format_cooldown_highrisk_only),
                        "format_cooldown_min_step": int(format_cooldown_min_step),
                        "format_cooldown_normal_steps": (
                            None if format_cooldown_normal_steps is None else int(format_cooldown_normal_steps)
                        ),
                        "format_cooldown_highrisk_steps": (
                            None if format_cooldown_highrisk_steps is None else int(format_cooldown_highrisk_steps)
                        ),
                        "format_cooldown_mix_lambda": float(format_lambda),
                        "format_cooldown_max_active": int(format_cooldown_max_active),
                        "format_cooldown_entropy_min": (
                            None if format_cooldown_entropy_min is None else float(format_cooldown_entropy_min)
                        ),
                        "format_cooldown_top1_max": (
                            None if format_cooldown_top1_max is None else float(format_cooldown_top1_max)
                        ),
                        "format_cooldown_margin_max": (
                            None if format_cooldown_margin_max is None else float(format_cooldown_margin_max)
                        ),
                        "image_pad_bias_active": bool(image_pad_bias_mask[bi].item()),
                        "visual_bias_candidate": bool(image_candidate),
                        "visual_bias_effective": (
                            bool(image_candidate)
                            and not bool(answer_active)
                            and not bool(collapse_active)
                            and not bool(format_active)
                        ),
                        "image_pad_bias_lambda": float(image_pad_bias_lambda),
                        "image_pad_bias_min_step": int(image_pad_bias_min_step),
                        "image_pad_bias_max_step": (
                            None if image_pad_bias_max_step is None else int(image_pad_bias_max_step)
                        ),
                        "image_pad_bias_entropy_min": (
                            None if image_pad_bias_entropy_min is None else float(image_pad_bias_entropy_min)
                        ),
                        "format_cooldown_active_count": int(format_cooldown_active_counts[orig]),
                        "format_cooldown_remaining": int(format_cooldowns[orig]),
                        "answer_zone_discrete_active": bool(answer_zone_mask[bi].item()),
                        "answer_zone_trigger": bool(answer_zone_trigger_mask[bi].item()),
                        "collapse_entropy_delta": float(entropy_delta[bi].item()),
                    "collapse_entropy_history_count": int(entropy_history_count[bi].item()),
                    "raw_top1_prob": float(raw_top1_prob[bi].item()),
                    "raw_margin": float(raw_margin[bi].item()),
                    "route_override_active": bool(route_override_active),
                    "route_override_kind": trace_route_override_kind if route_override_active else None,
                    "forced_answer_probe": forced_answer_probe,
                    "entropy_spike_mask": bool(spike_mask[bi].item()),
                    "diffuse_mask": bool(diffuse_mask[bi].item()),
                    "repeat_degen_detected": bool(repeat_degen_mask[bi].item()),
                    "collapse_candidate": bool((spike_mask[bi] & diffuse_mask[bi]).item()) if collapse_on_diffuse else False,
                }
                trace_event = bool(
                    step in trace_event_steps
                    or format_mask[bi].item()
                    or collapse_mask[bi].item()
                    or answer_zone_trigger_mask[bi].item()
                )
                record["trace_event"] = trace_event
                record["trace_event_kind"] = (
                    "format" if bool(format_mask[bi].item())
                    else "collapse" if bool(collapse_mask[bi].item())
                    else "answer_zone" if bool(answer_zone_trigger_mask[bi].item())
                    else "checkpoint" if step in trace_event_steps
                    else None
                )
                if trace_event_geometry and trace_event:
                    record.update(_embedding_geometry_record(
                        normal_emb,
                        soft_emb,
                        last_emb,
                        bi,
                        visual_anchor=image_pad_emb,
                    ))
                _add_topk_trace_fields(
                    record,
                    tokenizer,
                    probs_original,
                    probs,
                    bi,
                    trace_topk,
                )
                token_trace.append(record)
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        for bi in range(cur_batch):
            entropy_history[bi].append(float(raw_entropy[bi].item()))
            if collapse_on_diffuse:
                collapse_candidate_history[unfinished_idx[bi]].append(
                    bool((spike_mask[bi] & diffuse_mask[bi]).item())
                )
            if format_cooldown and int(format_cooldown_steps) > 0:
                orig = unfinished_idx[bi]
                if step < int(format_cooldown_min_step):
                    format_cooldowns[orig] = 0
                elif int(format_cooldown_max_active) > 0 and format_cooldown_active_counts[orig] >= int(format_cooldown_max_active):
                    format_cooldowns[orig] = 0
                elif bool(format_token_mask[bi].item()):
                    token_text = tokenizer.decode(
                        [int(next_tokens[bi].item())],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    is_highrisk = _is_high_risk_format_token_text(token_text)
                    if is_highrisk and format_cooldown_highrisk_steps is not None:
                        steps = int(format_cooldown_highrisk_steps)
                    elif (not is_highrisk) and format_cooldown_normal_steps is not None:
                        steps = int(format_cooldown_normal_steps)
                    else:
                        steps = int(format_cooldown_steps)
                    format_cooldowns[orig] = max(0, steps - 1)
                elif format_cooldowns[orig] > 0:
                    format_cooldowns[orig] -= 1
                if bool(format_mask[bi].item()):
                    format_cooldown_active_counts[orig] += 1
            if answer_zone_discrete and bool(answer_zone_trigger_mask[bi].item()):
                answer_zone_active[unfinished_idx[bi]] = True

        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break

        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        entropy_history = [entropy_history[i] for i in keep_idx.tolist()]
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
        if hasattr(past_key_values, "batch_select_indices"):
            past_key_values.batch_select_indices(keep_idx_tensor)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


def generate_cot_visual_reanchor(model, tokenizer, **kwargs):
    """COT decoding with gated visual-anchor injection on the next-step embedding."""
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    temperature = kwargs.get("temperature", 1.0)
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 0)
    min_p = kwargs.get("min_p", 0)
    max_new_tokens = kwargs.get("max_new_tokens", 32768)
    do_sample = kwargs.get("do_sample", True)

    reanchor_entropy_threshold = kwargs.pop("reanchor_entropy_threshold", 1.0)
    reanchor_visual_attn_threshold = kwargs.pop("reanchor_visual_attn_threshold", 0.12)
    reanchor_lambda = kwargs.pop("reanchor_lambda", 0.15)
    reanchor_top_m = kwargs.pop("reanchor_top_m", 4)
    reanchor_attn_last_k = kwargs.pop("reanchor_attn_last_k", 4)
    reanchor_max_trigger_count = kwargs.pop("reanchor_max_trigger_count", 1)
    reanchor_cooldown = kwargs.pop("reanchor_cooldown", 32)
    reanchor_min_step = kwargs.pop("reanchor_min_step", None)
    reanchor_max_step = kwargs.pop("reanchor_max_step", None)
    reanchor_anchor_mode = kwargs.pop("reanchor_anchor_mode", "dynamic")
    reanchor_trigger_mode = kwargs.pop("reanchor_trigger_mode", "absolute")
    reanchor_rolling_window = kwargs.pop("reanchor_rolling_window", 8)
    reanchor_min_history = kwargs.pop("reanchor_min_history", 3)
    reanchor_entropy_delta_threshold = kwargs.pop("reanchor_entropy_delta_threshold", 0.5)
    reanchor_visual_drop_threshold = kwargs.pop("reanchor_visual_drop_threshold", 0.03)

    stream_callback = kwargs.pop("stream_callback", None)
    token_trace = kwargs.pop("token_trace", None)
    log_visual_attn_summary = kwargs.pop("log_visual_attn_summary", False)
    visual_attn_summary_last_k = kwargs.pop("visual_attn_summary_last_k", 4)

    batch_size = input_ids.shape[0]
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    visual_token_mask = _build_visual_token_mask(input_ids, tokenizer)
    prompt_hidden_states = None
    E = model.get_input_embeddings().weight

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    attn_mask = attention_mask.clone() if attention_mask is not None else None
    past_key_values = None
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    last_emb = None
    trigger_count = torch.zeros(batch_size, dtype=torch.long, device=device)
    cooldown_remaining = torch.zeros(batch_size, dtype=torch.long, device=device)
    entropy_history = [[] for _ in range(batch_size)]
    visual_mass_history = [[] for _ in range(batch_size)]

    attn_config_values = _get_text_attn_implementation(model)
    _set_text_attn_implementation(attn_config_values, "eager")

    try:
        for step in range(max_new_tokens):
            cur_batch = attn_mask.shape[0] if attn_mask is not None else len(unfinished_idx)
            if cur_batch == 0:
                break

            if past_key_values is None:
                model_inputs = {"input_ids": input_ids.clone()}
                if attn_mask is not None:
                    model_inputs["attention_mask"] = attn_mask
                if vision_inputs:
                    model_inputs.update(vision_inputs)
                model_inputs["cache_position"] = cache_position
            else:
                if attn_mask is not None:
                    attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
                    attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
                model_inputs = {
                    "inputs_embeds": last_emb.unsqueeze(1),
                    "past_key_values": past_key_values,
                    "cache_position": cache_position,
                }
                if attn_mask is not None:
                    model_inputs["attention_mask"] = attn_mask

            need_attn = past_key_values is not None
            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    use_cache=True,
                    output_attentions=need_attn,
                    output_hidden_states=(prompt_hidden_states is None),
                )
            past_key_values = outputs.past_key_values
            if prompt_hidden_states is None:
                prompt_hidden_states = outputs.hidden_states[-1][:, :prompt_len, :].detach()
            if vision_inputs:
                vision_inputs = {}
            cache_position = cache_position[-1:] + 1

            visual_attn_summary = None
            if need_attn and log_visual_attn_summary:
                visual_attn_summary = _summarize_visual_attention(
                    attn_layers=outputs.attentions,
                    visual_token_mask=visual_token_mask,
                    prompt_len=prompt_len,
                    attn_last_k=visual_attn_summary_last_k,
                )

            next_token_logits = outputs.logits[:, -1, :]
            raw_probs = F.softmax(next_token_logits, dim=-1)
            raw_entropy = -(
                raw_probs * raw_probs.clamp(min=1e-8).log()
            ).sum(dim=-1)
            logits = next_token_logits / temperature
            logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
            probs = F.softmax(logits, dim=-1)
            filtered_entropy = -(
                probs * probs.clamp(min=1e-8).log()
            ).sum(dim=-1)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            next_emb = E[next_tokens]
            reanchor_triggered = torch.zeros(cur_batch, dtype=torch.bool, device=device)
            entropy_delta = torch.zeros(cur_batch, dtype=raw_entropy.dtype, device=device)
            visual_drop = torch.zeros(cur_batch, dtype=raw_entropy.dtype, device=device)
            entropy_history_count = torch.zeros(cur_batch, dtype=torch.long, device=device)
            visual_history_count = torch.zeros(cur_batch, dtype=torch.long, device=device)

            window_size = max(1, int(reanchor_rolling_window))
            for bi in range(cur_batch):
                ent_hist = entropy_history[bi][-window_size:]
                entropy_history_count[bi] = len(ent_hist)
                if ent_hist:
                    prev_entropy = raw_entropy.new_tensor(ent_hist).mean()
                    entropy_delta[bi] = raw_entropy[bi] - prev_entropy
                vis_hist = visual_mass_history[bi][-window_size:]
                visual_history_count[bi] = len(vis_hist)
                if vis_hist and visual_attn_summary is not None:
                    prev_visual = raw_entropy.new_tensor(vis_hist).mean()
                    visual_drop[bi] = prev_visual - visual_attn_summary["mass"][bi].to(raw_entropy.device)

            if need_attn and outputs.attentions is not None:
                trigger_mask = (raw_entropy >= float(reanchor_entropy_threshold))
                if visual_attn_summary is not None:
                    trigger_mask = trigger_mask & visual_attn_summary["available"]
                    trigger_mask = trigger_mask & (
                        visual_attn_summary["mass"] <= float(reanchor_visual_attn_threshold)
                    )
                else:
                    trigger_mask = torch.zeros_like(trigger_mask)
                if reanchor_trigger_mode == "absolute":
                    pass
                elif reanchor_trigger_mode == "entropy_delta":
                    trigger_mask = trigger_mask & (entropy_history_count >= int(reanchor_min_history))
                    trigger_mask = trigger_mask & (
                        entropy_delta >= float(reanchor_entropy_delta_threshold)
                    )
                elif reanchor_trigger_mode == "visual_drop":
                    trigger_mask = trigger_mask & (visual_history_count >= int(reanchor_min_history))
                    trigger_mask = trigger_mask & (
                        visual_drop >= float(reanchor_visual_drop_threshold)
                    )
                elif reanchor_trigger_mode == "entropy_delta_visual_drop":
                    trigger_mask = trigger_mask & (entropy_history_count >= int(reanchor_min_history))
                    trigger_mask = trigger_mask & (visual_history_count >= int(reanchor_min_history))
                    trigger_mask = trigger_mask & (
                        entropy_delta >= float(reanchor_entropy_delta_threshold)
                    )
                    trigger_mask = trigger_mask & (
                        visual_drop >= float(reanchor_visual_drop_threshold)
                    )
                else:
                    raise ValueError(f"Unsupported reanchor_trigger_mode: {reanchor_trigger_mode}")
                if reanchor_min_step is not None:
                    trigger_mask = trigger_mask & (step >= int(reanchor_min_step))
                if reanchor_max_step is not None:
                    trigger_mask = trigger_mask & (step <= int(reanchor_max_step))
                trigger_mask = trigger_mask & (trigger_count < int(reanchor_max_trigger_count))
                trigger_mask = trigger_mask & (cooldown_remaining <= 0)

                if trigger_mask.any():
                    dynamic_anchor, has_anchor = _compute_dynamic_visual_anchor(
                        attn_layers=outputs.attentions,
                        soft_emb=torch.matmul(raw_probs, E),
                        prompt_hidden_states=prompt_hidden_states,
                        visual_token_mask=visual_token_mask,
                        prompt_len=prompt_len,
                        top_m=reanchor_top_m,
                        attn_last_k=reanchor_attn_last_k,
                        anchor_mode=reanchor_anchor_mode,
                    )
                    apply_anchor = trigger_mask & has_anchor.to(trigger_mask.device)
                    if apply_anchor.any():
                        next_emb = torch.where(
                            apply_anchor[:, None],
                            (1.0 - float(reanchor_lambda)) * next_emb
                            + float(reanchor_lambda) * dynamic_anchor.to(next_emb.device),
                            next_emb,
                        )
                        reanchor_triggered = apply_anchor
                        trigger_count = trigger_count + apply_anchor.long()
                        cooldown_remaining = torch.where(
                            apply_anchor,
                            torch.full_like(cooldown_remaining, int(reanchor_cooldown)),
                            cooldown_remaining,
                        )

            cooldown_remaining = torch.clamp(cooldown_remaining - 1, min=0)

            for bi, orig in enumerate(unfinished_idx):
                token_id = next_tokens[bi].item()
                all_generated[orig].append(token_id)
                if token_trace is not None:
                    record = {
                        "step": int(step),
                        "batch_index": int(orig),
                        "token_id": int(token_id),
                        "raw_entropy": float(raw_entropy[bi].item()),
                        "filtered_entropy": float(filtered_entropy[bi].item()),
                        "selected_prob": float(probs[bi, next_tokens[bi]].item()),
                        "mode": "normal",
                        "reanchor_triggered": bool(reanchor_triggered[bi].item()),
                        "reanchor_trigger_count": int(trigger_count[bi].item()),
                        "reanchor_trigger_mode": reanchor_trigger_mode,
                        "reanchor_entropy_delta": float(entropy_delta[bi].item()),
                        "reanchor_visual_drop": float(visual_drop[bi].item()),
                        "reanchor_entropy_history_count": int(entropy_history_count[bi].item()),
                        "reanchor_visual_history_count": int(visual_history_count[bi].item()),
                    }
                    if log_visual_attn_summary and visual_attn_summary is None:
                        record.update({
                            "visual_attn_available": False,
                            "visual_attn_mass": 0.0,
                            "visual_attn_top1": 0.0,
                            "visual_attn_top4_sum": 0.0,
                            "visual_attn_entropy": None,
                            "visual_attn_token_count": 0,
                        })
                    elif visual_attn_summary is not None:
                        record.update({
                            "visual_attn_available": bool(visual_attn_summary["available"][bi].item()),
                            "visual_attn_mass": float(visual_attn_summary["mass"][bi].item()),
                            "visual_attn_top1": float(visual_attn_summary["top1"][bi].item()),
                            "visual_attn_top4_sum": float(visual_attn_summary["top4_sum"][bi].item()),
                            "visual_attn_entropy": (
                                float(visual_attn_summary["entropy"][bi].item())
                                if visual_attn_summary["available"][bi].item()
                                else None
                            ),
                            "visual_attn_token_count": int(visual_attn_summary["token_count"][bi].item()),
                        })
                    token_trace.append(record)
                if stream_callback is not None:
                    stream_callback(all_generated[orig][-1])

            for bi in range(cur_batch):
                entropy_history[bi].append(float(raw_entropy[bi].item()))
                if visual_attn_summary is not None and visual_attn_summary["available"][bi].item():
                    visual_mass_history[bi].append(float(visual_attn_summary["mass"][bi].item()))

            if tokenizer.eos_token_id is not None:
                cur_finished = (next_tokens == tokenizer.eos_token_id)
            else:
                cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)
            keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
            unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

            if len(unfinished_idx) == 0:
                break
            last_emb = next_emb[keep_idx]
            if attention_mask is not None:
                attention_mask = attention_mask[keep_idx]
            if attn_mask is not None:
                attn_mask = attn_mask[keep_idx]
            visual_token_mask = visual_token_mask[keep_idx]
            prompt_hidden_states = prompt_hidden_states[keep_idx]
            trigger_count = trigger_count[keep_idx]
            cooldown_remaining = cooldown_remaining[keep_idx]
            entropy_history = [entropy_history[i] for i in keep_idx.tolist()]
            visual_mass_history = [visual_mass_history[i] for i in keep_idx.tolist()]
            keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
            if hasattr(past_key_values, "batch_select_indices"):
                past_key_values.batch_select_indices(keep_idx_tensor)
    finally:
        _restore_text_attn_implementation(attn_config_values)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


def _compute_early_actual_visual_anchor(
    prompt_hidden_states,
    visual_token_mask,
    query_state,
    reference_emb,
    top_m=8,
    temperature=0.10,
):
    """Build a question-conditioned, norm-matched anchor from visual states."""
    if int(top_m) <= 0:
        raise ValueError("top_m must be positive")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    device = reference_emb.device
    prompt_hidden_states = prompt_hidden_states.to(device)
    visual_token_mask = visual_token_mask.to(device)
    query_state = query_state.to(device)
    anchors = reference_emb.clone()
    applied = torch.zeros(reference_emb.shape[0], dtype=torch.bool, device=device)
    query_similarity = torch.zeros(
        reference_emb.shape[0], dtype=reference_emb.dtype, device=device
    )
    norm_ratio = torch.ones(
        reference_emb.shape[0], dtype=reference_emb.dtype, device=device
    )

    normalized_visual = F.normalize(prompt_hidden_states.float(), dim=-1)
    normalized_query = F.normalize(query_state.float(), dim=-1)
    for bi in range(reference_emb.shape[0]):
        positions = visual_token_mask[bi, : prompt_hidden_states.shape[1]].nonzero(
            as_tuple=False
        ).squeeze(-1)
        if positions.numel() == 0:
            continue
        scores = torch.matmul(normalized_visual[bi, positions], normalized_query[bi])
        count = min(int(top_m), int(positions.numel()))
        top_scores, top_indices = torch.topk(scores, k=count, dim=0)
        selected = prompt_hidden_states[bi, positions[top_indices]]
        weights = F.softmax(top_scores / float(temperature), dim=0).to(selected.dtype)
        raw_anchor = torch.sum(selected * weights.unsqueeze(-1), dim=0)
        raw_norm = torch.linalg.vector_norm(raw_anchor.float()).clamp_min(1e-8)
        ref_norm = torch.linalg.vector_norm(reference_emb[bi].float())
        ratio = ref_norm / raw_norm
        anchors[bi] = raw_anchor * ratio.to(raw_anchor.dtype)
        applied[bi] = True
        query_similarity[bi] = torch.sum(weights.float() * top_scores).to(
            query_similarity.dtype
        )
        norm_ratio[bi] = ratio.to(norm_ratio.dtype)

    return anchors, applied, query_similarity, norm_ratio


def generate_lead(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)

    # ---- lead ----
    alpha_0                = kwargs.pop("alpha_0", 1.0) # adjustable
    beta_0                 = kwargs.pop("beta_0", 0.7)
    window_size            = kwargs.pop("window_size", 256) #“冷却时间 / 稳定窗口”，用来防止 soft 模式与 normal 模式之间的频繁来回震荡
    thinking_token_id      = kwargs.pop("thinking_token_id", None)
    end_thinking_token_id  = kwargs.pop("end_thinking_token_id", None)
    max_switch_count       = kwargs.pop("max_switch_count", None) # adjustable for efficiency
    math_ids_tensor        = kwargs.pop("math_ids_tensor", None)
    convergence_words      = kwargs.get("convergence_words", "</think>")
    termination_words      = kwargs.get("termination_words", "</think>\n\nThe final answer is")
    termination_max_tokens = kwargs.pop("termination_max_tokens", 32)
    forced_prefix_ids = kwargs.pop("forced_prefix_ids", None)
    if forced_prefix_ids is not None:
        forced_prefix_ids = [int(token_id) for token_id in forced_prefix_ids]
    capture_logits_steps = {
        int(step) for step in (kwargs.pop("capture_logits_steps", None) or [])
    }
    capture_logits_sink = kwargs.pop("capture_logits_sink", None)

    stream_callback       = kwargs.pop("stream_callback", None)
    token_trace           = kwargs.pop("token_trace", None)
    trace_online_residual_provider = kwargs.pop("trace_online_residual_provider", None)
    trace_topk            = kwargs.pop("trace_topk", 0)
    trace_event_geometry  = bool(kwargs.pop("trace_event_geometry", False))
    trace_event_steps     = {int(value) for value in kwargs.pop("trace_event_steps", [0, 1, 2, 4, 8, 16, 32])}
    trace_route_override_step = int(kwargs.pop("trace_route_override_step", -1))
    trace_route_override_kind = str(kwargs.pop("trace_route_override_kind", "none"))
    trace_route_override_mix_lambda = float(
        kwargs.pop("trace_route_override_mix_lambda", 0.95)
    )
    trace_route_override_duration = int(
        kwargs.pop("trace_route_override_duration", 1)
    )
    trace_route_cache_rebuild_after_tokens = int(
        kwargs.pop("trace_route_cache_rebuild_after_tokens", -1)
    )
    trace_external_route_vector = kwargs.pop("trace_external_route_vector", None)
    trace_external_route_source = kwargs.pop("trace_external_route_source", None)
    trace_soft_vector_collector = kwargs.pop("trace_soft_vector_collector", None)
    trace_capture_soft_vector_step = int(
        kwargs.pop("trace_capture_soft_vector_step", -1)
    )
    trace_route_override_lambda_max = (
        2.0
        if trace_route_override_kind == "method_plus_external_residual"
        else 1.0
    )
    if not 0.0 <= trace_route_override_mix_lambda <= trace_route_override_lambda_max:
        raise ValueError(
            "trace_route_override_mix_lambda must be in "
            f"[0, {trace_route_override_lambda_max:g}] for "
            f"kind={trace_route_override_kind}"
        )
    if trace_route_override_duration < 1:
        raise ValueError("trace_route_override_duration must be >= 1")
    if trace_route_cache_rebuild_after_tokens == 0:
        raise ValueError(
            "trace_route_cache_rebuild_after_tokens must be -1 or positive"
        )
    if trace_route_cache_rebuild_after_tokens > 0:
        if trace_route_override_step < 0 or trace_route_override_kind == "none":
            raise ValueError("Route cache rebuild requires an active route override")
        if trace_route_cache_rebuild_after_tokens < trace_route_override_duration:
            raise ValueError(
                "Route cache rebuild must occur after the route override finishes"
            )
    trace_forced_answer_probe = bool(kwargs.pop("trace_forced_answer_probe", False))
    trace_probe_gold_choice = kwargs.pop("trace_probe_gold_choice", None)
    trace_probe_choice_case = kwargs.pop("trace_probe_choice_case", "upper")
    lead_soft_veto_on_diffuse = kwargs.pop("lead_soft_veto_on_diffuse", False)
    lead_veto_entropy_window = kwargs.pop("lead_veto_entropy_window", 16)
    lead_veto_entropy_alpha = kwargs.pop("lead_veto_entropy_alpha", 2.0)
    lead_veto_min_history = kwargs.pop("lead_veto_min_history", 4)
    lead_veto_min_entropy = kwargs.pop("lead_veto_min_entropy", 1.0)
    lead_veto_low_conf_tau = kwargs.pop("lead_veto_low_conf_tau", 0.20)
    lead_veto_low_margin_tau = kwargs.pop("lead_veto_low_margin_tau", 0.05)
    lead_veto_min_step = kwargs.pop("lead_veto_min_step", 64)
    lead_veto_require_repeat_degen = kwargs.pop("lead_veto_require_repeat_degen", True)
    lead_veto_repeat_ngram = kwargs.pop("lead_veto_repeat_ngram", 3)
    lead_veto_recent_repeat_window = kwargs.pop("lead_veto_recent_repeat_window", 32)
    lead_veto_recent_repeat_tau = kwargs.pop("lead_veto_recent_repeat_tau", 0.35)
    lead_disable_simple_visual_anchor = kwargs.pop("lead_disable_simple_visual_anchor", False)
    lead_force_normal = kwargs.pop("lead_force_normal", False)
    lead_initial_soft_only = kwargs.pop("lead_initial_soft_only", False)
    lead_initial_transition_only = kwargs.pop("lead_initial_transition_only", False)
    lead_initial_transition_with_refinement = bool(
        kwargs.pop("lead_initial_transition_with_refinement", False)
    )
    lead_force_initial_transition_step1 = bool(
        kwargs.pop("lead_force_initial_transition_step1", False)
    )
    lead_transition_source = str(kwargs.pop("lead_transition_source", "soft"))
    lead_transition_anchor = str(kwargs.pop("lead_transition_anchor", "end_thinking"))
    lead_transition_norm_match = bool(kwargs.pop("lead_transition_norm_match", False))
    lead_transition_random_seed = int(kwargs.pop("lead_transition_random_seed", 42))
    if lead_initial_transition_only and lead_initial_transition_with_refinement:
        raise ValueError(
            "lead_initial_transition_only and lead_initial_transition_with_refinement are mutually exclusive"
        )
    early_transition_enabled = (
        lead_initial_transition_only or lead_initial_transition_with_refinement
    )
    if lead_transition_source not in {"soft", "hard"}:
        raise ValueError("lead_transition_source must be 'soft' or 'hard'")
    if lead_transition_anchor not in {
        "end_thinking", "generated_token", "start_thinking", "newline", "im_end", "random_residual",
    }:
        raise ValueError(
            "Unknown lead_transition_anchor"
        )
    if (
        lead_force_initial_transition_step1
        or lead_transition_source != "soft"
        or lead_transition_anchor != "end_thinking"
    ) and not early_transition_enabled:
        raise ValueError(
            "Custom early-transition bridge controls require an initial-transition mode"
        )
    lead_initial_transition_cache_rebuild_after_step = int(
        kwargs.pop("lead_initial_transition_cache_rebuild_after_step", -1)
    )
    lead_initial_transition_cache_rebuild_prefix_len = int(
        kwargs.pop("lead_initial_transition_cache_rebuild_prefix_len", -1)
    )
    if lead_initial_transition_cache_rebuild_after_step >= 0:
        legacy_prefix_len = lead_initial_transition_cache_rebuild_after_step + 1
        if (
            lead_initial_transition_cache_rebuild_prefix_len >= 0
            and lead_initial_transition_cache_rebuild_prefix_len != legacy_prefix_len
        ):
            raise ValueError("Conflicting cache rebuild step and prefix-length controls")
        lead_initial_transition_cache_rebuild_prefix_len = legacy_prefix_len
    if lead_initial_transition_cache_rebuild_prefix_len == 0:
        raise ValueError("lead_initial_transition_cache_rebuild_prefix_len must be -1 or positive")
    if lead_initial_transition_cache_rebuild_prefix_len >= 0 and not lead_initial_transition_only:
        raise ValueError("Initial-transition cache rebuild requires lead_initial_transition_only")
    if (
        lead_initial_transition_cache_rebuild_prefix_len >= 0
        and trace_route_cache_rebuild_after_tokens > 0
    ):
        raise ValueError("Initial-transition and route cache rebuild cannot be combined")
    lead_initial_transition_hard_boundary_only = bool(
        kwargs.pop("lead_initial_transition_hard_boundary_only", False)
    )
    if lead_initial_transition_hard_boundary_only and not lead_initial_transition_only:
        raise ValueError("lead_initial_transition_hard_boundary_only requires lead_initial_transition_only")
    lead_early_visual_anchor = kwargs.pop("lead_early_visual_anchor", False)
    lead_early_visual_anchor_source = kwargs.pop(
        "lead_early_visual_anchor_source", "visual_hidden"
    )
    lead_early_visual_anchor_top_m = int(
        kwargs.pop("lead_early_visual_anchor_top_m", 8)
    )
    lead_early_visual_anchor_lambda = float(
        kwargs.pop("lead_early_visual_anchor_lambda", 0.10)
    )
    lead_early_visual_anchor_temperature = float(
        kwargs.pop("lead_early_visual_anchor_temperature", 0.10)
    )
    if lead_early_visual_anchor_source not in {"visual_hidden", "image_pad"}:
        raise ValueError("Unsupported lead_early_visual_anchor_source")
    if not 0.0 <= lead_early_visual_anchor_lambda <= 1.0:
        raise ValueError("lead_early_visual_anchor_lambda must be in [0, 1]")
    if lead_early_visual_anchor_top_m <= 0:
        raise ValueError("lead_early_visual_anchor_top_m must be positive")
    if lead_early_visual_anchor_temperature <= 0.0:
        raise ValueError("lead_early_visual_anchor_temperature must be positive")
    lead_initial_transition_delay_steps = int(kwargs.pop("lead_initial_transition_delay_steps", 0) or 0)
    if lead_initial_transition_delay_steps < 0:
        lead_initial_transition_delay_steps = 0
    lead_transition_dynamic_entropy_window = int(
        kwargs.pop("lead_transition_dynamic_entropy_window", 0) or 0
    )
    lead_transition_dynamic_entropy_ratio = float(
        kwargs.pop("lead_transition_dynamic_entropy_ratio", 0.5)
    )
    lead_transition_dynamic_min_history = int(
        kwargs.pop("lead_transition_dynamic_min_history", 2)
    )
    lead_transition_dynamic_max_step = int(
        kwargs.pop("lead_transition_dynamic_max_step", 4)
    )
    dynamic_transition_entropy = lead_transition_dynamic_entropy_window > 0
    lead_transition_semantic_adaptive = bool(
        kwargs.pop("lead_transition_semantic_adaptive", False)
    )
    lead_transition_semantic_entropy_threshold = float(
        kwargs.pop("lead_transition_semantic_entropy_threshold", 0.82)
    )
    lead_transition_semantic_max_extra_steps = int(
        kwargs.pop("lead_transition_semantic_max_extra_steps", 1)
    )
    if dynamic_transition_entropy and not early_transition_enabled:
        raise ValueError(
            "Dynamic entropy handoff requires an initial-transition mode"
        )
    if dynamic_transition_entropy and (
        lead_initial_transition_delay_steps > 0
        or lead_force_initial_transition_step1
    ):
        raise ValueError(
            "Dynamic entropy handoff cannot be combined with forced or delayed transition"
        )
    if dynamic_transition_entropy and (
        not 0.0 < lead_transition_dynamic_entropy_ratio <= 1.0
        or lead_transition_dynamic_min_history < 1
        or lead_transition_dynamic_max_step < 1
    ):
        raise ValueError("Invalid dynamic entropy handoff configuration")
    if lead_transition_semantic_adaptive and not early_transition_enabled:
        raise ValueError(
            "Semantic-adaptive handoff requires an initial-transition mode"
        )
    if lead_transition_semantic_adaptive and (
        dynamic_transition_entropy
        or lead_initial_transition_delay_steps > 0
        or lead_force_initial_transition_step1
    ):
        raise ValueError(
            "Semantic-adaptive handoff cannot be combined with other timing controls"
        )
    if lead_transition_semantic_adaptive and (
        lead_transition_semantic_entropy_threshold < 0.0
        or lead_transition_semantic_max_extra_steps < 0
    ):
        raise ValueError("Invalid semantic-adaptive handoff configuration")
    if lead_force_initial_transition_step1 and lead_initial_transition_delay_steps > 0:
        raise ValueError("Forced step-1 transition cannot be combined with delayed transition")
    lead_disable_step0_linebreak_mix = kwargs.pop("lead_disable_step0_linebreak_mix", False)
    lead_disable_step0_soft = bool(kwargs.pop("lead_disable_step0_soft", False))
    lead_disable_to_normal_transition = kwargs.pop("lead_disable_to_normal_transition", False)
    lead_soft_quota_ratio = float(kwargs.pop("lead_soft_quota_ratio", 0.0) or 0.0)
    lead_refinement_window = int(kwargs.pop("lead_refinement_window", 0) or 0)
    lead_refinement_start_offset = int(kwargs.pop("lead_refinement_start_offset", 0))
    if lead_refinement_start_offset < 0:
        raise ValueError("refinement start offset must be non-negative")
    lead_refinement_soft_cap = int(kwargs.pop("lead_refinement_soft_cap", 0) or 0)
    lead_refinement_entropy_threshold = float(
        kwargs.pop("lead_refinement_entropy_threshold", b1)
    )
    lead_refinement_threshold_mode = str(
        kwargs.pop("lead_refinement_threshold_mode", "fixed")
    )
    lead_refinement_online_quantile = float(
        kwargs.pop("lead_refinement_online_quantile", 0.75)
    )
    lead_refinement_online_jump_multiplier = float(
        kwargs.pop("lead_refinement_online_jump_multiplier", 1.0)
    )
    lead_refinement_reference_ratio = float(
        kwargs.pop("lead_refinement_reference_ratio", 1.0)
    )
    lead_refinement_soft_mix_lambda = float(
        kwargs.pop("lead_refinement_soft_mix_lambda", 1.0)
    )
    lead_guard_candidate_only = bool(kwargs.pop("lead_guard_candidate_only", False))
    lead_disable_answer_zone_lock = bool(
        kwargs.pop("lead_disable_answer_zone_lock", False)
    )
    if lead_refinement_window < 0 or lead_refinement_soft_cap < 0:
        raise ValueError("TALR refinement window and soft cap must be non-negative")
    if lead_refinement_reference_ratio < 0.0:
        raise ValueError("TALR refinement reference ratio must be non-negative")
    if lead_refinement_threshold_mode not in {
        "fixed", "online_quantile", "online_jump"
    }:
        raise ValueError(
            "TALR refinement threshold mode must be fixed, online_quantile, "
            "or online_jump"
        )
    if not 0.0 <= lead_refinement_online_quantile <= 1.0:
        raise ValueError("TALR online refinement quantile must be in [0, 1]")
    if lead_refinement_online_jump_multiplier <= 0.0:
        raise ValueError("TALR online jump multiplier must be positive")
    if not 0.0 <= lead_refinement_soft_mix_lambda <= 1.0:
        raise ValueError("TALR refinement soft mix lambda must be in [0, 1]")
    lead_windowed_refinement = (
        lead_refinement_window > 0 and lead_refinement_soft_cap > 0
    )
    if lead_windowed_refinement and not lead_initial_transition_with_refinement:
        raise ValueError(
            "Windowed refinement requires lead_initial_transition_with_refinement"
        )
    if lead_windowed_refinement and lead_soft_quota_ratio > 0.0:
        raise ValueError(
            "Windowed strict-cap refinement cannot be combined with quota catch-up"
        )
    lead_format_cooldown = kwargs.pop("lead_format_cooldown", False)
    format_cooldown_steps = kwargs.pop("format_cooldown_steps", 0)
    format_cooldown_min_step = kwargs.pop("format_cooldown_min_step", 0)
    format_cooldown_highrisk_only = kwargs.pop("format_cooldown_highrisk_only", False)
    format_cooldown_normal_steps = kwargs.pop("format_cooldown_normal_steps", None)
    format_cooldown_highrisk_steps = kwargs.pop("format_cooldown_highrisk_steps", None)
    format_cooldown_max_active = kwargs.pop("format_cooldown_max_active", 0)
    format_cooldown_entropy_min = kwargs.pop("format_cooldown_entropy_min", None)
    format_cooldown_top1_max = kwargs.pop("format_cooldown_top1_max", None)
    format_cooldown_margin_max = kwargs.pop("format_cooldown_margin_max", None)

    # ============================================

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    if (
        lead_initial_transition_cache_rebuild_prefix_len >= 0
        or trace_route_cache_rebuild_after_tokens > 0
    ) and input_ids.shape[0] != 1:
        raise ValueError("Cache rebuild currently supports batch_size=1 only")

    prompt_len = input_ids.shape[1]
    prompt_input_ids = input_ids.clone()
    prompt_attention_mask = attention_mask.clone()
    prompt_vision_inputs = dict(vision_inputs)
    visual_token_mask = _build_visual_token_mask(input_ids, tokenizer)
    prompt_hidden_states = None
    batch_size, device = input_ids.shape[0], input_ids.device
    E = model.get_input_embeddings().weight  # [vocab_size, dim]
    def _resolve_token_id(token_text, fallback_text=None):
        token_id = None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
        except Exception:
            token_id = None
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if token_id is None or token_id == tokenizer.unk_token_id or (isinstance(token_id, int) and token_id < 0):
            text = fallback_text if fallback_text is not None else token_text
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                token_id = encoded[0]
        if token_id is None:
            token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        return token_id
    if thinking_token_id is None:
        thinking_token_id = _resolve_token_id("<think>")
    if end_thinking_token_id is None:
        end_thinking_token_id = _resolve_token_id("</think>")
    reasoning_start_token_id = _resolve_token_id("<think>")
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if lead_transition_anchor == "im_end" and (
        im_end_token_id is None
        or im_end_token_id == tokenizer.unk_token_id
        or int(im_end_token_id) < 0
    ):
        raise ValueError("<|im_end|> is unavailable as a single tokenizer token")

    if not lead_disable_simple_visual_anchor:
        # 原始 LEAD：把 <think> anchor 替换成 <|image_pad|> embedding。
        imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        thinking_token_id = imgpad_id

    start_thinking_emb, end_thinking_emb = E[thinking_token_id], E[end_thinking_token_id]
    newline_id = _resolve_token_id("\\n", "\n")
    line_break_emb = E[newline_id]
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(lead_transition_random_seed)
    random_direction = torch.randn(
        E.shape[-1], generator=random_generator, device=device, dtype=torch.float32
    )
    random_direction = random_direction / torch.linalg.vector_norm(random_direction).clamp_min(1e-8)
    past_key_values = None
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    prefetched_outputs = None
        
    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    prompt_lens = [len(ids) for ids in all_generated]
    unfinished_idx = list(range(batch_size)) # bs >= 1 is supported
    delay_initial_transition = lead_initial_transition_only and lead_initial_transition_delay_steps > 0
    start_normal = lead_force_normal or delay_initial_transition
    mode = torch.ones(batch_size, dtype=torch.long, device=device) if start_normal else torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: soft, 1: normal
    mode_stay_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    locked_normal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    entropy_history = [[] for _ in range(batch_size)]
    refinement_entropy_histories = [[] for _ in range(batch_size)]
    lead_soft_quota_counts = [0 for _ in range(batch_size)]
    lead_refinement_counts = [0 for _ in range(batch_size)]
    lead_initial_transition_steps = [-1 for _ in range(batch_size)]
    lead_reasoning_boundary_steps = [-1 for _ in range(batch_size)]
    format_cooldowns = [0 for _ in range(batch_size)]
    format_cooldown_active_counts = [0 for _ in range(batch_size)]
    
    if max_switch_count is not None:
        switch_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        convergence_ids = tokenizer.encode(convergence_words, add_special_tokens=False)
        termination_ids = tokenizer.encode(termination_words, add_special_tokens=False)
        injecting = torch.zeros(batch_size, dtype=torch.bool, device=device)
        inject_queues = [[] for _ in range(batch_size)]
        answer_budget = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        cur_batch = attention_mask.shape[0]
        if cur_batch == 0:
            break

        using_prefetched_outputs = prefetched_outputs is not None
        if using_prefetched_outputs:
            outputs = prefetched_outputs
            prefetched_outputs = None
        elif past_key_values is None:
            model_inputs = {
                "input_ids": input_ids.clone(), 
            }
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if vision_inputs:
                model_inputs.update(vision_inputs)
            model_inputs["cache_position"] = cache_position
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "inputs_embeds": last_emb.unsqueeze(1), 
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            model_inputs["cache_position"] = cache_position

        need_prompt_hidden = (
            trace_event_geometry or lead_early_visual_anchor
        ) and past_key_values is None
        if not using_prefetched_outputs:
            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    use_cache=True,
                    output_hidden_states=need_prompt_hidden,
                    return_dict=True,
                )
        past_key_values = outputs.past_key_values
        if need_prompt_hidden and getattr(outputs, "hidden_states", None):
            prompt_hidden_states = outputs.hidden_states[-1][:, :prompt_len, :].detach()
        if vision_inputs:
            vision_inputs = {}
        cache_position = cache_position[-1:] + 1
        
        logits_original = outputs.logits[:, -1, :]
        probs_original = F.softmax(logits_original, dim=-1)
        if capture_logits_sink is not None and step in capture_logits_steps:
            for bi, orig in enumerate(unfinished_idx):
                capture_logits_sink.append({
                    "kind": "logits",
                    "step": int(step),
                    "sample_index": int(orig),
                    "probs": probs_original[bi].detach().float().cpu(),
                })
        logits = logits_original / temperature  
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)  # [B, N, V]
        probs = F.softmax(logits_filtered, dim=-1)
        filtered_entropy = -(
            probs * probs.clamp(min=1e-8).log()
        ).sum(dim=-1)

        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)  # [B, N]
        if forced_prefix_ids is not None and step < len(forced_prefix_ids):
            next_tokens = torch.full(
                (cur_batch,),
                int(forced_prefix_ids[step]),
                dtype=torch.long,
                device=device,
            )
        initial_transition_cache_rebuild = bool(
            lead_initial_transition_cache_rebuild_prefix_len == step + 1
        )
        route_cache_rebuild = bool(
            trace_route_cache_rebuild_after_tokens > 0
            and step
            == trace_route_override_step + trace_route_cache_rebuild_after_tokens
        )
        cache_rebuild_after_this_step = bool(
            initial_transition_cache_rebuild or route_cache_rebuild
        )
        cache_rebuild_kind = (
            "route_override"
            if route_cache_rebuild
            else "initial_transition"
            if initial_transition_cache_rebuild
            else None
        )
        if not lead_disable_answer_zone_lock:
            locked_normal_mask = (
                locked_normal_mask | (next_tokens == end_thinking_token_id)
            )

        if max_switch_count is not None and injecting.any():
            mask_list = [injecting[i].item() and len(inject_queues[i]) > 0 for i in range(cur_batch)]
            force_mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
            if force_mask.any():
                force_toks = torch.tensor([inject_queues[i].pop(0) for i in range(cur_batch) if mask_list[i]], \
                                          device=device, dtype=torch.long)
                next_tokens[force_mask] = force_toks
            if injecting.any():
                done_mask = torch.tensor([injecting[i] and (len(inject_queues[i]) == 0) for i in range(cur_batch)], \
                                         device=device, dtype=torch.bool)
                injecting[done_mask] = False
        
        cur_entropy = -(probs_original * (probs_original.clamp(min=1e-8).log())).sum(dim=-1)
        mode_before = mode.clone()
        to_normal = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        to_soft = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        entropy_refinement_proposal = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        refinement_dynamic_threshold = torch.full(
            (cur_batch,), float("nan"), dtype=cur_entropy.dtype, device=device
        )
        refinement_previous_entropy = torch.full(
            (cur_batch,), float("nan"), dtype=cur_entropy.dtype, device=device
        )
        refinement_local_scale = torch.full(
            (cur_batch,), float("nan"), dtype=cur_entropy.dtype, device=device
        )
        refinement_history_length = torch.zeros(
            cur_batch, dtype=torch.long, device=device
        )
        dynamic_transition_condition = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        dynamic_transition_deadline = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        dynamic_transition_reference = torch.full(
            (cur_batch,), float("nan"), dtype=cur_entropy.dtype, device=device
        )
        semantic_transition_condition = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        semantic_transition_deadline = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        semantic_transition_offset = torch.full(
            (cur_batch,), -1, dtype=torch.long, device=device
        )
        forced_transition_step1 = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        delayed_transition_entry = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        delayed_transition_exit = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if delay_initial_transition:
            delayed_transition_entry = (
                torch.full((cur_batch,), step == lead_initial_transition_delay_steps, dtype=torch.bool, device=device)
                & (~locked_normal_mask)
            )
            delayed_transition_exit = (
                torch.full((cur_batch,), step == lead_initial_transition_delay_steps + 1, dtype=torch.bool, device=device)
                & (~locked_normal_mask)
            )
        if step == 0:
            cur_ref_entropy = cur_entropy.clone()
        else:
            mode_stay_steps += 1
            allow_switch = (mode_stay_steps >= window_size)
            to_normal = (mode == 0) & (cur_entropy < cur_ref_entropy) & (cur_entropy < b2)
            if dynamic_transition_entropy:
                to_normal = torch.zeros_like(to_normal)
                for bi in range(cur_batch):
                    history = entropy_history[bi]
                    if len(history) >= lead_transition_dynamic_min_history:
                        recent = history[-lead_transition_dynamic_entropy_window:]
                        rolling_reference = sum(recent) / len(recent)
                        dynamic_transition_reference[bi] = rolling_reference
                        dynamic_transition_condition[bi] = (
                            mode[bi] == 0
                            and cur_entropy[bi]
                            <= float(lead_transition_dynamic_entropy_ratio)
                            * rolling_reference
                        )
                    dynamic_transition_deadline[bi] = (
                        mode[bi] == 0
                        and step >= lead_transition_dynamic_max_step
                    )
                to_normal = (
                    dynamic_transition_condition | dynamic_transition_deadline
                )
            if lead_transition_semantic_adaptive:
                to_normal = torch.zeros_like(to_normal)
                for bi, orig in enumerate(unfinished_idx):
                    generated_prefix = (
                        all_generated[orig][prompt_lens[orig]:]
                        + [int(next_tokens[bi].item())]
                    )
                    prefix_text = tokenizer.decode(
                        generated_prefix,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if (
                        lead_reasoning_boundary_steps[orig] < 0
                        and "<think>" in prefix_text
                        and prefix_text.rstrip(" \t").endswith("\n")
                    ):
                        lead_reasoning_boundary_steps[orig] = int(step)
                    boundary_step = lead_reasoning_boundary_steps[orig]
                    if boundary_step < 0 or mode[bi] != 0:
                        continue
                    offset = int(step) - int(boundary_step)
                    semantic_transition_offset[bi] = offset
                    if offset == 1:
                        semantic_transition_condition[bi] = (
                            cur_entropy[bi]
                            <= lead_transition_semantic_entropy_threshold
                        )
                    semantic_transition_deadline[bi] = (
                        offset
                        >= 1 + lead_transition_semantic_max_extra_steps
                    )
                to_normal = (
                    semantic_transition_condition | semantic_transition_deadline
                )
            if lead_refinement_threshold_mode in {"online_quantile", "online_jump"}:
                for bi, orig in enumerate(unfinished_idx):
                    history = refinement_entropy_histories[orig]
                    refinement_history_length[bi] = len(history)
                    if not history:
                        continue
                    if lead_refinement_threshold_mode == "online_quantile":
                        threshold = nearest_rank_quantile(
                            history, lead_refinement_online_quantile
                        )
                    else:
                        if len(history) < 2:
                            continue
                        local_deltas = [
                            abs(history[index] - history[index - 1])
                            for index in range(1, len(history))
                        ]
                        local_scale = max(
                            nearest_rank_quantile(local_deltas, 0.5), 1e-6
                        )
                        refinement_previous_entropy[bi] = history[-1]
                        refinement_local_scale[bi] = local_scale
                        threshold = history[-1] + (
                            lead_refinement_online_jump_multiplier * local_scale
                        )
                    refinement_dynamic_threshold[bi] = threshold
                    entropy_refinement_proposal[bi] = bool(
                        mode[bi] == 1
                        and not bool(locked_normal_mask[bi].item())
                        and float(cur_entropy[bi].item()) > threshold
                        and (
                            lead_refinement_threshold_mode != "online_jump"
                            or float(cur_entropy[bi].item())
                            > lead_refinement_reference_ratio
                            * float(cur_ref_entropy[bi].item())
                        )
                    )
            else:
                entropy_refinement_proposal = (
                    (mode == 1)
                    & (cur_entropy > lead_refinement_reference_ratio * cur_ref_entropy)
                    & (~locked_normal_mask)
                    & (cur_entropy > lead_refinement_entropy_threshold)
                )
            to_soft = entropy_refinement_proposal & allow_switch
            if (
                lead_force_normal
                or lead_initial_soft_only
                or lead_initial_transition_only
                or lead_initial_transition_with_refinement
            ):
                to_soft = torch.zeros_like(to_soft)
            if delay_initial_transition:
                to_normal = to_normal | delayed_transition_exit
            if lead_force_initial_transition_step1:
                forced_transition_step1 = (
                    torch.full((cur_batch,), step == 1, dtype=torch.bool, device=device)
                    & (mode == 0)
                    & (~locked_normal_mask)
                )
                to_normal = to_normal | forced_transition_step1

            
            if (to_normal.any() or to_soft.any()) and step % 50 == 0:  # 每50步打印一次
                print(f"[LEAD] Step {step}: to_normal={to_normal.nonzero().squeeze(-1).tolist()}, to_soft={to_soft.nonzero().squeeze(-1).tolist()}")
            
            mode[to_normal] = 1
            mode[to_soft] = 0
            mode[delayed_transition_entry] = 0
            mode_stay_steps[to_normal | to_soft] = 0
            mode_stay_steps[delayed_transition_entry] = 0
            cur_ref_entropy[to_normal | to_soft] = cur_entropy[to_normal | to_soft]
            cur_ref_entropy[delayed_transition_entry] = cur_entropy[delayed_transition_entry]
            for bi, orig in enumerate(unfinished_idx):
                if (
                    bool(to_normal[bi].item())
                    and lead_initial_transition_steps[orig] < 0
                ):
                    lead_initial_transition_steps[orig] = int(step)
                    if lead_refinement_threshold_mode in {
                        "online_quantile", "online_jump"
                    }:
                        refinement_entropy_histories[orig] = [
                            float(cur_entropy[bi].item())
                        ]

            #输出表明print
            if to_normal.any() or to_soft.any():
                print(f"[LEAD] Step {step}: switch → "
                    f"{'to_normal' if to_normal.any() else 'to_soft'} | "
                    f"entropy={cur_entropy.mean().item():.4f}")


            if max_switch_count is not None:
                switch_count = switch_count + to_normal.long() 
            
        is_normal = (mode == 1) | locked_normal_mask
        if math_ids_tensor is not None:
            is_math_token = (next_tokens.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
            is_normal[is_math_token] = True
        is_soft = ~is_normal
        if lead_force_normal:
            is_soft = torch.zeros_like(is_soft)
        elif lead_initial_soft_only:
            is_soft = is_soft & (step == 0)
        elif lead_initial_transition_only:
            if delay_initial_transition:
                is_soft = delayed_transition_entry
            else:
                is_soft = is_soft & (step == 0)
        
        normal_emb = E[next_tokens]
        soft_emb = torch.matmul(probs_original, E)
        raw_soft_emb = soft_emb
        if (
            trace_soft_vector_collector is not None
            and step == trace_capture_soft_vector_step
        ):
            trace_soft_vector_collector.append(
                {
                    "step": int(step),
                    "soft_embedding": raw_soft_emb[0].detach().float().cpu(),
                    "hard_embedding": normal_emb[0].detach().float().cpu(),
                    "raw_entropy": float(cur_entropy[0].item()),
                    "selected_token_id": int(next_tokens[0].item()),
                    "selected_token_prob": float(
                        probs_original[0, next_tokens[0]].item()
                    ),
                }
            )
        if capture_logits_sink is not None and step == 0:
            soft_hard_cos = F.cosine_similarity(
                raw_soft_emb.float(), normal_emb.float(), dim=-1
            )
            for bi, orig in enumerate(unfinished_idx):
                capture_logits_sink.append({
                    "kind": "step0_geometry",
                    "step": 0,
                    "sample_index": int(orig),
                    "entropy": float(cur_entropy[bi].item()),
                    "top1_probability": float(probs_original[bi].max().item()),
                    "soft_hard_distance": float(1.0 - soft_hard_cos[bi].item()),
                })
        early_visual_anchor_applied = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        early_visual_anchor_query_similarity = torch.zeros(
            cur_batch, dtype=soft_emb.dtype, device=device
        )
        early_visual_anchor_norm_ratio = torch.ones(
            cur_batch, dtype=soft_emb.dtype, device=device
        )
        raw_top2 = torch.topk(probs_original, k=min(2, probs_original.shape[-1]), dim=-1).values
        raw_top1_prob = raw_top2[:, 0]
        raw_margin = (
            raw_top2[:, 0] - raw_top2[:, 1]
            if raw_top2.shape[-1] > 1
            else raw_top2[:, 0]
        )
        lead_soft_veto_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        lead_veto_candidate = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        lead_soft_quota_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        lead_refinement_candidate = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        lead_refinement_mask = torch.zeros(
            cur_batch, dtype=torch.bool, device=device
        )
        lead_refinement_elapsed = torch.full(
            (cur_batch,), -1, dtype=torch.long, device=device
        )
        lead_format_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        lead_format_token_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        lead_highrisk_format_token_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if lead_windowed_refinement:
            for bi, orig in enumerate(unfinished_idx):
                transition_step = lead_initial_transition_steps[orig]
                if transition_step < 0:
                    continue
                elapsed = int(step) - int(transition_step)
                lead_refinement_elapsed[bi] = elapsed
                proposed = bool(entropy_refinement_proposal[bi].item())
                if _talr_refinement_eligible(
                    step=step,
                    transition_step=transition_step,
                    refinement_count=lead_refinement_counts[orig],
                    window=lead_refinement_window,
                    soft_cap=lead_refinement_soft_cap,
                    entropy_proposed=proposed,
                    locked_normal=bool(locked_normal_mask[bi].item()),
                    start_offset=lead_refinement_start_offset,
                ):
                    lead_refinement_candidate[bi] = True
                    lead_refinement_mask[bi] = True
        if lead_soft_veto_on_diffuse:
            spike_mask, _, _ = _entropy_spike_mask(
                raw_entropy=cur_entropy,
                entropy_history=entropy_history,
                window=lead_veto_entropy_window,
                alpha=lead_veto_entropy_alpha,
                min_history=lead_veto_min_history,
                min_entropy=lead_veto_min_entropy,
            )
            diffuse_mask = (
                (raw_top1_prob < float(lead_veto_low_conf_tau))
                | (raw_margin < float(lead_veto_low_margin_tau))
            )
            lead_veto_candidate = spike_mask & diffuse_mask & (step >= int(lead_veto_min_step))
            if lead_guard_candidate_only:
                lead_veto_candidate = (
                    lead_veto_candidate & lead_refinement_candidate
                )
            for bi, orig in enumerate(unfinished_idx):
                if not bool(lead_veto_candidate[bi].item()):
                    continue
                generated_only = all_generated[orig][prompt_lens[orig]:]
                repeat_degen = False
                ngram = int(lead_veto_repeat_ngram)
                if ngram > 0 and len(generated_only) >= ngram * 2:
                    last_ngram = tuple(generated_only[-ngram:])
                    prior = generated_only[:-ngram]
                    repeat_degen = any(
                        tuple(prior[i:i + ngram]) == last_ngram
                        for i in range(0, len(prior) - ngram + 1)
                    )
                repeat_tau = float(lead_veto_recent_repeat_tau)
                if repeat_tau > 0.0 and generated_only:
                    window = max(1, int(lead_veto_recent_repeat_window))
                    recent = generated_only[-window:]
                    duplicate_ratio = 1.0 - (len(set(recent)) / max(1, len(recent)))
                    repeat_degen = repeat_degen or (duplicate_ratio >= repeat_tau)
                if lead_veto_require_repeat_degen and not repeat_degen:
                    continue
                lead_soft_veto_mask[bi] = True

        if lead_format_cooldown and int(format_cooldown_steps) > 0 and step >= int(format_cooldown_min_step):
            for bi, orig in enumerate(unfinished_idx):
                token_text = tokenizer.decode(
                    [int(next_tokens[bi].item())],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                is_highrisk = _is_high_risk_format_token_text(token_text)
                is_format = is_highrisk if format_cooldown_highrisk_only else _is_format_token_text(token_text)
                lead_highrisk_format_token_mask[bi] = is_highrisk
                if is_format and (
                    format_cooldown_entropy_min is not None
                    or format_cooldown_top1_max is not None
                    or format_cooldown_margin_max is not None
                ):
                    unstable = False
                    if format_cooldown_entropy_min is not None:
                        unstable = unstable or (
                            float(cur_entropy[bi].item()) >= float(format_cooldown_entropy_min)
                        )
                    if format_cooldown_top1_max is not None:
                        unstable = unstable or (
                            float(raw_top1_prob[bi].item()) <= float(format_cooldown_top1_max)
                        )
                    if format_cooldown_margin_max is not None:
                        unstable = unstable or (
                            float(raw_margin[bi].item()) <= float(format_cooldown_margin_max)
                        )
                    is_format = unstable
                lead_format_token_mask[bi] = is_format
                lead_format_mask[bi] = is_format or format_cooldowns[orig] > 0
                if (
                    lead_guard_candidate_only
                    and not bool(lead_refinement_candidate[bi].item())
                ):
                    lead_format_mask[bi] = False
                if int(format_cooldown_max_active) > 0 and format_cooldown_active_counts[orig] >= int(format_cooldown_max_active):
                    lead_format_mask[bi] = False

        alpha = alpha_0 + (1 - alpha_0) * float(step) / float(max_new_tokens)
        if step == 0:
            if lead_initial_transition_hard_boundary_only or lead_disable_step0_soft:
                soft_emb = normal_emb
            if not lead_disable_step0_linebreak_mix:
                soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb
            if lead_early_visual_anchor:
                if lead_early_visual_anchor_source == "visual_hidden":
                    if prompt_hidden_states is None:
                        raise RuntimeError("Early visual anchor requires prompt hidden states")
                    (
                        early_anchor,
                        early_visual_anchor_applied,
                        early_visual_anchor_query_similarity,
                        early_visual_anchor_norm_ratio,
                    ) = _compute_early_actual_visual_anchor(
                        prompt_hidden_states=prompt_hidden_states,
                        visual_token_mask=visual_token_mask,
                        query_state=prompt_hidden_states[:, prompt_len - 1, :],
                        reference_emb=soft_emb,
                        top_m=lead_early_visual_anchor_top_m,
                        temperature=lead_early_visual_anchor_temperature,
                    )
                else:
                    imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    static_anchor = E[imgpad_id].unsqueeze(0).expand_as(soft_emb)
                    static_norm = torch.linalg.vector_norm(
                        static_anchor.float(), dim=-1
                    ).clamp_min(1e-8)
                    reference_norm = torch.linalg.vector_norm(
                        soft_emb.float(), dim=-1
                    )
                    early_visual_anchor_norm_ratio = (
                        reference_norm / static_norm
                    ).to(soft_emb.dtype)
                    early_anchor = static_anchor * early_visual_anchor_norm_ratio[:, None]
                    early_visual_anchor_applied = torch.ones(
                        cur_batch, dtype=torch.bool, device=device
                    )
                    if prompt_hidden_states is not None:
                        early_visual_anchor_query_similarity = F.cosine_similarity(
                            prompt_hidden_states[:, prompt_len - 1, :].float(),
                            static_anchor.float(),
                            dim=-1,
                        ).to(soft_emb.dtype)
                weight = lead_early_visual_anchor_lambda
                soft_emb = torch.where(
                    early_visual_anchor_applied[:, None],
                    (1.0 - weight) * soft_emb + weight * early_anchor,
                    soft_emb,
                )
        else:
            mixed_emb = alpha * soft_emb + a * (1 - alpha) * start_thinking_emb
            
            soft_emb = torch.where(to_soft[:, None], mixed_emb, soft_emb)
        beta = beta_0 + (1 - beta_0) * float(step) / float(max_new_tokens)

        if step % 200 == 0:
            print(f"[LEAD] Step {step}: alpha={alpha:.3f}, beta={beta:.3f}, soft_ratio={(mode==0).float().mean():.2f}")


        transition_source_emb = normal_emb if lead_transition_source == "hard" else soft_emb
        if lead_transition_anchor == "generated_token":
            transition_anchor_emb = normal_emb
            transition_anchor_token_id = None
        elif lead_transition_anchor == "start_thinking":
            transition_anchor_emb = E[reasoning_start_token_id].unsqueeze(0).expand_as(normal_emb)
            transition_anchor_token_id = int(reasoning_start_token_id)
        elif lead_transition_anchor == "newline":
            transition_anchor_emb = line_break_emb.unsqueeze(0).expand_as(normal_emb)
            transition_anchor_token_id = int(newline_id)
        elif lead_transition_anchor == "im_end":
            transition_anchor_emb = E[im_end_token_id].unsqueeze(0).expand_as(normal_emb)
            transition_anchor_token_id = int(im_end_token_id)
        elif lead_transition_anchor == "random_residual":
            eot_residual_norm = torch.linalg.vector_norm(
                (end_thinking_emb.unsqueeze(0) - normal_emb).float(), dim=-1, keepdim=True
            )
            transition_anchor_emb = normal_emb + random_direction.to(normal_emb.dtype).unsqueeze(0) * eot_residual_norm.to(normal_emb.dtype)
            transition_anchor_token_id = None
        else:
            transition_anchor_emb = end_thinking_emb.unsqueeze(0).expand_as(normal_emb)
            transition_anchor_token_id = int(end_thinking_token_id)
        if lead_initial_transition_hard_boundary_only:
            transition_source_emb = normal_emb
        transition_bridge_raw = (
            beta * transition_source_emb + (1 - beta) * transition_anchor_emb
        )
        if lead_transition_norm_match:
            transition_bridge_emb = transition_bridge_raw * (
                torch.linalg.vector_norm(normal_emb.float(), dim=-1, keepdim=True)
                / torch.linalg.vector_norm(transition_bridge_raw.float(), dim=-1, keepdim=True).clamp_min(1e-8)
            ).to(transition_bridge_raw.dtype)
        else:
            transition_bridge_emb = transition_bridge_raw
        transition_source_norm = torch.linalg.vector_norm(
            transition_source_emb.float(), dim=-1
        ).reshape(-1)
        transition_anchor_norm = torch.linalg.vector_norm(
            transition_anchor_emb.float(), dim=-1
        ).reshape(-1)
        transition_bridge_norm = torch.linalg.vector_norm(
            transition_bridge_emb.float(), dim=-1
        ).reshape(-1)
        transition_source_anchor_cos = F.cosine_similarity(
            transition_source_emb.float(), transition_anchor_emb.float(), dim=-1
        ).reshape(-1)

        if step > 0 and not lead_initial_soft_only and not lead_disable_to_normal_transition:
            mixed_emb = transition_bridge_emb
            normal_emb = torch.where(to_normal[:, None], mixed_emb, normal_emb)
        if (
            lead_soft_quota_ratio > 0.0
            and not lead_force_normal
            and not lead_initial_soft_only
            and not lead_initial_transition_only
        ):
            for bi, orig in enumerate(unfinished_idx):
                target_count = int(math.ceil(float(lead_soft_quota_ratio) * float(step + 1)))
                if lead_soft_quota_counts[orig] < target_count and not bool(locked_normal_mask[bi].item()):
                    lead_soft_quota_mask[bi] = True
            is_soft = is_soft | lead_soft_quota_mask
        is_soft = is_soft | lead_refinement_mask
        is_soft = is_soft & (~lead_soft_veto_mask) & (~lead_format_mask)
        refinement_emb = (
            lead_refinement_soft_mix_lambda * soft_emb
            + (1.0 - lead_refinement_soft_mix_lambda) * normal_emb
        )
        routed_soft_emb = torch.where(
            lead_refinement_mask[:, None], refinement_emb, soft_emb
        )
        last_emb = torch.where(is_soft[:, None], routed_soft_emb, normal_emb)
        # Execution-only optimization: obtain an action while preserving this live KV.
        if trace_online_residual_provider is not None:
            assert cur_batch == 1
            online_request = trace_online_residual_provider(
                step=int(step), selected_token_id=int(next_tokens[0].item()),
                raw_soft=raw_soft_emb[0], hard=normal_emb[0],
                transition_step=int(lead_initial_transition_steps[0]),
                entropy_proposed=bool(entropy_refinement_proposal[0].item()),
                locked_normal=bool(locked_normal_mask[0].item()),
                controller_mode=int(mode[0].item()),
            )
            if online_request is not None:
                trace_route_override_step = int(online_request["event"])
                trace_route_override_kind = "method_plus_external_residual"
                trace_route_override_mix_lambda = 1.0
                trace_route_override_duration = int(online_request.get("duration", 4))
                trace_external_route_vector = online_request["vector"]
        route_override_active = (
            trace_route_override_step <= step
            < trace_route_override_step + trace_route_override_duration
            and trace_route_override_kind != "none"
        )
        if route_override_active:
            if trace_route_override_kind == "hard":
                last_emb = E[next_tokens]
            elif trace_route_override_kind == "raw_soft":
                last_emb = raw_soft_emb
            elif trace_route_override_kind == "method_soft":
                last_emb = soft_emb
            elif trace_route_override_kind in {
                "external_residual",
                "method_plus_external_residual",
            }:
                if trace_external_route_vector is None:
                    raise ValueError(
                        "external_residual requires trace_external_route_vector"
                    )
                external_emb = torch.as_tensor(
                    trace_external_route_vector,
                    device=device,
                    dtype=E.dtype,
                )
                if external_emb.ndim == 1:
                    external_emb = external_emb.unsqueeze(0)
                if external_emb.shape != last_emb.shape:
                    if external_emb.shape[0] == 1 and last_emb.shape[0] > 1:
                        external_emb = external_emb.expand(last_emb.shape[0], -1)
                    else:
                        raise ValueError(
                            "External residual shape mismatch: "
                            f"{tuple(external_emb.shape)} vs {tuple(last_emb.shape)}"
                        )
                route_origin = (
                    last_emb
                    if trace_route_override_kind == "method_plus_external_residual"
                    else E[next_tokens]
                )
                last_emb = route_origin + trace_route_override_mix_lambda * external_emb
            else:
                raise ValueError(f"Unsupported trace_route_override_kind={trace_route_override_kind}")
        forced_answer_probe = None
        if trace_forced_answer_probe and step == trace_route_override_step:
            forced_answer_probe = _forced_answer_probe_from_route(
                model, tokenizer, last_emb, attention_mask, past_key_values,
                cache_position, trace_probe_gold_choice,
                choice_case=trace_probe_choice_case,
                prompt_hidden_states=prompt_hidden_states,
                visual_token_mask=visual_token_mask,
                prompt_len=prompt_len,
            )

        if max_switch_count is not None and step > 0:
            trigger = (switch_count >= max_switch_count) & (switch_count <= 2 * max_switch_count) & to_normal
            
            if trigger.any():
                print(f"[LEAD] Inject convergence at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")

            
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(convergence_ids)
                injecting = injecting | trigger

            trigger = (switch_count > 2 * max_switch_count) & to_normal

            if trigger.any():
                print(f"[LEAD] Inject termination at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")


            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(termination_ids) 
                injecting = injecting | trigger 
                answer_budget[trigger] = termination_max_tokens
            active = (answer_budget >= 0)
            if active.any():
                answer_budget = torch.where(active, answer_budget - 1, answer_budget)

        for bi, orig in enumerate(unfinished_idx):
            token_id = next_tokens[bi].item()
            all_generated[orig].append(token_id)
            if token_trace is not None:
                phase = (
                    "early"
                    if step <= 128
                    else "mid" if step <= 512 else "late"
                )
                veto_active = bool(lead_soft_veto_mask[bi].item())
                format_active = bool(lead_format_mask[bi].item())
                if veto_active:
                    route_signal = "diffuse_repeat_degen"
                    route_action = "hard_discrete"
                    route_priority = 90
                elif format_active:
                    route_signal = (
                        "highrisk_format_uncertain"
                        if bool(lead_highrisk_format_token_mask[bi].item())
                        else "format_uncertain"
                    )
                    route_action = "format_cooldown"
                    route_priority = 80
                elif bool(is_soft[bi].item()):
                    if bool(lead_refinement_mask[bi].item()):
                        route_signal = "talr_windowed_refinement"
                        route_action = "talr_windowed_refinement"
                        route_priority = 30
                    else:
                        route_signal = "lead_soft_quota" if bool(lead_soft_quota_mask[bi].item()) else "lead_soft"
                        route_action = "lead_soft_quota" if bool(lead_soft_quota_mask[bi].item()) else "lead_soft"
                        route_priority = 20 if bool(lead_soft_quota_mask[bi].item()) else 10
                else:
                    route_signal = "lead_normal"
                    route_action = "normal"
                    route_priority = 0
                record = {
                    "step": int(step),
                    "batch_index": int(orig),
                    "token_id": int(token_id),
                    "raw_entropy": float(cur_entropy[bi].item()),
                    "filtered_entropy": float(filtered_entropy[bi].item()),
                        "selected_prob": float(probs[bi, next_tokens[bi]].item()),
                        "mode": "soft" if bool(is_soft[bi].item()) else "normal",
                        "generation_phase": phase,
                        "route_signal": route_signal,
                        "route_action": route_action,
                        "route_priority": int(route_priority),
                        "route_suppressed_by": (
                            (["diffuse_repeat_veto"] if bool(lead_soft_veto_mask[bi].item()) else [])
                            + (["format_cooldown"] if bool(lead_format_mask[bi].item()) else [])
                            + (["locked_normal"] if bool(locked_normal_mask[bi].item()) else [])
                        ),
                        "lead_soft_veto": bool(lead_soft_veto_mask[bi].item()),
                        "lead_veto_candidate": bool(lead_veto_candidate[bi].item()),
                        "lead_disable_step0_linebreak_mix": bool(lead_disable_step0_linebreak_mix),
                        "lead_disable_to_normal_transition": bool(lead_disable_to_normal_transition),
                        "lead_force_initial_transition_step1": bool(
                            lead_force_initial_transition_step1
                        ),
                        "lead_transition_source": lead_transition_source,
                        "lead_transition_anchor": lead_transition_anchor,
                        "lead_transition_anchor_token_id": transition_anchor_token_id,
                        "lead_transition_beta0": float(beta_0),
                        "lead_transition_norm_match": bool(lead_transition_norm_match),
                        "lead_transition_random_seed": int(lead_transition_random_seed),
                        "lead_transition_dynamic_entropy_window": int(
                            lead_transition_dynamic_entropy_window
                        ),
                        "lead_transition_dynamic_entropy_ratio": float(
                            lead_transition_dynamic_entropy_ratio
                        ),
                        "lead_transition_dynamic_min_history": int(
                            lead_transition_dynamic_min_history
                        ),
                        "lead_transition_dynamic_max_step": int(
                            lead_transition_dynamic_max_step
                        ),
                        "lead_transition_semantic_adaptive": bool(
                            lead_transition_semantic_adaptive
                        ),
                        "lead_transition_semantic_entropy_threshold": float(
                            lead_transition_semantic_entropy_threshold
                        ),
                        "lead_transition_semantic_max_extra_steps": int(
                            lead_transition_semantic_max_extra_steps
                        ),
                        "reasoning_boundary_step": int(
                            lead_reasoning_boundary_steps[orig]
                        ),
                        "semantic_transition_offset": int(
                            semantic_transition_offset[bi].item()
                        ),
                        "semantic_transition_condition": bool(
                            semantic_transition_condition[bi].item()
                        ),
                        "semantic_transition_deadline": bool(
                            semantic_transition_deadline[bi].item()
                        ),
                        "dynamic_transition_condition": bool(
                            dynamic_transition_condition[bi].item()
                        ),
                        "dynamic_transition_deadline": bool(
                            dynamic_transition_deadline[bi].item()
                        ),
                        "dynamic_transition_reference": (
                            float(dynamic_transition_reference[bi].item())
                            if not torch.isnan(dynamic_transition_reference[bi])
                            else None
                        ),
                        "forced_transition_step1": bool(forced_transition_step1[bi].item()),
                        "actual_transition_step": int(step) if bool(to_normal[bi].item()) else None,
                        "transition_source_norm": (
                            float(transition_source_norm[bi].item())
                            if bool(to_normal[bi].item()) else None
                        ),
                        "transition_anchor_norm": (
                            float(transition_anchor_norm[bi].item())
                            if bool(to_normal[bi].item()) else None
                        ),
                        "transition_bridge_norm": (
                            float(transition_bridge_norm[bi].item())
                            if bool(to_normal[bi].item()) else None
                        ),
                        "transition_source_anchor_cos": (
                            float(transition_source_anchor_cos[bi].item())
                            if bool(to_normal[bi].item()) else None
                        ),
                        "early_visual_anchor_applied": bool(early_visual_anchor_applied[bi].item()),
                        "early_visual_anchor_source": (
                            lead_early_visual_anchor_source if lead_early_visual_anchor else None
                        ),
                        "early_visual_anchor_top_m": int(lead_early_visual_anchor_top_m),
                        "early_visual_anchor_lambda": float(lead_early_visual_anchor_lambda),
                        "early_visual_anchor_query_similarity": float(
                            early_visual_anchor_query_similarity[bi].item()
                        ),
                        "early_visual_anchor_norm_ratio": float(
                            early_visual_anchor_norm_ratio[bi].item()
                        ),
                        "lead_initial_transition_delay_steps": int(lead_initial_transition_delay_steps),
                        "lead_initial_transition_cache_rebuild_after_step": int(
                            lead_initial_transition_cache_rebuild_after_step
                        ),
                        "lead_initial_transition_with_refinement": bool(
                            lead_initial_transition_with_refinement
                        ),
                        "lead_initial_transition_cache_rebuild_prefix_len": int(
                            lead_initial_transition_cache_rebuild_prefix_len
                        ),
                        "lead_initial_transition_hard_boundary_only": bool(
                            lead_initial_transition_hard_boundary_only
                        ),
                        "cache_rebuild_after_this_step": cache_rebuild_after_this_step,
                        "forced_prefix_active": bool(
                            forced_prefix_ids is not None and step < len(forced_prefix_ids)
                        ),
                        "lead_delayed_transition_entry": bool(delayed_transition_entry[bi].item()),
                        "lead_delayed_transition_exit": bool(delayed_transition_exit[bi].item()),
                        "to_normal": bool(to_normal[bi].item()),
                        "to_soft": bool(to_soft[bi].item()),
                        "lead_soft_quota_active": bool(lead_soft_quota_mask[bi].item()),
                        "lead_soft_quota_ratio": float(lead_soft_quota_ratio),
                        "lead_refinement_candidate": bool(
                            lead_refinement_candidate[bi].item()
                        ),
                        "lead_refinement_active": bool(
                            lead_refinement_mask[bi].item()
                            and is_soft[bi].item()
                        ),
                        "lead_refinement_window": int(lead_refinement_window),
                        "lead_refinement_start_offset": int(lead_refinement_start_offset),
                        "lead_refinement_soft_cap": int(lead_refinement_soft_cap),
                        "lead_refinement_entropy_threshold": float(
                            lead_refinement_entropy_threshold
                        ),
                        "lead_refinement_threshold_mode": str(
                            lead_refinement_threshold_mode
                        ),
                        "lead_refinement_online_quantile": float(
                            lead_refinement_online_quantile
                        ),
                        "lead_refinement_online_jump_multiplier": float(
                            lead_refinement_online_jump_multiplier
                        ),
                        "lead_refinement_dynamic_threshold": (
                            None
                            if torch.isnan(refinement_dynamic_threshold[bi]).item()
                            else float(refinement_dynamic_threshold[bi].item())
                        ),
                        "lead_refinement_previous_entropy": (
                            None
                            if torch.isnan(refinement_previous_entropy[bi]).item()
                            else float(refinement_previous_entropy[bi].item())
                        ),
                        "lead_refinement_local_scale": (
                            None
                            if torch.isnan(refinement_local_scale[bi]).item()
                            else float(refinement_local_scale[bi].item())
                        ),
                        "lead_refinement_history_length": int(
                            refinement_history_length[bi].item()
                        ),
                        "lead_refinement_current_entropy": float(
                            cur_entropy[bi].item()
                        ),
                        "lead_refinement_entropy_proposed": bool(
                            entropy_refinement_proposal[bi].item()
                        ),
                        "lead_refinement_reference_ratio": float(
                            lead_refinement_reference_ratio
                        ),
                        "lead_refinement_soft_mix_lambda": float(
                            lead_refinement_soft_mix_lambda
                        ),
                        "lead_refinement_count_before": int(
                            lead_refinement_counts[orig]
                        ),
                        "lead_refinement_elapsed": int(
                            lead_refinement_elapsed[bi].item()
                        ),
                        "lead_initial_transition_step": (
                            None
                            if lead_initial_transition_steps[orig] < 0
                            else int(lead_initial_transition_steps[orig])
                        ),
                        "lead_guard_candidate_only": bool(
                            lead_guard_candidate_only
                        ),
                        "lead_answer_zone_lock_enabled": bool(
                            not lead_disable_answer_zone_lock
                        ),
                        "format_cooldown_active": bool(lead_format_mask[bi].item()),
                        "format_token": bool(lead_format_token_mask[bi].item()),
                        "is_highrisk_format_token": bool(lead_highrisk_format_token_mask[bi].item()),
                        "format_cooldown_highrisk_only": bool(format_cooldown_highrisk_only),
                        "format_cooldown_min_step": int(format_cooldown_min_step),
                        "format_cooldown_normal_steps": (
                            None if format_cooldown_normal_steps is None else int(format_cooldown_normal_steps)
                        ),
                        "format_cooldown_highrisk_steps": (
                            None if format_cooldown_highrisk_steps is None else int(format_cooldown_highrisk_steps)
                        ),
                        "format_cooldown_max_active": int(format_cooldown_max_active),
                        "format_cooldown_entropy_min": (
                            None if format_cooldown_entropy_min is None else float(format_cooldown_entropy_min)
                        ),
                        "format_cooldown_top1_max": (
                            None if format_cooldown_top1_max is None else float(format_cooldown_top1_max)
                        ),
                        "format_cooldown_margin_max": (
                            None if format_cooldown_margin_max is None else float(format_cooldown_margin_max)
                        ),
                        "format_cooldown_active_count": int(format_cooldown_active_counts[orig]),
                        "format_cooldown_remaining": int(format_cooldowns[orig]),
                        "raw_top1_prob": float(raw_top1_prob[bi].item()),
                        "raw_margin": float(raw_margin[bi].item()),
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "mode_before": "soft" if int(mode_before[bi].item()) == 0 else "normal",
                        "cur_ref_entropy": float(cur_ref_entropy[bi].item()),
                        "threshold_to_soft_b1": float(b1),
                        "threshold_to_normal_b2": float(b2),
                        "mode_stay_steps": int(mode_stay_steps[bi].item()),
                        "switch_count": int(switch_count[bi].item()),
                        "locked_normal": bool(locked_normal_mask[bi].item()),
                        "route_override_active": bool(route_override_active),
                        "route_override_kind": trace_route_override_kind if route_override_active else None,
                        "route_override_duration": (
                            int(trace_route_override_duration) if route_override_active else None
                        ),
                        "route_override_offset": (
                            int(step - trace_route_override_step) if route_override_active else None
                        ),
                        "route_cache_rebuild_after_tokens": (
                            int(trace_route_cache_rebuild_after_tokens)
                            if trace_route_cache_rebuild_after_tokens > 0 else None
                        ),
                        "cache_rebuild_after_this_step": bool(
                            cache_rebuild_after_this_step
                        ),
                        "cache_rebuild_kind": cache_rebuild_kind,
                        "forced_answer_probe": forced_answer_probe,
                    }
                trace_event = bool(
                    step in trace_event_steps
                    or to_soft[bi].item()
                    or to_normal[bi].item()
                    or lead_refinement_candidate[bi].item()
                    or lead_format_mask[bi].item()
                    or lead_soft_veto_mask[bi].item()
                    or cache_rebuild_after_this_step
                )
                record["trace_event"] = trace_event
                record["trace_event_kind"] = (
                    "to_soft" if bool(to_soft[bi].item())
                    else "to_normal" if bool(to_normal[bi].item())
                    else "veto" if bool(lead_soft_veto_mask[bi].item())
                    else "format" if bool(lead_format_mask[bi].item())
                    else "refinement" if bool(lead_refinement_candidate[bi].item())
                    else "cache_rebuild" if cache_rebuild_after_this_step
                    else "checkpoint" if step in trace_event_steps
                    else None
                )
                if trace_event_geometry and trace_event:
                    record.update(_embedding_geometry_record(
                        normal_emb,
                        raw_soft_emb,
                        last_emb,
                        bi,
                        visual_anchor=start_thinking_emb,
                    ))
                _add_topk_trace_fields(
                    record,
                    tokenizer,
                    probs_original,
                    probs,
                    bi,
                    trace_topk,
                )
                token_trace.append(record)
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])
        for bi in range(cur_batch):
            orig = unfinished_idx[bi]
            if bool(is_soft[bi].item()):
                lead_soft_quota_counts[orig] += 1
            if (
                bool(lead_refinement_mask[bi].item())
                and bool(is_soft[bi].item())
            ):
                lead_refinement_counts[orig] += 1
            entropy_history[bi].append(float(cur_entropy[bi].item()))
            transition_step = lead_initial_transition_steps[orig]
            if (
                lead_refinement_threshold_mode in {"online_quantile", "online_jump"}
                and transition_step >= 0
                and int(step) > int(transition_step)
            ):
                refinement_entropy_histories[orig].append(
                    float(cur_entropy[bi].item())
                )
            if lead_format_cooldown and int(format_cooldown_steps) > 0:
                if step < int(format_cooldown_min_step):
                    format_cooldowns[orig] = 0
                elif int(format_cooldown_max_active) > 0 and format_cooldown_active_counts[orig] >= int(format_cooldown_max_active):
                    format_cooldowns[orig] = 0
                elif (
                    bool(lead_format_token_mask[bi].item())
                    and (
                        not lead_guard_candidate_only
                        or bool(lead_refinement_candidate[bi].item())
                    )
                ):
                    token_text = tokenizer.decode(
                        [int(next_tokens[bi].item())],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    is_highrisk = _is_high_risk_format_token_text(token_text)
                    if is_highrisk and format_cooldown_highrisk_steps is not None:
                        steps = int(format_cooldown_highrisk_steps)
                    elif (not is_highrisk) and format_cooldown_normal_steps is not None:
                        steps = int(format_cooldown_normal_steps)
                    else:
                        steps = int(format_cooldown_steps)
                    format_cooldowns[orig] = max(0, steps - 1)
                elif format_cooldowns[orig] > 0:
                    format_cooldowns[orig] -= 1
                if bool(lead_format_mask[bi].item()):
                    format_cooldown_active_counts[orig] += 1

        # Replace route-specific history with the same multimodal prompt and
        # the actual discrete prefix emitted up to this point.
        if cache_rebuild_after_this_step:
            discrete_suffix = torch.tensor(
                [all_generated[0][prompt_len:]], dtype=input_ids.dtype, device=device
            )
            rebuild_input_ids = torch.cat([prompt_input_ids, discrete_suffix], dim=1)
            rebuild_attention_mask = torch.cat(
                [
                    prompt_attention_mask,
                    torch.ones(
                        (1, discrete_suffix.shape[1]),
                        dtype=prompt_attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            with torch.no_grad():
                prefetched_outputs = model(
                    input_ids=rebuild_input_ids,
                    attention_mask=rebuild_attention_mask,
                    **prompt_vision_inputs,
                    cache_position=torch.arange(
                        rebuild_input_ids.shape[1], device=device, dtype=torch.long
                    ),
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = prefetched_outputs.past_key_values
            attention_mask = rebuild_attention_mask
            cache_position = torch.tensor(
                [rebuild_input_ids.shape[1] - 1], device=device, dtype=torch.long
            )
        
        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        if max_switch_count is not None:
            budget_done = (answer_budget == 0) 
            cur_finished = cur_finished | budget_done

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break
        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        mode = mode[keep_idx]
        mode_stay_steps = mode_stay_steps[keep_idx]
        cur_ref_entropy = cur_ref_entropy[keep_idx]
        locked_normal_mask = locked_normal_mask[keep_idx]
        entropy_history = [entropy_history[i] for i in keep_idx.tolist()]
        if hasattr(past_key_values, "batch_select_indices"):
            keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
            past_key_values.batch_select_indices(keep_idx_tensor)
        if max_switch_count is not None:
            switch_count = switch_count[keep_idx]
            injecting = injecting[keep_idx]
            inject_queues = [inject_queues[i] for i in keep_idx.tolist()]
            answer_budget = answer_budget[keep_idx]

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    return out

# 找出输入中的视觉token位置并掩盖其他token，得到视觉token掩码，供后续动态视觉锚点计算使用
def _build_visual_token_mask(input_ids, tokenizer):
    imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if imgpad_id is None or imgpad_id < 0:
        return visual_mask

    batch_size, seq_len = input_ids.shape
    for bi in range(batch_size):
        inside_vision = False
        for pos in range(seq_len):
            token_id = int(input_ids[bi, pos].item())
            if token_id == vision_start_id:
                inside_vision = True
            elif token_id == vision_end_id:
                inside_vision = False
            elif inside_vision and token_id == imgpad_id:
                visual_mask[bi, pos] = True

    # Fallback: if the explicit vision span is unavailable, use raw image pad positions.
    if not visual_mask.any():
        visual_mask = input_ids == imgpad_id
    return visual_mask


def _compute_dynamic_visual_anchor(
    attn_layers,
    soft_emb,
    prompt_hidden_states,
    visual_token_mask,
    prompt_len,
    top_m,
    attn_last_k,
    anchor_mode="dynamic",
):
    """
    Select top-m visual tokens from current-token attention, then pool them with
    latent embedding attention to build a dynamic visual anchor.
    """
    attn_device = next(layer.device for layer in attn_layers if layer is not None)
    soft_emb = soft_emb.to(attn_device)
    prompt_hidden_states = prompt_hidden_states.to(attn_device)
    visual_token_mask = visual_token_mask.to(attn_device)
    batch_size, hidden_size = soft_emb.shape
    anchors = soft_emb.clone()
    has_anchor = torch.zeros(batch_size, dtype=torch.bool, device=soft_emb.device)

    if not attn_layers:
        raise RuntimeError(
            "generate_lead_attenachor requires decoder attentions, "
            "but model outputs.attentions is empty. Load the model with "
            "attn_implementation='eager'."
        )

    layer_views = []
    if attn_last_k is not None and int(attn_last_k) > 0:
        attn_layers = attn_layers[-int(attn_last_k):]

    for layer_attn in attn_layers:
        if layer_attn is None:
            continue
        if layer_attn.dim() != 4:
            raise RuntimeError(
                f"Unexpected attention tensor rank {layer_attn.dim()} "
                "in generate_lead_attenachor; expected [B, heads, q_len, kv_len]."
            )
        layer_views.append(layer_attn[:, :, -1, :].mean(dim=1))
    if not layer_views:
        raise RuntimeError("No usable attention layers found in outputs.attentions.")

    # Aggregate selected decoder layers, then mean over layers.
    current_attn = torch.stack(layer_views, dim=0).mean(dim=0)
    kv_len = current_attn.shape[-1]
    prompt_key_len = min(prompt_len, kv_len)

    for bi in range(batch_size):
        visual_positions = visual_token_mask[bi, :prompt_key_len].nonzero(as_tuple=False).squeeze(-1)
        if visual_positions.numel() == 0:
            continue

        visual_scores = current_attn[bi, visual_positions]
        cur_top_m = min(int(top_m), int(visual_positions.numel()))
        if cur_top_m <= 0:
            continue
        top_indices = torch.topk(visual_scores, k=cur_top_m, dim=0).indices
        top_visual_positions = visual_positions[top_indices]

        selected_visual_states = prompt_hidden_states[bi, top_visual_positions, :]
        if anchor_mode == "mean":
            anchor = selected_visual_states.mean(dim=0)
        elif anchor_mode == "dynamic":
            latent_scores = torch.matmul(
                selected_visual_states,
                soft_emb[bi],
            ) / math.sqrt(hidden_size)
            latent_weights = F.softmax(latent_scores, dim=0)
            anchor = torch.sum(
                selected_visual_states * latent_weights.unsqueeze(-1),
                dim=0,
            )
        else:
            raise ValueError(f"Unsupported reanchor_anchor_mode: {anchor_mode}")
        anchors[bi] = anchor
        has_anchor[bi] = True

    return anchors, has_anchor


def _observe_sidecar_visual_attention(
    model,
    tokenizer,
    generated_ids,
    prompt_len,
    visual_token_mask,
    vision_inputs,
    device,
    attn_last_k,
):
    """Replay a fixed prefix with eager attention without touching main-path KV."""
    if len(generated_ids) <= prompt_len:
        return {
            "sidecar_attn_observed": False,
            "sidecar_attn_error_type": "NoGeneratedToken",
            "sidecar_attn_error_message": "No generated token available for sidecar replay.",
        }

    prefix_ids = torch.tensor([generated_ids[:-1]], dtype=torch.long, device=device)
    cur_ids = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
    prefix_attention_mask = torch.ones_like(prefix_ids, device=device)
    cur_attention_mask = torch.ones((1, prefix_ids.shape[1] + 1), dtype=torch.long, device=device)
    prefix_cache_position = torch.arange(prefix_ids.shape[1], device=device, dtype=torch.long)
    cur_cache_position = torch.tensor([prefix_ids.shape[1]], device=device, dtype=torch.long)

    attn_config_values = _get_text_attn_implementation(model)
    _set_text_attn_implementation(attn_config_values, "eager")
    try:
        replay_inputs = {
            "input_ids": prefix_ids,
            "attention_mask": prefix_attention_mask,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": True,
            "return_dict": True,
            "cache_position": prefix_cache_position,
        }
        replay_inputs.update(vision_inputs)
        with torch.no_grad():
            prefix_outputs = model(**replay_inputs)
            outputs = model(
                input_ids=cur_ids,
                attention_mask=cur_attention_mask,
                past_key_values=prefix_outputs.past_key_values,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cur_cache_position,
            )
        summary = _summarize_visual_attention(
            attn_layers=outputs.attentions,
            visual_token_mask=visual_token_mask,
            prompt_len=prompt_len,
            attn_last_k=attn_last_k,
        )
        alignment = _summarize_hidden_visual_alignment(
            current_hidden=outputs.hidden_states[-1][:, -1, :],
            prompt_hidden_states=prefix_outputs.hidden_states[-1][:, :prompt_len, :],
            visual_token_mask=visual_token_mask,
            top_k=4,
        )
    finally:
        _restore_text_attn_implementation(attn_config_values)

    available = bool(summary["available"][0].item())
    align_available = bool(alignment["available"][0].item())
    return {
        "sidecar_attn_observed": True,
        "sidecar_attn_error_type": None,
        "sidecar_attn_error_message": None,
        "sidecar_visual_attn_available": available,
        "sidecar_visual_attn_mass": float(summary["mass"][0].item()),
        "sidecar_visual_attn_top1": float(summary["top1"][0].item()),
        "sidecar_visual_attn_top4_sum": float(summary["top4_sum"][0].item()),
        "sidecar_visual_attn_entropy": (
            float(summary["entropy"][0].item()) if available else None
        ),
        "sidecar_visual_attn_entropy_norm": (
            float(summary["entropy_norm"][0].item()) if available else None
        ),
        "sidecar_visual_attn_concentration": (
            float(summary["concentration"][0].item()) if available else None
        ),
        "sidecar_visual_attn_token_count": int(summary["token_count"][0].item()),
        "sidecar_hidden_visual_align_available": align_available,
        "sidecar_hidden_visual_align_max": (
            float(alignment["max"][0].item()) if align_available else None
        ),
        "sidecar_hidden_visual_align_top4_mean": (
            float(alignment["topk_mean"][0].item()) if align_available else None
        ),
        "sidecar_hidden_visual_align_token_count": int(alignment["token_count"][0].item()),
    }


def _summarize_hidden_visual_alignment(
    current_hidden,
    prompt_hidden_states,
    visual_token_mask,
    top_k=4,
):
    device = current_hidden.device
    visual_token_mask = visual_token_mask.to(device)
    prompt_hidden_states = prompt_hidden_states.to(device)
    batch_size = current_hidden.shape[0]
    available = torch.zeros(batch_size, dtype=torch.bool, device=device)
    max_sim = torch.zeros(batch_size, dtype=current_hidden.dtype, device=device)
    topk_mean = torch.zeros(batch_size, dtype=current_hidden.dtype, device=device)
    token_count = torch.zeros(batch_size, dtype=torch.long, device=device)

    current_norm = F.normalize(current_hidden, dim=-1)
    visual_norm = F.normalize(prompt_hidden_states, dim=-1)
    for bi in range(batch_size):
        visual_positions = visual_token_mask[bi, : prompt_hidden_states.shape[1]].nonzero(as_tuple=False).squeeze(-1)
        if visual_positions.numel() == 0:
            continue
        sims = torch.matmul(visual_norm[bi, visual_positions, :], current_norm[bi])
        cur_top_k = min(int(top_k), int(sims.numel()))
        if cur_top_k <= 0:
            continue
        top_vals = torch.topk(sims, k=cur_top_k, dim=0).values
        available[bi] = True
        token_count[bi] = int(visual_positions.numel())
        max_sim[bi] = top_vals[0]
        topk_mean[bi] = top_vals.mean()

    return {
        "available": available,
        "max": max_sim,
        "topk_mean": topk_mean,
        "token_count": token_count,
    }


def _summarize_visual_attention(
    attn_layers,
    visual_token_mask,
    prompt_len,
    attn_last_k,
):
    if not attn_layers:
        raise RuntimeError(
            "visual attention summary requires decoder attentions, "
            "but model outputs.attentions is empty. Load the model with "
            "attn_implementation='eager'."
        )

    layer_views = []
    if attn_last_k is not None and int(attn_last_k) > 0:
        attn_layers = attn_layers[-int(attn_last_k):]

    for layer_attn in attn_layers:
        if layer_attn is None:
            continue
        if layer_attn.dim() != 4:
            raise RuntimeError(
                f"Unexpected attention tensor rank {layer_attn.dim()} "
                "in visual attention summary; expected [B, heads, q_len, kv_len]."
            )
        layer_views.append(layer_attn[:, :, -1, :].mean(dim=1))
    if not layer_views:
        raise RuntimeError("No usable attention layers found in outputs.attentions.")

    current_attn = torch.stack(layer_views, dim=0).mean(dim=0)
    kv_len = current_attn.shape[-1]
    prompt_key_len = min(prompt_len, kv_len)
    batch_size = current_attn.shape[0]
    device = current_attn.device

    available = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mass = torch.zeros(batch_size, dtype=current_attn.dtype, device=device)
    top1 = torch.zeros(batch_size, dtype=current_attn.dtype, device=device)
    top4_sum = torch.zeros(batch_size, dtype=current_attn.dtype, device=device)
    entropy = torch.zeros(batch_size, dtype=current_attn.dtype, device=device)
    entropy_norm = torch.zeros(batch_size, dtype=current_attn.dtype, device=device)
    concentration = torch.zeros(batch_size, dtype=current_attn.dtype, device=device)
    token_count = torch.zeros(batch_size, dtype=torch.long, device=device)

    visual_token_mask = visual_token_mask.to(device)
    for bi in range(batch_size):
        visual_positions = visual_token_mask[bi, :prompt_key_len].nonzero(as_tuple=False).squeeze(-1)
        if visual_positions.numel() == 0:
            continue
        visual_scores = current_attn[bi, visual_positions]
        available[bi] = True
        token_count[bi] = int(visual_positions.numel())
        mass[bi] = visual_scores.sum()
        top1[bi] = visual_scores.max()
        cur_topk = min(4, int(visual_scores.numel()))
        top4_sum[bi] = torch.topk(visual_scores, k=cur_topk, dim=0).values.sum()
        if mass[bi].item() > 0:
            norm_scores = visual_scores / mass[bi]
            entropy[bi] = -(
                norm_scores * norm_scores.clamp(min=1e-8).log()
            ).sum()
            if visual_positions.numel() > 1:
                entropy_norm[bi] = entropy[bi] / math.log(float(visual_positions.numel()))
                concentration[bi] = 1.0 - entropy_norm[bi]

    return {
        "available": available,
        "mass": mass,
        "top1": top1,
        "top4_sum": top4_sum,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "concentration": concentration,
        "token_count": token_count,
    }


def _get_text_attn_implementation(model):
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ]
    values = []
    for config in configs:
        if config is not None and hasattr(config, "_attn_implementation"):
            values.append((config, config._attn_implementation))
    return values


def _set_text_attn_implementation(config_values, implementation):
    for config, _ in config_values:
        config._attn_implementation = implementation


def _restore_text_attn_implementation(config_values):
    for config, value in config_values:
        config._attn_implementation = value


def generate_lead_attenachor(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    # ---- **gen_kwargs ----
    temperature = kwargs.get("temperature", 1.0)
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 0)
    min_p = kwargs.get("min_p", 0)
    max_new_tokens = kwargs.get("max_new_tokens", 32768)
    do_sample = kwargs.get("do_sample", True)

    # ---- lead ----
    alpha_0 = kwargs.pop("alpha_0", 1.0)
    beta_0 = kwargs.pop("beta_0", 0.7)
    window_size = kwargs.pop("window_size", 256)
    thinking_token_id = kwargs.pop("thinking_token_id", None)
    end_thinking_token_id = kwargs.pop("end_thinking_token_id", None)
    max_switch_count = kwargs.pop("max_switch_count", None)
    math_ids_tensor = kwargs.pop("math_ids_tensor", None)
    convergence_words = kwargs.get("convergence_words", "</think>")
    termination_words = kwargs.get("termination_words", "</think>\n\nThe final answer is")
    termination_max_tokens = kwargs.pop("termination_max_tokens", 32)
    visual_anchor_top_m = kwargs.pop("visual_anchor_top_m", 32)
    visual_anchor_attn_last_k = kwargs.pop("visual_anchor_attn_last_k", 4)
    visual_anchor_lambda_scale = kwargs.pop("visual_anchor_lambda_scale", 1.0)
    visual_anchor_entropy_upper = kwargs.pop("visual_anchor_entropy_upper", None)
    visual_anchor_skip_nonword = kwargs.pop("visual_anchor_skip_nonword", False)
    visual_anchor_single_use = kwargs.pop("visual_anchor_single_use", False)
    soft_trigger_mode = kwargs.pop("soft_trigger_mode", "legacy")
    soft_warning_margin = kwargs.pop("soft_warning_margin", 0.4)
    soft_confirm_margin = kwargs.pop("soft_confirm_margin", 0.6)
    soft_delta2_threshold = kwargs.pop("soft_delta2_threshold", 0.25)
    soft_repeat_warning_boost = kwargs.pop("soft_repeat_warning_boost", 0.0)
    soft_repeat_confirm_boost = kwargs.pop("soft_repeat_confirm_boost", 0.0)
    soft_repeat_delta2_boost = kwargs.pop("soft_repeat_delta2_boost", 0.0)
    soft_repeat_cooldown = kwargs.pop("soft_repeat_cooldown", 0)
    soft_post_reset_ref_margin = kwargs.pop("soft_post_reset_ref_margin", 0.0)
    soft_post_reset_cooldown = kwargs.pop("soft_post_reset_cooldown", 0)

    stream_callback = kwargs.pop("stream_callback", None)
    token_trace = kwargs.pop("token_trace", None)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    batch_size, device = input_ids.shape[0], input_ids.device
    E = model.get_input_embeddings().weight

    def _resolve_token_id(token_text, fallback_text=None):
        token_id = None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
        except Exception:
            token_id = None
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if token_id is None or token_id == tokenizer.unk_token_id or (isinstance(token_id, int) and token_id < 0):
            text = fallback_text if fallback_text is not None else token_text
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                token_id = encoded[0]
        if token_id is None:
            token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        return token_id

    visual_token_mask = _build_visual_token_mask(input_ids, tokenizer)
    prompt_len = input_ids.shape[1]
    prompt_hidden_states = None

    if thinking_token_id is None:
        thinking_token_id = _resolve_token_id("<think>")
    if end_thinking_token_id is None:
        end_thinking_token_id = _resolve_token_id("</think>")

    imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    thinking_token_id = imgpad_id

    end_thinking_emb = E[end_thinking_token_id]
    newline_id = _resolve_token_id("\\n", "\n")
    line_break_emb = E[newline_id]
    past_key_values = None
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))
    mode = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: soft, 1: normal
    mode_stay_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    locked_normal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    anchor_used = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pre_soft_armed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    soft_entry_count = torch.zeros(batch_size, dtype=torch.long, device=device)
    post_soft_reset_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    entropy_history = [[] for _ in range(batch_size)]

    if max_switch_count is not None:
        switch_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        convergence_ids = tokenizer.encode(convergence_words, add_special_tokens=False)
        termination_ids = tokenizer.encode(termination_words, add_special_tokens=False)
        injecting = torch.zeros(batch_size, dtype=torch.bool, device=device)
        inject_queues = [[] for _ in range(batch_size)]
        answer_budget = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        cur_batch = attention_mask.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {
                "input_ids": input_ids.clone(),
            }
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if vision_inputs:
                model_inputs.update(vision_inputs)
            model_inputs["cache_position"] = cache_position
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "inputs_embeds": last_emb.unsqueeze(1),
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            model_inputs["cache_position"] = cache_position

        potential_to_soft = (
            past_key_values is not None
            and ((mode == 1) & ((mode_stay_steps + 1) >= window_size) & (~locked_normal_mask)).any()
        )
        need_decode_attn = bool(potential_to_soft)
        attn_config_values = None
        if need_decode_attn:
            attn_config_values = _get_text_attn_implementation(model)
            _set_text_attn_implementation(attn_config_values, "eager")
        try:
            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    use_cache=True,
                    output_attentions=need_decode_attn,
                    output_hidden_states=(prompt_hidden_states is None),
                )
        finally:
            if attn_config_values is not None:
                _restore_text_attn_implementation(attn_config_values)
        past_key_values = outputs.past_key_values
        if prompt_hidden_states is None:
            prompt_hidden_states = outputs.hidden_states[-1][:, :prompt_len, :].detach()
        if vision_inputs:
            vision_inputs = {}
        cache_position = cache_position[-1:] + 1

        logits_original = outputs.logits[:, -1, :]
        probs_original = F.softmax(logits_original, dim=-1)
        logits = logits_original / temperature
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
        probs = F.softmax(logits_filtered, dim=-1)
        filtered_entropy = -(
            probs * probs.clamp(min=1e-8).log()
        ).sum(dim=-1)

        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)
        locked_normal_mask = locked_normal_mask | (next_tokens == end_thinking_token_id)

        if max_switch_count is not None and injecting.any():
            mask_list = [injecting[i].item() and len(inject_queues[i]) > 0 for i in range(cur_batch)]
            force_mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
            if force_mask.any():
                force_toks = torch.tensor([inject_queues[i].pop(0) for i in range(cur_batch) if mask_list[i]], device=device, dtype=torch.long)
                next_tokens[force_mask] = force_toks
            if injecting.any():
                done_mask = torch.tensor([injecting[i] and (len(inject_queues[i]) == 0) for i in range(cur_batch)], device=device, dtype=torch.bool)
                injecting[done_mask] = False

        cur_entropy = -(probs_original * (probs_original.clamp(min=1e-8).log())).sum(dim=-1)
        delta2 = torch.zeros(cur_batch, dtype=cur_entropy.dtype, device=device)
        for bi in range(cur_batch):
            hist = entropy_history[bi]
            if hist:
                recent = hist[-3:]
                delta2[bi] = cur_entropy[bi] - (sum(recent) / float(len(recent)))
        to_soft = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        to_normal = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        armed_before_step = pre_soft_armed.clone()
        soft_entry_count_before = soft_entry_count.clone()
        if step == 0:
            cur_ref_entropy = cur_entropy.clone()
        else:
            mode_stay_steps += 1
            allow_switch = (mode_stay_steps >= window_size)
            to_normal = (mode == 0) & (cur_entropy < cur_ref_entropy) & (cur_entropy < b2)
            legacy_to_soft = (mode == 1) & (cur_entropy > cur_ref_entropy) & allow_switch & (~locked_normal_mask) & (cur_entropy > b1)
            if soft_trigger_mode == "dual_delta2":
                repeat_mask = (soft_entry_count > 0)
                effective_allow_switch = allow_switch & (
                    mode_stay_steps >= (window_size + repeat_mask.long() * int(soft_repeat_cooldown))
                )
                effective_warning_margin = cur_ref_entropy + soft_warning_margin + (
                    repeat_mask.float() * soft_repeat_warning_boost
                )
                effective_confirm_margin = cur_ref_entropy + soft_confirm_margin + (
                    repeat_mask.float() * soft_repeat_confirm_boost
                )
                effective_delta2_threshold = soft_delta2_threshold + (
                    repeat_mask.float() * soft_repeat_delta2_boost
                )
                warn_cond = (
                    (mode == 1)
                    & effective_allow_switch
                    & (~locked_normal_mask)
                    & (cur_entropy > effective_warning_margin)
                    & (delta2 > effective_delta2_threshold)
                )
                confirm_cond = (
                    (mode == 1)
                    & effective_allow_switch
                    & (~locked_normal_mask)
                    & pre_soft_armed
                    & (
                        (cur_entropy > effective_confirm_margin)
                        | (delta2 > effective_delta2_threshold)
                    )
                )
                to_soft = legacy_to_soft | confirm_cond
                clear_armed = to_soft | to_normal | locked_normal_mask | (cur_entropy <= cur_ref_entropy)
                pre_soft_armed = pre_soft_armed & (~clear_armed)
                pre_soft_armed = pre_soft_armed | (warn_cond & (~to_soft))
            else:
                to_soft = legacy_to_soft

            if (to_normal.any() or to_soft.any()) and step % 50 == 0:
                print(f"[LEAD-ATTEN] Step {step}: to_normal={to_normal.nonzero().squeeze(-1).tolist()}, to_soft={to_soft.nonzero().squeeze(-1).tolist()}")

            mode[to_normal] = 1
            mode[to_soft] = 0
            mode_stay_steps[to_normal | to_soft] = 0
            cur_ref_entropy[to_normal | to_soft] = cur_entropy[to_normal | to_soft]
            soft_entry_count = soft_entry_count + to_soft.long()
            post_soft_reset_mask = (
                to_normal
                & (soft_entry_count > 0)
                & (~post_soft_reset_done)
            )
            if post_soft_reset_mask.any():
                cur_ref_entropy = torch.where(
                    post_soft_reset_mask,
                    cur_entropy + soft_post_reset_ref_margin,
                    cur_ref_entropy,
                )
                if soft_post_reset_cooldown > 0:
                    mode_stay_steps = torch.where(
                        post_soft_reset_mask,
                        torch.full_like(mode_stay_steps, -int(soft_post_reset_cooldown)),
                        mode_stay_steps,
                    )
                post_soft_reset_done = post_soft_reset_done | post_soft_reset_mask

            if to_normal.any() or to_soft.any():
                print(f"[LEAD-ATTEN] Step {step}: switch entropy={cur_entropy.mean().item():.4f}")

            if max_switch_count is not None:
                switch_count = switch_count + to_normal.long()

        is_normal = (mode == 1) | locked_normal_mask
        if math_ids_tensor is not None:
            is_math_token = (next_tokens.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
            is_normal[is_math_token] = True
        is_soft = ~is_normal

        normal_emb = E[next_tokens]
        soft_emb = torch.matmul(probs_original, E)

        alpha = alpha_0 + (1 - alpha_0) * float(step) / float(max_new_tokens)
        lambda_t = max(0.0, min(1.0, a * (1.0 - alpha) * visual_anchor_lambda_scale))
        if step == 0:
            soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb

        guided_soft_emb = soft_emb
        anchor_applied = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if step > 0 and to_soft.any():
            attn_layers = outputs.attentions[-4:] if outputs.attentions is not None else ()
            dynamic_anchor, has_anchor = _compute_dynamic_visual_anchor(
                attn_layers=attn_layers,
                soft_emb=soft_emb,
                prompt_hidden_states=prompt_hidden_states,
                visual_token_mask=visual_token_mask,
                prompt_len=prompt_len,
                top_m=visual_anchor_top_m,
                attn_last_k=visual_anchor_attn_last_k,
            )
            apply_anchor = to_soft & has_anchor.to(to_soft.device)
            if visual_anchor_single_use:
                apply_anchor = apply_anchor & (~anchor_used)
            if visual_anchor_entropy_upper is not None:
                apply_anchor = apply_anchor & (cur_entropy <= visual_anchor_entropy_upper)
            if visual_anchor_skip_nonword:
                keep_flags = []
                for bi in range(cur_batch):
                    token_text = tokenizer.decode(
                        [int(next_tokens[bi].item())],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    keep_flags.append(bool(re.search(r"[A-Za-z0-9]", token_text)))
                keep_mask = torch.tensor(keep_flags, dtype=torch.bool, device=device)
                apply_anchor = apply_anchor & keep_mask
            anchor_applied = apply_anchor.to(device)
            guided_candidates = (1.0 - lambda_t) * soft_emb + lambda_t * dynamic_anchor.to(soft_emb.device)
            guided_soft_emb = torch.where(
                apply_anchor[:, None],
                guided_candidates,
                soft_emb,
            )
            anchor_used = anchor_used | anchor_applied
            print(
                f"[LEAD-ATTEN] Step {step}: visual_anchor_applied="
                f"{anchor_applied.nonzero().squeeze(-1).tolist()}"
            )

        beta = beta_0 + (1 - beta_0) * float(step) / float(max_new_tokens)

        if step % 200 == 0:
            print(
                f"[LEAD-ATTEN] Step {step}: alpha={alpha:.3f}, beta={beta:.3f}, "
                f"lambda={lambda_t:.3f}, soft_ratio={(mode==0).float().mean():.2f}"
            )

        if step > 0:
            mixed_emb = beta * guided_soft_emb + (1 - beta) * end_thinking_emb
            normal_emb = torch.where(to_normal[:, None], mixed_emb, normal_emb)
        last_emb = torch.where(is_soft[:, None], guided_soft_emb, normal_emb)

        if max_switch_count is not None and step > 0:
            trigger = (switch_count >= max_switch_count) & (switch_count <= 2 * max_switch_count) & to_normal
            if trigger.any():
                print(f"[LEAD-ATTEN] Inject convergence at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(convergence_ids)
                injecting = injecting | trigger

            trigger = (switch_count > 2 * max_switch_count) & to_normal
            if trigger.any():
                print(f"[LEAD-ATTEN] Inject termination at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(termination_ids)
                injecting = injecting | trigger
                answer_budget[trigger] = termination_max_tokens
            active = (answer_budget >= 0)
            if active.any():
                answer_budget = torch.where(active, answer_budget - 1, answer_budget)

        for bi, orig in enumerate(unfinished_idx):
            token_id = next_tokens[bi].item()
            all_generated[orig].append(token_id)
            entropy_history[bi].append(float(cur_entropy[bi].item()))
            if token_trace is not None:
                token_trace.append({
                    "step": int(step),
                    "batch_index": int(orig),
                    "token_id": int(token_id),
                    "raw_entropy": float(cur_entropy[bi].item()),
                    "filtered_entropy": float(filtered_entropy[bi].item()),
                    "selected_prob": float(probs[bi, next_tokens[bi]].item()),
                    "mode": "soft" if bool(is_soft[bi].item()) else "normal",
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "lambda_t": float(lambda_t),
                    "delta2": float(delta2[bi].item()),
                    "armed_before_step": bool(armed_before_step[bi].item()),
                    "soft_entry_count_before": int(soft_entry_count_before[bi].item()),
                    "to_soft": bool(to_soft[bi].item()),
                    "to_normal": bool(to_normal[bi].item()),
                    "soft_trigger_mode": soft_trigger_mode,
                    "post_soft_reset_done": bool(post_soft_reset_done[bi].item()),
                    "anchor_applied": bool(anchor_applied[bi].item()),
                })
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        if max_switch_count is not None:
            budget_done = (answer_budget == 0)
            cur_finished = cur_finished | budget_done

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break
        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        mode = mode[keep_idx]
        mode_stay_steps = mode_stay_steps[keep_idx]
        cur_ref_entropy = cur_ref_entropy[keep_idx]
        locked_normal_mask = locked_normal_mask[keep_idx]
        prompt_hidden_states = prompt_hidden_states[keep_idx]
        visual_token_mask = visual_token_mask[keep_idx]
        anchor_used = anchor_used[keep_idx]
        pre_soft_armed = pre_soft_armed[keep_idx]
        soft_entry_count = soft_entry_count[keep_idx]
        post_soft_reset_done = post_soft_reset_done[keep_idx]
        entropy_history = [entropy_history[i] for i in keep_idx.tolist()]
        if hasattr(past_key_values, "batch_select_indices"):
            keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
            past_key_values.batch_select_indices(keep_idx_tensor)
        if max_switch_count is not None:
            switch_count = switch_count[keep_idx]
            injecting = injecting[keep_idx]
            inject_queues = [inject_queues[i] for i in keep_idx.tolist()]
            answer_budget = answer_budget[keep_idx]

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out
