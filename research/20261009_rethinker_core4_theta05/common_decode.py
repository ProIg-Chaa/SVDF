"""Shared native greedy CoT decoder. No gold labels or future-token inputs."""
import torch
import torch.nn.functional as F

class GeometryUnavailable(RuntimeError):
    """A represented random update cannot meet the preregistered geometry."""
    pass

def apply_sampling_filter(logits, top_k=0, top_p=1.0, min_p=0.0):
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float("-inf"), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, float("-inf"), logits)
    return logits

def select_token(logits, config):
    """Preserve native dtype through division, filtering, softmax, and argmax."""
    raw_probs = F.softmax(logits, dim=-1)
    temperature = float(config.get("temperature", 0.6))
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    filtered = apply_sampling_filter(
        logits / temperature, top_k=int(config.get("top_k", 20)),
        top_p=float(config.get("top_p", 0.95)), min_p=float(config.get("min_p", 0.0)))
    probs = F.softmax(filtered, dim=-1)
    if config.get("do_sample", False):
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        next_tokens = torch.argmax(probs, dim=-1)
    return next_tokens, raw_probs, probs

def prefill(model, inputs, *, output_attentions=False, output_hidden_states=False):
    """Return (outputs, state) with position/mask ready for the next token."""
    input_ids = inputs["input_ids"].clone()
    if input_ids.shape[0] != 1:
        raise ValueError("Audit decoder requires batch size 1")
    attention_mask = inputs.get("attention_mask")
    attention_mask = attention_mask.clone() if attention_mask is not None else None
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
    kwargs = {"input_ids": input_ids, "cache_position": cache_position}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    for key, value in inputs.items():
        if any(tag in key for tag in ("pixel", "image", "video")) and value is not None:
            kwargs[key] = value
    outputs = model(**kwargs, use_cache=True, output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states)
    state = {"past_key_values": outputs.past_key_values, "attention_mask": attention_mask,
             "cache_position": cache_position[-1:] + 1}
    return outputs, state

def advance(model, next_tokens, state, *, inputs_embeds=None,
            output_attentions=False, output_hidden_states=False):
    """Consume the selected token or supplied [1,1,H] feedback exactly once."""
    attention_mask = state["attention_mask"]
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones(
            (next_tokens.shape[0], 1), dtype=attention_mask.dtype,
            device=attention_mask.device)], dim=1)
    kwargs = {"past_key_values": state["past_key_values"],
              "cache_position": state["cache_position"]}
    if inputs_embeds is None:
        kwargs["input_ids"] = next_tokens.unsqueeze(1)
    else:
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(1)
        kwargs["inputs_embeds"] = inputs_embeds
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    outputs = model(**kwargs, use_cache=True, output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states)
    new_state = {"past_key_values": outputs.past_key_values, "attention_mask": attention_mask,
                 "cache_position": state["cache_position"][-1:] + 1}
    return outputs, new_state

def token_record(step, token, logits, raw_probs, probs):
    raw_top2 = torch.topk(logits, min(2, logits.shape[-1]), dim=-1)
    token_id = int(token.item())
    return {"step":int(step), "token_id":token_id, "mode":"hard",
            "selected_raw_probability":float(raw_probs[0,token_id].item()),
            "selected_filtered_probability":float(probs[0,token_id].item()),
            "raw_logits_dtype":str(logits.dtype),
            "raw_top1_id":int(raw_top2.indices[0,0].item()),
            "raw_top2_gap":float((raw_top2.values[0,0]-raw_top2.values[0,1]).float().item())}

@torch.inference_mode()
def generate_baseline(model, tokenizer, inputs, *, seed=0, config=None, context=None):
    config = dict(config or {})
    outputs, state = prefill(model, inputs)
    token_ids, trace = [], []
    maximum = int(config.get("max_new_tokens", 1024))
    reached_eos = False
    for step in range(maximum):
        logits = outputs.logits[:, -1, :]
        token, raw_probs, probs = select_token(logits, config)
        token_ids.append(int(token.item()))
        trace.append(token_record(step, token, logits, raw_probs, probs))
        if tokenizer.eos_token_id is not None and int(token.item()) == int(tokenizer.eos_token_id):
            reached_eos = True
            break
        if step + 1 < maximum:
            outputs, state = advance(model, token, state)
    return {"token_ids":token_ids, "trace":trace, "metadata":{
        "method":"cot", "variant":"baseline", "seed":int(seed),
        "n_generated":len(token_ids), "reached_eos":reached_eos,
        "max_new_tokens":maximum, "truncated":not reached_eos and len(token_ids)==maximum,
        "selection_dtype":"native", "cache_semantics":"prefill IDs+vision, incremental hard IDs"}}
