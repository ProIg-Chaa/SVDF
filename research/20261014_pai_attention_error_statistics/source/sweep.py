from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat
from lead import format_prompt_from_sample
from lead.inference import prepare_inputs
from lead.generation_utils import generate_lead

def create_mean_mask(image_path: str, output_path: Path) -> str:
    if output_path.is_file():
        return str(output_path)
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        mean = tuple(int(round(value)) for value in ImageStat.Stat(rgb).mean[:3])
        masked = Image.new("RGB", rgb.size, mean)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked.save(output_path, format="PNG")
    return str(output_path)



def prepare(processor, sample: dict[str, Any], image_path: str, device: torch.device):
    prompt = format_prompt_from_sample(sample, use_cot=False)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = prepare_inputs(processor, messages, device)
    return inputs, int(inputs["input_ids"].shape[1])



def talr_kwargs(
    math_ids_tensor,
    max_new_tokens: int,
    refinement_entropy_threshold: float = 1.25,
    refinement_threshold_mode: str = "fixed",
    refinement_online_quantile: float = 0.75,
    refinement_online_jump_multiplier: float = 1.0,
    refinement_window: int = 8,
    refinement_soft_cap: int = 2,
) -> dict[str, Any]:
    return {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "alpha_0": 0.4,
        "beta_0": 0.7,
        "max_switch_count": 5,
        "window_size": 128,
        "math_ids_tensor": math_ids_tensor,
        "convergence_words": "</think>",
        "lead_initial_transition_with_refinement": True,
        "lead_refinement_window": refinement_window,
        "lead_refinement_soft_cap": refinement_soft_cap,
        "lead_refinement_entropy_threshold": refinement_entropy_threshold,
        "lead_refinement_threshold_mode": refinement_threshold_mode,
        "lead_refinement_online_quantile": refinement_online_quantile,
        "lead_refinement_online_jump_multiplier": refinement_online_jump_multiplier,
        "lead_refinement_soft_mix_lambda": 0.95,
        "lead_guard_candidate_only": True,
        "lead_disable_answer_zone_lock": True,
    }



def run_talr(
    model,
    processor,
    tokenizer,
    sample: dict[str, Any],
    math_ids_tensor,
    max_new_tokens: int,
    forced_prefix_ids: list[int] | None = None,
    override_step: int = -1,
    external_residual: torch.Tensor | None = None,
    residual_strength: float = 0.75,
    residual_duration: int = 1,
    refinement_entropy_threshold: float = 1.25,
    refinement_threshold_mode: str = "fixed",
    refinement_online_quantile: float = 0.75,
    refinement_online_jump_multiplier: float = 1.0,
    refinement_window: int = 8,
    refinement_soft_cap: int = 2,
) -> tuple[str, list[int], list[dict[str, Any]], float]:
    inputs, prompt_len = prepare(
        processor, sample, str(sample["image"]), next(model.parameters()).device
    )
    trace: list[dict[str, Any]] = []
    kwargs = talr_kwargs(
        math_ids_tensor, max_new_tokens, refinement_entropy_threshold,
        refinement_threshold_mode, refinement_online_quantile,
        refinement_online_jump_multiplier, refinement_window,
        refinement_soft_cap,
    )
    kwargs["token_trace"] = trace
    kwargs["trace_topk"] = 0
    if forced_prefix_ids is not None:
        kwargs["forced_prefix_ids"] = forced_prefix_ids
    if external_residual is not None:
        kwargs["trace_route_override_step"] = override_step
        kwargs["trace_route_override_kind"] = "method_plus_external_residual"
        kwargs["trace_route_override_mix_lambda"] = residual_strength
        kwargs["trace_route_override_duration"] = residual_duration
        kwargs["trace_external_route_vector"] = external_residual
        kwargs["trace_external_route_source"] = "true_mask_residual"
    started = time.perf_counter()
    with torch.no_grad():
        output = generate_lead(model, tokenizer, **inputs, **kwargs)[0]
    elapsed = time.perf_counter() - started
    token_ids = [int(token) for token in output[prompt_len:].tolist()]
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return text, token_ids, trace, elapsed



def matched_visual_direction(true_capture: dict[str, Any], mask_capture: dict[str, Any]):
    hard = true_capture["hard_embedding"].float()
    true_soft = true_capture["soft_embedding"].float()
    hard_norm = float(torch.linalg.vector_norm(hard).item())
    true_soft_norm = float(torch.linalg.vector_norm(true_soft).item())
    target_norm = float(torch.linalg.vector_norm(true_soft - hard).item())
    mask_delta = (
        mask_capture["soft_embedding"].float()
        - mask_capture["hard_embedding"].float()
    )
    mask_norm = float(torch.linalg.vector_norm(mask_delta).item())
    aligned_mask = hard.clone()
    if mask_norm > 1e-12:
        aligned_mask = hard + mask_delta * (target_norm / mask_norm)
    raw_direction = true_soft - aligned_mask
    raw_norm = float(torch.linalg.vector_norm(raw_direction).item())
    direction = torch.zeros_like(hard)
    if raw_norm > 1e-12:
        direction = raw_direction * (target_norm / raw_norm)
    return direction, target_norm, raw_norm, hard_norm, true_soft_norm



def patch_segmented_vision_sdpa() -> None:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    def segmented_vision_sdpa(module, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        seq_length = hidden_states.shape[0]
        q, k, v = (
            module.qkv(hidden_states)
            .reshape(seq_length, 3, module.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos().float(), emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = modeling_qwen2_5_vl.apply_rotary_pos_emb_vision(q, k, cos, sin)
        boundaries = cu_seqlens.detach().cpu().tolist()
        pieces = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            output = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
            )
            pieces.append(output.squeeze(0).transpose(0, 1))
        return module.proj(torch.cat(pieces, dim=0).reshape(seq_length, -1))

    modeling_qwen2_5_vl.Qwen2_5_VLVisionSdpaAttention.forward = segmented_vision_sdpa


