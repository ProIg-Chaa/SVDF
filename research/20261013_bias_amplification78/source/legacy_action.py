from __future__ import annotations
from typing import Any
from pathlib import Path
import hashlib
import torch
from sweep import prepare,talr_kwargs
from lead.generation_utils import generate_lead

def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")

def capture_talr_vector(
    model,
    processor,
    tokenizer,
    sample: dict[str, Any],
    image_path: str,
    prefix_ids: list[int],
    event_step: int,
    math_ids_tensor,
    refinement_entropy_threshold: float = 1.25,
    refinement_threshold_mode: str = "fixed",
    refinement_online_quantile: float = 0.75,
    refinement_online_jump_multiplier: float = 1.0,
    refinement_window: int = 8,
    refinement_soft_cap: int = 2,
) -> dict[str, Any]:
    inputs, _ = prepare(processor, sample, image_path, next(model.parameters()).device)
    collector: list[dict[str, Any]] = []
    kwargs = talr_kwargs(
        math_ids_tensor, event_step + 1, refinement_entropy_threshold,
        refinement_threshold_mode, refinement_online_quantile,
        refinement_online_jump_multiplier, refinement_window,
        refinement_soft_cap,
    )
    kwargs["forced_prefix_ids"] = prefix_ids
    kwargs["trace_soft_vector_collector"] = collector
    kwargs["trace_capture_soft_vector_step"] = event_step
    with torch.no_grad():
        generate_lead(model, tokenizer, **inputs, **kwargs)
    if len(collector) != 1:
        raise RuntimeError(f"Expected one vector at step {event_step}, got {len(collector)}")
    return collector[0]

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

def random_direction_like(reference: torch.Tensor, norm: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(reference.numel(), generator=generator, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    return direction.reshape_as(reference) * norm
