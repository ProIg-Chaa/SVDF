"""R1/Qwen2.5VL adaptation of DAMO-NLP-SG/VCD, with a matched-logit null.

This module does not monkeypatch transformers or edit the official repository.
Original/random use native score arithmetic and uniform greedy decoding; zero keeps
the native logit dtype and the shared selection function for baseline equivalence.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch


OFFICIAL_REPOSITORY = "https://github.com/DAMO-NLP-SG/VCD"
NOISE_SEED = 12345
RADIUS_TOLERANCE = 0.05


from adapters.common_decode import GeometryUnavailable as SharedGeometryUnavailable


class GeometryUnavailable(SharedGeometryUnavailable):
    """The native-cast random radius cannot be matched at this prefix."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = dict(details or {})


def _selector(logits, config):
    from adapters.common_decode import select_token
    return select_token(logits, config)


def add_diffusion_noise(pixel_values, noise_step=500, noise_seed=NOISE_SEED):
    """Official sigmoid beta schedule, applied after common image processing.

    Qwen patch ordering is retained: no image resize, extra processor invocation,
    or grid change. CPU FP32 corruption precedes conversion to the prepared dtype.
    """
    step = int(noise_step)
    if not 0 <= step < 1000:
        raise ValueError("VCD noise_step must be in [0, 999]")
    if not torch.isfinite(pixel_values).all():
        raise ValueError("non-finite prepared pixels")
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000, dtype=torch.float32))
    betas = betas * (0.5e-2 - 1e-5) + 1e-5
    alpha_bar = torch.cumprod(1 - betas, dim=0)[step]
    clean = pixel_values.detach().to(device="cpu", dtype=torch.float32)
    rng = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    noise = torch.randn(clean.shape, generator=rng, dtype=torch.float32)
    corrupted = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
    corrupted = corrupted.to(device=pixel_values.device, dtype=pixel_values.dtype)
    if not torch.isfinite(corrupted).all():
        raise ValueError("non-finite diffusion pixels")
    return corrupted


def represented_permutation(base, effective, dtype, rng, support_cap=512):
    """One random priority, repaired only by native-cast representability.

    The identity is always feasible. The matching never consults token semantics,
    likelihood ranks, generated output quality, or benchmark answers. A large
    incompatible support uses an explicitly tagged identity to bound CPU memory.
    """
    base = base.detach().to(device="cpu", dtype=torch.float32)
    effective = effective.detach().to(device="cpu", dtype=torch.float32)
    count = int(base.numel())
    priority = torch.randperm(count, generator=rng)
    candidate = (base + effective[priority]).to(dtype).float() - base
    compatible_direct = torch.equal(candidate, effective[priority])
    repaired = False
    support_cap_noop = False
    if compatible_direct:
        permutation = priority
    elif count > support_cap:
        permutation = torch.arange(count)
        support_cap_noop = True
    else:
        repaired = True
        compatibility = ((base[:,None] + effective[None,:]).to(dtype).float()
                         - base[:,None]) == effective[None,:]
        if not bool(compatibility.diag().all()):
            raise GeometryUnavailable("Native original deltas lost diagonal representability")
        matrix = compatibility.numpy()
        destination_to_source = [-1] * count
        source_to_destination = [-1] * count
        fixed_priority = priority.tolist()
        # Iterative augmenting paths avoid recursion depth limits.
        for initial_destination in fixed_priority:
            parent_source = [-1] * count
            seen_destination = [False] * count
            seen_destination[initial_destination] = True
            queue = [initial_destination]
            queue_index = 0
            free_source = -1
            while queue_index < len(queue) and free_source < 0:
                destination = queue[queue_index]
                queue_index += 1
                preferred_source = fixed_priority[destination]
                order = [preferred_source] + [source for source in fixed_priority
                                             if source != preferred_source]
                for source in order:
                    if parent_source[source] >= 0 or not matrix[destination,source]:
                        continue
                    parent_source[source] = destination
                    occupied_destination = source_to_destination[source]
                    if occupied_destination < 0:
                        free_source = source
                        break
                    if not seen_destination[occupied_destination]:
                        seen_destination[occupied_destination] = True
                        queue.append(occupied_destination)
            if free_source < 0:
                raise GeometryUnavailable("Feasible native permutation matching unexpectedly failed")
            source = free_source
            while source >= 0:
                destination = parent_source[source]
                old_source = destination_to_source[destination]
                destination_to_source[destination] = source
                source_to_destination[source] = destination
                source = old_source
        permutation = torch.tensor(destination_to_source, dtype=torch.long)
    if sorted(permutation.tolist()) != list(range(count)):
        raise GeometryUnavailable("Native permutation is not a bijection")
    permuted = effective[permutation]
    scores = (base + permuted).to(dtype)
    realized = scores.float() - base
    if not torch.equal(realized, permuted):
        raise GeometryUnavailable("Permuted effective components were not exactly represented")
    identity = torch.equal(permutation, torch.arange(count))
    details = {
        "random_null_type": "native_representable_coordinate_permutation",
        "permutation_identity": identity,
        "permuted_coordinates": int((permutation != torch.arange(count)).sum().item()),
        "changed_delta_coordinates": int((permuted != effective).sum().item()),
        "permutation_no_direction_change": torch.equal(permuted, effective),
        "permutation_numeric_repair_used": repaired,
        "permutation_support_cap": support_cap,
        "permutation_support_cap_identity": support_cap_noop,
        "permutation_sha256": hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
        "effective_component_multiset_exactly_preserved": True,
    }
    return scores, details


def build_scores(original_logits, noisy_logits, *, alpha, beta, variant, rng):
    """Official native VCD logits, or represented component-permutation null."""
    if variant not in ("original", "random"):
        raise ValueError("build_scores requires original or random")
    if original_logits.shape != noisy_logits.shape:
        raise ValueError("original/noise logits shapes differ")
    if original_logits.ndim != 2 or original_logits.shape[0] != 1:
        raise ValueError("VCD audit adapter requires a single-sample batch")
    if original_logits.dtype != noisy_logits.dtype:
        raise ValueError("original/noise logit dtypes differ")
    if not torch.isfinite(original_logits).all() or not torch.isfinite(noisy_logits).all():
        raise ValueError("non-finite original/noise logits")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("VCD alpha must be finite and nonnegative")
    if not math.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("VCD beta must be in (0, 1]")
    cutoff = torch.log(torch.tensor(beta, device=original_logits.device)) + original_logits.max(
        dim=-1, keepdim=True).values
    allowed = original_logits >= cutoff
    allowed_ids = allowed[0].nonzero(as_tuple=False).flatten()
    if not len(allowed_ids):
        raise ValueError("empty original plausibility support")
    native_vcd = (1 + alpha) * original_logits - alpha * noisy_logits
    if not torch.isfinite(native_vcd).all():
        raise ValueError("non-finite native VCD logits")
    base = original_logits[0,allowed_ids].float()
    effective = native_vcd[0,allowed_ids].float() - base
    effective_mean = effective.mean()
    target_centered = effective - effective_mean
    target_value = float(torch.linalg.vector_norm(target_centered).item())
    permutation_stats = {}
    if variant == "original":
        scores = native_vcd
    else:
        supported_scores,permutation_stats = represented_permutation(
            base, effective, original_logits.dtype, rng)
        scores = original_logits.clone()
        scores[0,allowed_ids] = supported_scores.to(original_logits.device)
    applied = scores[0,allowed_ids].float() - base
    applied_centered = applied - applied.mean()
    applied_l2 = float(torch.linalg.vector_norm(applied_centered).item())
    error = abs(applied_l2 - target_value)
    relative_error = error / target_value if target_value else 0.0
    # Multiset equality is exact; reductions can differ by FP32 summation order.
    if relative_error > 1e-5:
        raise GeometryUnavailable("Exact component permutation radius reduction mismatch")
    scores = scores.masked_fill(~allowed, -float("inf"))
    if scores.dtype != original_logits.dtype:
        raise ValueError("VCD selection dtype drift")
    if not torch.isfinite(scores[allowed]).all():
        raise ValueError("non-finite supported VCD scores")
    ids_cpu = allowed_ids.detach().to("cpu")
    stats = {
        "allowed_count": int(len(allowed_ids)),
        "allowed_ids_sha256": hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest(),
        "allowed_token_ids": ids_cpu.tolist() if len(allowed_ids) <= 2048 else None,
        "cutoff": float(cutoff.item()),
        "effective_correction_mean": float(effective_mean.item()),
        "applied_correction_mean": float(applied.mean().item()),
        "effective_correction_l2": target_value,
        "applied_correction_l2": applied_l2,
        "correction_l2_abs_error": error,
        "correction_l2_relative_error": relative_error,
        "raw_correction_l2": float(torch.linalg.vector_norm(
            native_vcd.float() - original_logits.float()).item()),
        "finite_supported_scores": True, "native_score_dtype": str(scores.dtype),
        "random_match_max_relative_error": 1e-5,
        "random_skipped": False, "random_skip_reason": None,
        "intervention_applied": target_value > 0,
        "plausibility_mask_applied": True, "geometry_unavailable": False,
        "random_no_effect": target_value == 0,
    }
    stats.update(permutation_stats)
    return scores, allowed, stats


def _prefill(model, inputs):
    from adapters.common_decode import prefill
    return prefill(model, inputs)


def _advance(model, token, state):
    from adapters.common_decode import advance
    return advance(model, token, state)


def _cache_length(cache):
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        return int(cache[0][0].shape[-2])
    return None


def _entropy(probabilities):
    p = probabilities.float()
    nonzero = p > 0
    return float(-(p[nonzero] * p[nonzero].log()).sum().item())


def _eos_ids(model, tokenizer):
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    return {int(value)} if isinstance(value, int) else {int(x) for x in value}


def _official_commit():
    path = Path(__file__).resolve().parent.parent / "upstream/vcd_OFFICIAL_SOURCE.json"
    if not path.exists():
        raise ValueError("official VCD commit provenance is missing")
    return json.loads(path.read_text())["commit"]


@torch.inference_mode()
def generate(model, tokenizer, inputs, *, variant="original", seed=0,
             config=None, context=None):
    config = dict(config or {})
    context = context or {}
    if variant not in ("original", "random", "zero"):
        raise ValueError(f"unknown VCD variant {variant}")
    if config.get("do_sample", False):
        raise ValueError("cross-method audit freezes do_sample=False")
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("single-sample batches only")
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise ValueError("VCD requires prepared image pixels and image_grid_thw")
    alpha = float(config.get("vcd_alpha", 1.0))
    beta = float(config.get("vcd_beta", 0.1))
    noise_step = int(config.get("noise_step", 500))
    noise_seed = int(config.get("vcd_noise_seed", NOISE_SEED))
    max_new_tokens = int(config.get("max_new_tokens", 1024))
    if max_new_tokens <= 0:
        raise ValueError("positive max_new_tokens required")
    model.eval()
    if variant == "zero":
        from adapters.common_decode import generate_baseline
        result = generate_baseline(model, tokenizer, inputs, seed=seed,
                                   config=config, context=context)
        result["metadata"].update({
            "method": "VCD", "variant": "zero", "zero_uses_common_baseline": True,
            "official_repository": OFFICIAL_REPOSITORY, "official_commit": _official_commit(),
            "intervention_applied": False, "selection_dtype": "native",
        })
        return result
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    started = time.perf_counter()
    original, state_orig = _prefill(model, inputs)
    forward_calls = 1
    noisy = None
    grid = inputs["image_grid_thw"].detach().clone()
    if variant != "zero":
        noisy_inputs = dict(inputs)
        noisy_inputs["pixel_values"] = add_diffusion_noise(
            inputs["pixel_values"], noise_step, noise_seed
        )
        noisy, state_noise = _prefill(model, noisy_inputs)
        forward_calls += 1
        if original.past_key_values is noisy.past_key_values:
            raise ValueError("original/noise cache alias")
    generated = []
    trace = []
    eos_ids = _eos_ids(model, tokenizer)
    prompt_length = int(inputs["input_ids"].shape[1])
    stop_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        original_logits = original.logits[:, -1, :]
        if not torch.isfinite(original_logits).all():
            raise ValueError("non-finite original logits")
        base_token, base_probs, base_filtered_probs = _selector(original_logits, config)
        cache_orig_length = _cache_length(original.past_key_values)
        if cache_orig_length is not None and cache_orig_length != prompt_length + step:
            raise ValueError("original cache/prefix length mismatch")
        stats = {
            "step": step,
            "prefix_length": prompt_length + step,
            "base_token_id": int(base_token.item()),
            "original_entropy": _entropy(base_probs),
            "original_logit_norm": float(torch.linalg.vector_norm(original_logits.float()).item()),
            "cache_original_length": cache_orig_length,
            "finite_original_logits": True,
            "intervention_applied": variant != "zero",
        }
        if variant == "zero":
            selected, raw_probs, filtered_probs = base_token, base_probs, base_filtered_probs
            stats.update({"effective_correction_l2": 0.0, "applied_correction_l2": 0.0})
        else:
            if not torch.equal(grid, inputs["image_grid_thw"]):
                raise ValueError("image grid mutated")
            noisy_logits = noisy.logits[:, -1, :]
            scores, allowed, correction_stats = build_scores(
                original_logits, noisy_logits, alpha=alpha, beta=beta,
                variant=variant, rng=rng
            )
            selected, raw_probs, filtered_probs = _selector(scores, config)
            token_id = int(selected.item())
            if not bool(allowed[0, token_id].item()):
                raise ValueError("selected token outside original plausibility support")
            noise_length = _cache_length(noisy.past_key_values)
            if noise_length is not None and noise_length != prompt_length + step:
                raise ValueError("noisy cache/prefix length mismatch")
            stats.update(correction_stats)
            stats.update({
                "cache_noise_length": noise_length,
                "same_token_prefix": True,
                "noise_logit_norm": float(torch.linalg.vector_norm(noisy_logits.float()).item()),
                "finite_noise_logits": True,
                "selected_correction": float((scores[0, token_id] - original_logits[0, token_id].float()).item()),
                "selected_original_logit": float(original_logits[0, token_id].item()),
                "selected_noise_logit": float(noisy_logits[0, token_id].item()),
            })
        token_id = int(selected.item())
        generated.append(token_id)
        stats.update({
            "token_id": token_id,
            "selected_differs_from_base": token_id != int(base_token.item()),
            "selected_original_probability": float(base_probs[0, token_id].item()),
            "selected_probability": float(filtered_probs[0, token_id].item()),
            "contrast_entropy": _entropy(raw_probs),
            "filtered_entropy": _entropy(filtered_probs),
            "cumulative_forward_calls": forward_calls,
            "cumulative_elapsed_seconds": time.perf_counter() - started,
        })
        trace.append(stats)
        if token_id in eos_ids:
            stop_reason = "eos"
            break
        if step + 1 < max_new_tokens:
            original, state_orig = _advance(model, selected, state_orig)
            forward_calls += 1
            if variant != "zero":
                noisy, state_noise = _advance(model, selected, state_noise)
                forward_calls += 1
                if not torch.equal(state_orig["cache_position"], state_noise["cache_position"]):
                    raise ValueError("original/noise consumed different token positions")
    metadata = {
        "method": "VCD", "variant": variant, "seed": int(seed),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": _official_commit(),
        "vcd_alpha": alpha, "vcd_beta": beta,
        "noise_step": noise_step, "noise_seed": noise_seed,
        "score_dtype": str(original.logits.dtype),
        "image_corruption": "official sigmoid diffusion on prepared pixel_values; fixed original grid",
        "native_default_alpha": 1.0,
        "adaptation": "Qwen2.5VL fixed-grid dual cache; common greedy instead of official stochastic sampling; native official VCD score arithmetic",
        "random_control": "Native-representable permutation of actual effective correction components inside original plausibility support; exact amplitude multiset, mean and centered L2 preserved; fixed one random priority repaired only for representability; identity/no-direction-change explicitly tagged; independent fixed noise seed",
        "uses_gold_or_future": False,
        "stop_reason": stop_reason,
        "generated_tokens": len(generated),
        "prefill_calls": 1 if variant == "zero" else 2,
        "total_forward_calls": forward_calls,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {"token_ids": generated, "trace": trace, "metadata": metadata}
