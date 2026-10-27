"""Two frozen h2 rules. No text, labels, sample metadata, or I/O in this module."""
import math

import torch


RULES = ("faithful_h2_legacy", "positive_finite_h2_v1")


def legacy_cosine(left, right):
    denominator = float(left.float().norm() * right.float().norm())
    if denominator <= 1e-20:
        return None
    return float(torch.dot(left.float(), right.float()) / denominator)


def scalar(value):
    if value is None or math.isfinite(value):
        return value
    return str(value)


def decide(base, half, full, random_control):
    """Return literal legacy behavior and one separate fail-closed comparison."""
    vectors = (base, half, full, random_control)
    invalid_structure = any(not isinstance(p, torch.Tensor) for p in vectors)
    if not invalid_structure:
        invalid_structure = any(p.ndim != 1 or p.shape != base.shape or p.numel() == 0 for p in vectors)
    if invalid_structure:
        return {"faithful_accept": None, "conservative_accept": False,
                "available": False, "unavailable_reason": "missing_or_shape_mismatch",
                "inputs_valid": False}
    b, h, f, n = [p.detach().float().cpu() for p in vectors]
    df, dh, dn = f - b, h - b, n - b
    ch, cn = legacy_cosine(df, dh), legacy_cosine(df, dn)
    faithful = ch is not None and (cn is None or ch > cn)
    valid = all(bool(torch.isfinite(p).all()) and bool((p >= 0).all())
                and bool(torch.isfinite(p.sum())) and float(p.sum()) > 0 for p in (b, h, f, n))
    norms = [float(d.norm()) for d in (df, dh, dn)]
    conservative = (valid and all(math.isfinite(v) and v > 0 for v in norms)
                    and ch is not None and cn is not None
                    and math.isfinite(ch) and math.isfinite(cn)
                    and ch > max(0.0, cn))
    return {
        "faithful_accept": bool(faithful), "conservative_accept": bool(conservative),
        "available": True, "inputs_valid": valid,
        "full_half_cos": scalar(ch), "full_random_cos": scalar(cn),
        "full_half_defined": ch is not None, "full_random_defined": cn is not None,
        "delta_norms_f_h_n": [scalar(v) for v in norms],
        "delta_nonzero_f_h_n": [int(torch.count_nonzero(d)) for d in (df, dh, dn)],
        "finite_counts_b_h_f_n": [int(torch.isfinite(p).sum()) for p in (b, h, f, n)],
        "probability_mass_b_h_f_n": [scalar(float(p.sum())) for p in (b, h, f, n)],
        "vocabulary_size": b.numel(),
    }
