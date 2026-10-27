"""One fixed, label-free vocabulary-space visual action. No model calls."""
import hashlib
import torch

EPS = 1e-12

def norm_match(vector, target_norm):
    vector = vector.float()
    target = float(target_norm)
    if not torch.isfinite(vector).all() or not torch.isfinite(torch.tensor(target)) or target < 0:
        raise ValueError('nonfinite vector or invalid target norm')
    size = torch.linalg.vector_norm(vector)
    if size.item() <= EPS or target == 0:
        return torch.zeros_like(vector)
    return vector * (target / size)

def probability_shift(true_probs, mask_probs):
    """q(v) proportional to pI(v)^2/(pI(v)+pM(v)); return q-pI."""
    p, m = true_probs.float().flatten(), mask_probs.float().flatten()
    if p.shape != m.shape or p.numel() == 0:
        raise ValueError('probability shape mismatch')
    for values in (p, m):
        if not torch.isfinite(values).all() or (values < 0).any() or values.sum().item() <= 0:
            raise ValueError('invalid probability vector')
    p, m = p / p.sum(), m / m.sum()
    if torch.equal(p, m):
        return torch.zeros_like(p)
    denominator = p + m
    support = p / denominator.clamp_min(torch.finfo(p.dtype).tiny)
    q = p * support
    q = q / q.sum()
    return q - p

def bounded_reweight(true_probs, mask_probs, embedding_weight, target_norm, chunk_size=4096):
    """Float32 chunked vocabulary reduction on the embedding's original device."""
    shift = probability_shift(true_probs, mask_probs).to(embedding_weight.device)
    if embedding_weight.ndim != 2 or embedding_weight.shape[0] != shift.numel():
        raise ValueError('embedding vocabulary mismatch')
    if chunk_size <= 0:
        raise ValueError('chunk size must be positive')
    raw = torch.zeros(embedding_weight.shape[1], device=embedding_weight.device, dtype=torch.float32)
    for start in range(0, shift.numel(), chunk_size):
        stop = start + chunk_size
        raw += shift[start:stop] @ embedding_weight[start:stop].float()
    action = norm_match(raw, target_norm)
    metrics = {'probability_shift_l1': shift.abs().sum().item(),
        'probability_shift_sum': shift.sum().item(), 'raw_norm': torch.linalg.vector_norm(raw).item(),
        'effective_norm': torch.linalg.vector_norm(action).item(), 'target_norm': float(target_norm)}
    return action, metrics

def matched_random(reference, target_norm, sample_key):
    seed = int.from_bytes(hashlib.sha256(('visual-action-random-v1:' + sample_key).encode()).digest()[:8], 'big')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    vector = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return norm_match(vector, target_norm).to(reference.device)

def apply_action(state, action=None, strength=1.0):
    """Default-off returns the very same tensor, with no arithmetic or copy."""
    if action is None:
        return state
    return state + float(strength) * action.to(device=state.device, dtype=state.dtype)
