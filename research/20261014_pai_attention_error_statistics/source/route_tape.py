"""Record/replay only past generated decoder input embeddings, never outputs."""
import hashlib
import torch


def tensor_hash(value):
    raw = value.detach().cpu().contiguous()
    header = (str(raw.dtype)+str(tuple(raw.shape))).encode()
    return hashlib.sha256(header+raw.view(torch.uint8).numpy().tobytes()).hexdigest()


class RouteTape:
    def __init__(self, model, reference=None, enabled=True):
        self.model, self.reference, self.enabled = model, reference, enabled
        self.states, self.positions, self.mask_shapes = [], [], []
        self.input_drift_norms = []
        self.prefill = None
        self.forward_calls = 0
        self.handle = None

    def __enter__(self):
        if self.enabled:
            self.handle = self.model.register_forward_pre_hook(self._hook,with_kwargs=True)
        return self

    def __exit__(self, error_type, error, traceback):
        if self.handle is not None: self.handle.remove()
        if error_type is None and self.enabled:
            assert self.prefill is not None, 'missing prefill'
            if self.reference is not None:
                assert len(self.states)==len(self.reference.states), 'incomplete tape replay'
        return False

    def _hook(self, module, args, kwargs):
        assert not args, 'unexpected positional model arguments'
        self.forward_calls += 1
        if kwargs.get('input_ids') is not None:
            assert self.prefill is None and not self.states, 'unexpected repeated prefill'
            assert kwargs.get('pixel_values') is not None and kwargs['pixel_values'].numel()>0
            assert kwargs.get('image_grid_thw') is not None
            self.prefill = dict(input_ids_hash=tensor_hash(kwargs['input_ids']),
                grid_hash=tensor_hash(kwargs['image_grid_thw']),
                pixels_hash=tensor_hash(kwargs['pixel_values']))
            if self.reference is not None:
                assert self.prefill['input_ids_hash']==self.reference.prefill['input_ids_hash'], 'prompt mismatch'
                assert self.prefill['grid_hash']==self.reference.prefill['grid_hash'], 'image grid mismatch'
            return None
        assert self.prefill is not None, 'decoder before prefill'
        assert kwargs.get('past_key_values') is not None, 'missing decoder cache'
        value = kwargs['inputs_embeds']
        assert value.ndim==3 and value.shape[:2]==(1,1), 'decoder shape'
        assert torch.isfinite(value).all(), 'nonfinite decoder input'
        position = kwargs['cache_position'].detach().cpu().clone()
        shape = tuple(kwargs['attention_mask'].shape)
        result = None
        if self.reference is not None:
            index = len(self.states)
            assert index<len(self.reference.states), 'tape exhausted'
            assert torch.equal(position,self.reference.positions[index]), 'cache position mismatch'
            assert shape==self.reference.mask_shapes[index], 'attention mask shape mismatch'
            original = self.reference.states[index]
            assert original.shape==value.shape and original.dtype==value.dtype, 'replay shape or dtype mismatch'
            replacement = original.to(value.device)
            self.input_drift_norms.append((value.float()-replacement.float()).norm().item())
            result = (args,dict(kwargs,inputs_embeds=replacement))
            value = replacement
        self.states.append(value.detach().cpu().clone())
        self.positions.append(position)
        self.mask_shapes.append(shape)
        return result

    def fingerprint(self):
        return hashlib.sha256(''.join(tensor_hash(v) for v in self.states).encode()).hexdigest()
