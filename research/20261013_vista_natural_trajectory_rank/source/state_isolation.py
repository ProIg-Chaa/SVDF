"""Snapshot mutable model positional state and global RNG between parked branches."""
import hashlib,random
import numpy as np
import torch
from multi_device import synchronize
from scale_contract import assert_disjoint_metadata


def tensor_sha(x):
    return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def copy_value(value):
    if torch.is_tensor(value):return value.detach().clone()
    assert value is None or isinstance(value,(bool,int,float,str))
    return value


class StateIsolation:
    attrs=('inv_freq','original_inv_freq','max_seq_len_cached','attention_scaling')
    def __init__(self,model):
        self.model=model
        self.rotary=[(n,m) for n,m in model.named_modules() if hasattr(m,'rope_type')]
        assert self.rotary and all('dynamic' not in m.rope_type for _,m in self.rotary)
        assert hasattr(model,'rope_deltas')
        assert all(all(hasattr(m,a) for a in self.attrs) for _,m in self.rotary)

    def snapshot(self):
        synchronize()
        return dict(rope_deltas=copy_value(self.model.rope_deltas),
            rotary=[{a:copy_value(getattr(m,a)) for a in self.attrs} for _,m in self.rotary],
            torch_rng=torch.get_rng_state().clone(),cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()],
            python_rng=random.getstate(),numpy_rng=np.random.get_state())

    def restore(self,state):
        self.model.rope_deltas=copy_value(state['rope_deltas'])
        for (_,m),values in zip(self.rotary,state['rotary']):
            for a,value in values.items():setattr(m,a,copy_value(value))
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])

    def assert_equal(self,state):
        actual=self.snapshot()
        def equal(a,b):
            if torch.is_tensor(a):assert torch.equal(a,b)
            elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(tuple,list)):
                assert len(a)==len(b)
                for x,y in zip(a,b):equal(x,y)
            else:assert a==b
        equal(actual,state)

    def restore_checked(self,state):
        self.restore(state);self.assert_equal(state)


def cache_signature(frame):
    live=frame.f_locals;cache=live['past_key_values']
    assert cache is not None and hasattr(cache,'key_cache') and hasattr(cache,'value_cache')
    keys=list(cache.key_cache);values=list(cache.value_cache)
    assert keys and len(keys)==len(values)
    tensors=keys+values
    assert all(torch.is_tensor(x) and x.device.type=='cuda' for x in tensors)
    return dict(cache_type=type(cache).__name__,cache_id=id(cache),sequence_length=int(cache.get_seq_length()),
        storage=[dict(device=str(x.device),pointer=x.untyped_storage().data_ptr(),shape=list(x.shape),dtype=str(x.dtype),version=x._version) for x in tensors],
        step=int(live['step']),cache_position=live['cache_position'].detach().cpu().tolist(),
        generated_lengths=[len(x)-n for x,n in zip(live['all_generated'],live['prompt_lens'])])


def assert_disjoint(signatures):
    assert_disjoint_metadata(signatures)
