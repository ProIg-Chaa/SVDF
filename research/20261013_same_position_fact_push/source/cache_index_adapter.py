"""Preserve historical 32B CPU batch-index adaptation; KV stays on CUDA."""
import torch

def install_and_test():
    from transformers.cache_utils import DynamicCache
    original=DynamicCache.batch_select_indices
    def select(cache,indices):return original(cache,indices.cpu())
    DynamicCache.batch_select_indices=select
    cache=DynamicCache()
    for device in range(torch.cuda.device_count()):
        value=torch.arange(12,device='cuda:'+str(device)).reshape(3,1,2,2)
        cache.update(value,value+100,device)
    cache.batch_select_indices(torch.tensor([2,0],device='cuda:0'))
    for key,value in zip(cache.key_cache,cache.value_cache):
        expected=torch.arange(12,device=key.device).reshape(3,1,2,2)[[2,0]]
        assert torch.equal(key,expected) and torch.equal(value,expected+100)
    return dict(passed=True,devices=torch.cuda.device_count(),CPU_indices_only=True,KV_on_CUDA=True)
