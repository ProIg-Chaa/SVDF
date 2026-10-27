"""Synchronize and measure every allocated CUDA device, never CPU offload."""
import torch

def synchronize():
    for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)

def reset_peak_memory():
    for device in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(device)

def peak_memory():
    return {str(i):dict(allocated=torch.cuda.max_memory_allocated(i),reserved=torch.cuda.max_memory_reserved(i)) for i in range(torch.cuda.device_count())}

def peak_allocated():return sum(x['allocated'] for x in peak_memory().values())

def peak_reserved():return sum(x['reserved'] for x in peak_memory().values())
