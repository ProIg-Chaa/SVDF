"""Frozen checkpoint, processor, dtype and explicit CUDA placement."""
import random
from pathlib import Path
import torch,base
from transformers import AutoProcessor,AutoTokenizer,Qwen2_5_VLForConditionalGeneration
from lead import get_math_symbols_ids
from scale_contract import model_map,dimensions

def load_model(spec,folder):
    assert torch.cuda.device_count()==spec['gpus']
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    base.sweep.patch_segmented_vision_sdpa()
    adapter=None
    if spec['gpus']==2:
        from cache_index_adapter import install_and_test
        adapter=install_and_test()
    placement=model_map(spec);path=Path(spec['path'])
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',**placement).eval()
    model.requires_grad_(False)
    assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 and not p.requires_grad for p in model.parameters())
    assert {p.device.index for p in model.parameters()}==set(range(spec['gpus']))
    processor_kwargs={} if spec['processor_max_pixels_override'] is None else {'max_pixels':spec['processor_max_pixels_override']}
    processor=AutoProcessor.from_pretrained(path,local_files_only=True,**processor_kwargs)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    assert processor.image_processor.max_pixels==spec['effective_processor_max_pixels']
    assert processor.image_processor.min_pixels==spec['processor_min_pixels']
    anchor=tokenizer.encode('</think>',add_special_tokens=False);assert anchor[0]==522 and tokenizer.decode(anchor[:1])=='</'
    shape=dimensions(model.config,model.get_input_embeddings().weight.shape,model.get_output_embeddings().weight.shape,len(tokenizer),spec)
    assert model.get_input_embeddings().weight.device==torch.device('cuda:0')
    assert model.get_output_embeddings().weight.device.type=='cuda'
    from state_isolation import StateIsolation
    StateIsolation(model)
    base.dump(folder/'runtime_identity.json',dict(model=str(path),anchor=anchor,torch=torch.__version__,devices={str(i):torch.cuda.get_device_name(i) for i in range(spec['gpus'])},
        processor_class=type(processor).__name__,model_class=type(model).__name__,num_threads=torch.get_num_threads(),num_interop_threads=torch.get_num_interop_threads(),
        image_token_id=model.config.image_token_id,vocab_size=shape[0],hidden_size=shape[1],device_map=placement,actual_device_map=model.hf_device_map,
        processor_max_pixels=processor.image_processor.max_pixels,processor_min_pixels=processor.image_processor.min_pixels,
        no_CPU_or_disk_offload=True,cache_index_adapter=adapter,geometry_capture='legacy_event_plus_1',logical_horizon=1024))
    ids=get_math_symbols_ids(tokenizer);math_ids=torch.tensor(list(ids),device=model.device) if ids else None
    return model,processor,tokenizer,math_ids
