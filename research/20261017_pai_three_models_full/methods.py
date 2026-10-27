"""Official PAI 20261013 Qwen2.5VL adaptation, unchanged arithmetic."""
import torch
import pai_adapter as canonical
from adapters.lead_adapter import _compile_official
CONFIG=dict(temperature=.6,top_k=20,top_p=.95,min_p=0.,do_sample=False,max_new_tokens=1024)
PAI_KW=dict(alpha=.2,gamma=1.1,beta=.1,maximum=1024)
class Methods:
    def __init__(self,model,tokenizer):
        self.model,self.tokenizer=model,tokenizer
        self.native=_compile_official()
    def generate(self,inputs,state,zero=False):
        old_pre,old_advance=canonical.branch_prefill,canonical.branch_advance
        ids={}
        def pre(model,prepared):
            branch='main' if 'pixel_values' in prepared else 'weak'
            state['branch']=branch
            out,cache=old_pre(model,prepared)
            ids[id(cache['past_key_values'])]=branch
            return out,cache
        def advance(model,token,cache):
            state['branch']=ids[id(cache['past_key_values'])]
            out,nxt=old_advance(model,token,cache)
            ids[id(nxt['past_key_values'])]=state['branch']
            return out,nxt
        canonical.branch_prefill,canonical.branch_advance=pre,advance
        try:
            return canonical.generate(self.model,self.tokenizer,inputs,config=CONFIG,
                **(dict(PAI_KW,alpha=0.,gamma=1.) if zero else PAI_KW))
        finally:canonical.branch_prefill,canonical.branch_advance=old_pre,old_advance
