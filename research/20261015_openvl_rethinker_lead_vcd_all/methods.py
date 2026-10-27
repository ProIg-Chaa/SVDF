"""Frozen official LEAD and existing continuous VCD. No SVDF controller."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from adapters.lead_adapter import _compile_official, official_math_ids
from adapters.common_decode import prefill, advance
from adapters.vcd_adapter import add_diffusion_noise
CONFIG = dict(temperature=.6, top_k=20, top_p=.95, min_p=0., do_sample=False, max_new_tokens=1024)
LEAD_KW = dict(alpha_0=.6, beta_0=.7, window_size=256, max_switch_count=5,
    termination_max_tokens=32, convergence_words='</think>',
    termination_words='</think>\n\nThe final answer is')
VCD_KW = dict(vcd_alpha=1., vcd_beta=.1, noise_step=500, vcd_noise_seed=12345)
ARMS = ('full_lead', 'vcd_continuous')
class Methods:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.native = _compile_official()
        self.math_ids = torch.tensor(sorted(official_math_ids(tokenizer)), device=model.device, dtype=torch.long)
        source = Path(__file__).with_name('fair_worker_snapshot.py')
        tree = ast.parse(source.read_text(), filename=str(source))
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('choose','vcd_generate')]
        assert [n.name for n in selected] == ['choose','vcd_generate']
        exact = ast.Module(body=selected, type_ignores=[])
        ns = dict(eng=SimpleNamespace(torch=torch, model=model, tokenizer=tokenizer),
            CONFIG=CONFIG, prefill=prefill, advance=advance, add_diffusion_noise=add_diffusion_noise)
        exec(compile(exact,str(source),'exec'),ns)
        self.vcd = ns['vcd_generate']
    def generate(self, arm, inputs, state):
        length = inputs['input_ids'].shape[1]
        if arm == 'full_lead':
            out = self.native['generate_lead'](self.model,self.tokenizer,**inputs,**CONFIG,
                **LEAD_KW,math_ids_tensor=self.math_ids)
            return [int(x) for x in out[0,length:].tolist()]
        if arm == 'vcd_continuous':return self.vcd(inputs,state)
        raise ValueError(arm)
    def metadata(self, arm):
        if arm == 'full_lead':
            return dict(CONFIG, **LEAD_KW, upstream_commit='dd2651d2ce04abba0e77af803fad42177a51d7b1',
                official_unmodified=True, math_ids_source='official exact token set',
                native_answer_budget_preserved=True, svdf_threshold_used=False)
        return dict(CONFIG, **VCD_KW, upstream_commit='d6568ff81b8fd306a49e630df44f2db5c2300191',
            intervention='every generated step', initialization='ordinary hard',
            adaptation='Qwen2.5VL native fixed-grid dual-cache; greedy instead of upstream stochastic sampling',
            corruption='official sigmoid diffusion on prepared pixels; native grid unchanged',
            score_arithmetic='native dtype (1+alpha)*original-alpha*noise; original plausibility mask',
            implementation='exact AST of reviewed fair-worker choose and vcd_generate')
