"""Full official LEAD, with an isolated route-only audit callback.

The archived upstream remains unmodified. Original uses its exact arithmetic,
controller, anchor, and convergence code. Random substitutes only the continuous
route increment relative to E[selected_token]. Zero is a baseline bypass.
"""
import ast
import hashlib
from pathlib import Path
import torch
from .common_decode import generate_baseline, token_record, GeometryUnavailable

UPSTREAM_COMMIT = "dd2651d2ce04abba0e77af803fad42177a51d7b1"
UPSTREAM_SHA256 = "b24b75ac404a841161b15cc2e343276b5a928a0d448654125e19bc2f632e7db1"
UPSTREAM_PATH = Path(__file__).resolve().parent.parent / "upstream/lead/lead/generation_utils.py"

def _compile_official(callback=None):
    source = UPSTREAM_PATH.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError("Official LEAD source checksum mismatch")
    tree = ast.parse(source, filename=str(UPSTREAM_PATH))
    if callback is not None:
        found = 0
        class InjectCallback(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                nonlocal found
                if node.name != "generate_lead":
                    return node
                self.generic_visit(node)
                return node
            def visit_Assign(self, node):
                nonlocal found
                if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "last_emb" and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "torch" and node.value.func.attr == "where"):
                    found += 1
                    added = ast.parse("last_emb = __lead_audit_callback(last_emb, locals())").body[0]
                    return [node, ast.copy_location(added, node)]
                return node
        tree = InjectCallback().visit(tree)
        if found != 1:
            raise RuntimeError("Expected exactly one official route assignment")
        ast.fix_missing_locations(tree)
    namespace = {"__lead_audit_callback":callback, "__name__":"official_lead_audit"}
    exec(compile(tree, str(UPSTREAM_PATH), "exec"), namespace)
    return namespace

def official_math_ids(tokenizer):
    """Return the exact official token set, independent of historical production."""
    return _compile_official()["get_math_symbols_ids"](tokenizer)

def _kwargs(tokenizer, inputs, config, context):
    values = dict(inputs)
    values.update({k:config[k] for k in
                   ("temperature","top_k","top_p","min_p","max_new_tokens","do_sample")
                   if k in config})
    values.setdefault("max_new_tokens", 1024)
    values.setdefault("temperature", 0.6)
    values.setdefault("top_k", 20)
    values.setdefault("top_p", 0.95)
    values.setdefault("do_sample", False)
    values["alpha_0"] = float(config.get("alpha_0", 0.6))
    values["beta_0"] = float(config.get("beta_0", 0.7))
    values["window_size"] = int(config.get("window_size", 256))
    values["max_switch_count"] = config.get("max_switch_count", 5)
    values["termination_max_tokens"] = int(config.get("termination_max_tokens", 32))
    values["convergence_words"] = config.get("convergence_words", "</think>")
    values["termination_words"] = config.get(
        "termination_words", "</think>"+chr(10)+chr(10)+"The final answer is")
    official_ids = official_math_ids(tokenizer)
    supplied = (context or {}).get("math_ids")
    if supplied is not None:
        supplied_set = set(int(x) for x in (
            supplied.detach().cpu().tolist() if torch.is_tensor(supplied) else supplied))
        if supplied_set != official_ids:
            raise RuntimeError("Supplied math IDs disagree with the official LEAD set")
    values["math_ids_tensor"] = (torch.tensor(sorted(official_ids),
        device=inputs["input_ids"].device, dtype=torch.long) if official_ids else None)
    return values

def _random_route(hard, official_route, generator):
    """Match the actually representable official L2 radius after quantization."""
    hard32 = hard.float()
    target = torch.linalg.vector_norm((official_route.float()-hard32).reshape(-1))
    if not torch.isfinite(target):
        raise RuntimeError("Nonfinite official LEAD increment")
    if float(target.item()) == 0.0:
        return hard, 0.0
    direction = torch.randn(hard.shape, generator=generator, device=hard.device,
                            dtype=torch.float32)
    direction /= torch.linalg.vector_norm(direction.reshape(-1)).clamp_min(1e-30)
    scale = target.clone()
    best = None
    best_error = float("inf")
    for _ in range(8):
        candidate = (hard32 + scale*direction).to(hard.dtype)
        actual = torch.linalg.vector_norm((candidate.float()-hard32).reshape(-1))
        error = float(torch.abs(actual-target).item())
        if error < best_error:
            best, best_error = candidate, error
        if float(actual.item()) == 0:
            scale *= 2
        else:
            scale *= target/actual
        if error <= 1e-4 * float(target.item()):
            break
    relative = best_error/max(float(target.item()), 1e-30)
    if not torch.isfinite(best).all():
        raise RuntimeError("Nonfinite randomized LEAD input")
    if relative > 0.05:
        raise GeometryUnavailable("Random LEAD quantized norm mismatch exceeds frozen 5%")
    return best, relative

class _ForwardAudit:
    """Observe the tensor consumed in the next official model call."""
    def __init__(self, model):
        self.model, self.pending = model, None
        self.forward_count = 0
    def __getattr__(self, name):
        return getattr(self.model, name)
    def __call__(self, *args, **kwargs):
        if self.pending is not None:
            hard, expected, row = self.pending
            actual = kwargs.get("inputs_embeds")
            if actual is None or not torch.equal(actual[:,0,:], expected):
                raise RuntimeError("LEAD feedback was not consumed exactly once as prepared")
            delta = actual[:,0,:].float()-hard.float()
            actual_norm = float(torch.linalg.vector_norm(delta.reshape(-1)).item())
            row.update({"consumed":True, "consumer_step":row["step"]+1,
                        "actual_consumed_increment_norm":actual_norm,
                        "actual_input_nonzero":bool(torch.count_nonzero(delta).item()),
                        "actual_nonzero_elements":int(torch.count_nonzero(delta).item()),
                        "actual_input_finite":bool(torch.isfinite(actual).all().item())})
            if not row["actual_input_finite"]:
                raise RuntimeError("Nonfinite consumed LEAD input")
            self.pending = None
        self.forward_count += 1
        return self.model(*args, **kwargs)

@torch.inference_mode()
def generate_official_reference(model, tokenizer, inputs, *, seed=0, config=None, context=None):
    """Unmodified official implementation for GPU/native equivalence checks."""
    config, context = dict(config or {}), dict(context or {})
    native = _compile_official()["generate_lead"]
    out = native(model, tokenizer, **_kwargs(tokenizer,inputs,config,context))
    return {"token_ids":out[0,inputs["input_ids"].shape[1]:].tolist(),
            "trace":[], "metadata":{"method":"lead_official_unmodified",
            "upstream_commit":UPSTREAM_COMMIT,"source_sha256":UPSTREAM_SHA256}}

@torch.inference_mode()
def generate(model, tokenizer, inputs, *, variant="original", seed=0, config=None, context=None):
    """Generate one full trajectory using original or randomized LEAD updates."""
    config, context = dict(config or {}), dict(context or {})
    if variant == "zero":
        result = generate_baseline(model, tokenizer, inputs, seed=seed,config=config,context=context)
        result["metadata"].update({"method":"lead", "variant":"zero",
            "zero_semantics":"baseline bypass; not a same-controller equivalence test"})
        return result
    if variant not in ("original","random"):
        raise ValueError("Unknown LEAD variant")
    generator = torch.Generator(device=inputs["input_ids"].device)
    generator.manual_seed(int(seed))
    trace = []
    audited_model = _ForwardAudit(model)
    def callback(route, local):
        token, step = local["next_tokens"], int(local["step"])
        hard = local["E"][token]
        original_delta = route.float()-hard.float()
        original_norm = float(torch.linalg.vector_norm(original_delta.reshape(-1)).item())
        if not torch.isfinite(route).all() or not torch.isfinite(original_delta).all():
            raise RuntimeError("Nonfinite official LEAD route")
        if variant == "random":
            final, relative_error = _random_route(hard, route, generator)
        else:
            final, relative_error = route, 0.0
        actual_norm = float(torch.linalg.vector_norm((final.float()-hard.float()).reshape(-1)).item())
        row = token_record(step, token, local["logits_original"],
                           local["probs_original"], local["probs"])
        to_normal = bool(local["to_normal"][0].item()) if step > 0 else False
        to_soft = bool(local["to_soft"][0].item()) if step > 0 else False
        row.update({"method":"lead", "variant":variant,
                    "mode":"soft" if bool(local["is_soft"][0].item()) else "normal",
                    "controller_mode":int(local["mode"][0].item()),
                    "to_normal":to_normal,"to_soft":to_soft,
                    "locked_normal":bool(local["locked_normal_mask"][0].item()),
                    "entropy":float(local["cur_entropy"][0].item()),
                    "reference_entropy":float(local["cur_ref_entropy"][0].item()),
                    "switch_count":int(local["switch_count"][0].item())
                        if local["max_switch_count"] is not None else None,
                    "alpha":float(local["alpha"]),"beta":float(local["beta"]),
                    "original_increment_norm":original_norm,
                    "prepared_increment_norm":actual_norm,
                    "quantized_norm_relative_error":relative_error,
                    "quantized_norm_absolute_error":abs(actual_norm-original_norm),
                    "native_filtered_argmax_id":int(torch.argmax(local["probs"],dim=-1)[0].item()),
                    "forced_token_differs_from_native_choice":
                        int(token.item()) != int(torch.argmax(local["probs"],dim=-1)[0].item()),
                    "consumed":False,"actual_input_nonzero":False})
        trace.append(row)
        audited_model.pending = (hard,final,row)
        return final
    native = _compile_official(callback)["generate_lead"]
    out = native(audited_model,tokenizer,**_kwargs(tokenizer,inputs,config,context))
    token_ids = out[0,inputs["input_ids"].shape[1]:].tolist()
    if len(token_ids) != len(trace):
        raise RuntimeError("LEAD trace/token length mismatch")
    consumed = [r for r in trace if r["consumed"]]
    active = [r for r in consumed if r["actual_input_nonzero"]]
    return {"token_ids":token_ids,"trace":trace,"metadata":{
        "method":"lead","variant":variant,"seed":int(seed),
        "upstream_commit":UPSTREAM_COMMIT,"source_sha256":UPSTREAM_SHA256,
        "implementation":"official AST route callback; upstream source unchanged",
        "randomized_component":"full continuous feedback minus current selected hard embedding",
        "random_direction_refresh":"each current nonzero update",
        "norm_definition":"L2 of actually represented feedback minus E[selected], FP32 measurement",
        "norm_relative_tolerance":0.05,"selection_dtype":"native",
        "n_generated":len(token_ids),"n_forwards":audited_model.forward_count,
        "n_consumed_feedback":len(consumed),"n_nonzero_interventions":len(active),
        "max_norm_relative_error":max((r["quantized_norm_relative_error"] for r in active),default=0),
        "n_to_soft":sum(r["to_soft"] for r in trace),
        "n_to_normal":sum(r["to_normal"] for r in trace),
        "reached_eos":bool(token_ids and tokenizer.eos_token_id is not None and token_ids[-1]==tokenizer.eos_token_id),
        "truncated":len(token_ids)==int(config.get("max_new_tokens",1024)) and not (token_ids and tokenizer.eos_token_id is not None and token_ids[-1]==tokenizer.eos_token_id),
        "stop_reason":"eos" if token_ids and tokenizer.eos_token_id is not None and token_ids[-1]==tokenizer.eos_token_id else ("horizon" if len(token_ids)==int(config.get("max_new_tokens",1024)) else "native_answer_budget"),
        "official_defaults":{"alpha_0":0.6,"beta_0":0.7,"window_size":256,
            "max_switch_count":5,"a":1.0,"b1":1.0,"b2":0.2,"termination_max_tokens":32},
        "anchor_semantics":"input embedding of image_pad; end_think resolves first native subtoken"}}
