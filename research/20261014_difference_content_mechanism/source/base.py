import hashlib,json
from pathlib import Path
import torch
import sweep
from lead.generation_utils import generate_lead
INPUT_KEYS=('id','dataset','question','options','image','image_sha256','pair_id')
BRANCHES=('talr','existing_residual','bounded_reweight','norm_matched_random')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def append(path, row):
    with path.open('a') as handle:
        handle.write(json.dumps(row) + '\n')
        handle.flush()


def public_input(row):
    return {key: row[key] for key in INPUT_KEYS if key in row}


def controller(threshold):
    return dict(refinement_entropy_threshold=threshold, refinement_threshold_mode='fixed',
        refinement_online_quantile=.75, refinement_online_jump_multiplier=1.,
        refinement_window=8, refinement_soft_cap=2)


class StopAtEvent(Exception):
    pass


class EventTrace(list):
    def append(self, row):
        super().append(row)
        if row['lead_refinement_active']:
            raise StopAtEvent()


class CaptureStop(list):
    def append(self, row):
        super().append(row)
        raise StopAtEvent()


def prepare(processor, sample, image_path, device):
    inputs, length = sweep.prepare(processor, public_input(sample), image_path, device)
    assert 'pixel_values' in inputs and inputs['pixel_values'].numel() > 0, 'raw image missing'
    assert 'image_grid_thw' in inputs and inputs['image_grid_thw'].numel() > 0
    return inputs, length


def discover(model, processor, tokenizer, sample, math_ids, threshold):
    inputs, length = prepare(processor, sample, sample['image'], model.device)
    trace = EventTrace()
    kwargs = sweep.talr_kwargs(math_ids, 1024, **controller(threshold))
    kwargs.update(token_trace=trace, trace_topk=0)
    output = None
    try:
        with torch.no_grad():
            output = generate_lead(model, tokenizer, **inputs, **kwargs)[0]
    except StopAtEvent:
        pass
    event = trace[-1]['step'] if trace and trace[-1]['lead_refinement_active'] else -1
    prefix = [row['token_id'] for row in trace]
    assert [row['step'] for row in trace] == list(range(len(trace)))
    if event >= 0:
        assert len(prefix) == event + 1
    else:
        prefix = [int(x) for x in output[length:].tolist()]
    return event, prefix, list(trace)


def capture(model, processor, tokenizer, sample, image, prefix, event, math_ids, threshold, stop=True):
    inputs, _ = prepare(processor, sample, image, model.device)
    vectors, logits = CaptureStop() if stop else [], []
    kwargs = sweep.talr_kwargs(math_ids, 1024, **controller(threshold))
    kwargs.update(forced_prefix_ids=prefix, trace_soft_vector_collector=vectors,
        trace_capture_soft_vector_step=event, capture_logits_steps=[event], capture_logits_sink=logits)
    try:
        with torch.no_grad():
            generate_lead(model, tokenizer, **inputs, **kwargs)
    except StopAtEvent:
        assert stop
    probabilities = [r['probs'] for r in logits if r['kind']=='logits' and r['step']==event]
    assert len(vectors) == len(probabilities) == 1
    result = vectors[0]
    assert result['step'] == event and result['selected_token_id'] == prefix[event]
    result['probs'] = probabilities[0]
    return result


def assert_capture_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key],right[key]) if torch.is_tensor(left[key]) else left[key]==right[key], key


def continuation(model, processor, tokenizer, sample, math_ids, threshold, prefix=None, event=-1, action=None):
    return sweep.run_talr(model, processor, tokenizer, public_input(sample), math_ids, 1024,
        forced_prefix_ids=prefix, override_step=event, external_residual=action,
        residual_strength=1., residual_duration=4, **controller(threshold))

