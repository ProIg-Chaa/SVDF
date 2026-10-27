"""Public selection and exact Qwen2.5-VL model contracts, without model calls."""
import hashlib
from collections import Counter

PUBLIC_KEYS={'dataset','id','pair_id','question','options','image','image_sha256'}

def key(row):return row['dataset']+':'+str(row['id'])

def validate_public(rows,require_pairs=True):
    assert rows and all(set(r)==PUBLIC_KEYS for r in rows)
    assert len({key(r) for r in rows})==len(rows)
    assert all(r['dataset'] for r in rows)
    pairs=Counter(str(r['pair_id']) for r in rows if r['dataset']=='mmvp')
    assert not require_pairs or not pairs or set(pairs.values())=={2},'MMVP pairs must be complete'

def unit(row):
    return row['dataset']+':pair:'+str(row['pair_id']) if row['dataset']=='mmvp' else key(row)

def select_rows(rows,phase,shard_index=0,shard_count=1):
    validate_public(rows,require_pairs=phase!='smoke');assert shard_count>0 and 0<=shard_index<shard_count
    chosen=[r for r in rows if int(hashlib.sha256(unit(r).encode()).hexdigest(),16)%shard_count==shard_index]
    assert chosen,'Empty shard is not a run'
    if phase=='smoke':
        chosen=chosen[:2]
        assert len(chosen)==2,'Engineering smoke requires two public samples'
    else:assert phase=='full'
    validate_public(chosen,require_pairs=phase!='smoke');return chosen

def model_map(spec):
    assert spec['architecture']==['Qwen2_5_VLForConditionalGeneration'] and spec['model_type']=='qwen2_5_vl'
    if spec['gpus']==1:return dict(device_map={'':'cuda:0'})
    assert spec['gpus']==2 and spec['num_hidden_layers']==64 and spec['hidden_size']==5120
    return dict(device_map='auto',max_memory={0:'32GiB',1:'36GiB'})

def dimensions(config,in_shape,out_shape,tokenizer_size,spec):
    for name in ('vocab_size','hidden_size','num_hidden_layers','num_key_value_heads'):
        assert int(getattr(config,name))==spec[name],name
    expected=(spec['vocab_size'],spec['hidden_size'])
    assert tuple(in_shape)==tuple(out_shape)==expected
    assert 0<tokenizer_size<=spec['vocab_size']
    return expected

def assert_disjoint_metadata(signatures):
    pointers=set();cache_ids=set()
    for signature in signatures:
        assert signature['cache_id'] not in cache_ids;cache_ids.add(signature['cache_id'])
        current={(v['device'],v['pointer']) for v in signature['storage']}
        assert not pointers.intersection(current),'Native KV storage aliases on the same device'
        pointers.update(current)
