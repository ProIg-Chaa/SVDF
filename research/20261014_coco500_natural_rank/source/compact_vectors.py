import hashlib,json,os
import torch
def save_vectors(payload,path):
    if os.environ.get('H2_STORE_VECTORS')=='1':
        return torch.save(payload,path)
    out={k:v for k,v in payload.items() if k!='probs'}
    out['probability_metadata']={k:None if v is None else dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k,v in payload['probs'].items()}
    out['dense_vectors_retained']=False
    with open(str(path)+'.metadata.json','x') as f:json.dump(out,f)
