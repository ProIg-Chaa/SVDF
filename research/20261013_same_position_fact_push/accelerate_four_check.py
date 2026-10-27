from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parent
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((E/'ACCEL4_MANIFEST.json').read_text())
assert h(E/'manifest.json')==a['generation_manifest_sha256']
assert h(E/'positions.jsonl')==a['positions_sha256']
assert h(E/'continuation_manifest.json')==a['prior_continuation_manifest_sha256']
for n,s in a['files_sha256'].items():assert h(E/n)==s,n
approval=json.loads((E/'ACCEL4_APPROVED.json').read_text());assert approval['manifest_sha256']==h(E/'ACCEL4_MANIFEST.json') and approval['root_review_passed']
print('ACCEL4_HASHES_PASSED',flush=True)
