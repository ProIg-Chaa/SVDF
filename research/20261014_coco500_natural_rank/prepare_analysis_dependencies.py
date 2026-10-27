import urllib.request,zipfile,hashlib,json,time
from pathlib import Path
base=Path('/home/bingxing2/home/scx9ftv/gushuo/experiments/20261014_coco500_natural_rank/analysis_nltk_data')
items=['tokenizers/punkt_tab','taggers/averaged_perceptron_tagger_eng','corpora/wordnet']
result=[]
for name in items:
 path=base/(name+'.zip');path.parent.mkdir(parents=True,exist_ok=True)
 good=path.exists() and zipfile.is_zipfile(path)
 if not good:
  tmp=path.with_suffix('.download')
  url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/'+name+'.zip'
  with urllib.request.urlopen(url,timeout=60) as resp, tmp.open('wb') as out:
   while True:
    b=resp.read(65536)
    if not b:break
    out.write(b)
  assert zipfile.is_zipfile(tmp)
  tmp.replace(path)
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  if not name.startswith('corpora/'):z.extractall(path.parent)
 result.append({'name':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 print(json.dumps(result[-1]),flush=True)
(base/'DEPENDENCIES.json').write_text(json.dumps(result,indent=2))
