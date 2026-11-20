import json
from pathlib import Path
import numpy as np,collections,math,re
from training.lsfb_cont.train_narrow_context import measure
root=Path('.runtime/lsfb-narrow-context-data-v1');read=lambda s:[json.loads(x) for x in (root/f'{s}.jsonl').read_text().splitlines()];tr,va=read('train'),read('val');result={}
for field in ['current_french','input']:
 def features(text):
  words=re.findall(r'\w+',text.casefold());return collections.Counter(word[i:i+n] for word in [' '+w+' ' for w in words] for n in (3,4,5) for i in range(len(word)-n+1))
 features_train=[features(r[field]) for r in tr];vocab=sorted({t for f in features_train for t in f});idf=np.array([math.log((1+len(tr))/(1+sum(t in f for f in features_train)))+1 for t in vocab])
 def vectorize(fs):
  x=np.array([[f[t] for t in vocab] for f in fs],dtype=float)*idf;return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 similarity=vectorize([features(r[field]) for r in va])@vectorize(features_train).T;ids=similarity.argmax(1);texts=[tr[i]['target_text'] for i in ids];result[field]=measure(va,texts)
 result[field]['predictions']=[{'id':r['id'],'retrieved_train_id':tr[i]['id'],'prediction':tr[i]['target']} for r,i in zip(va,ids)]
result['empty_output']=measure(va,['']*len(va));(root/'RETRIEVAL.json').write_text(json.dumps(result,indent=2));print({k:{x:y for x,y in v.items() if x!='predictions'} for k,v in result.items()})
