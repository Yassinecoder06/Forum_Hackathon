"""Source-backed GeoQuery graph audit and training-only retrieval; no gloss conversion."""
import json,re,collections,math,hashlib
from pathlib import Path
import numpy as np

SOURCE=Path('/home/jmal/Dataset/geoquery-lsfb-source-v1')
PREPARED=Path('/home/jmal/Dataset/geoquery-lsfb-usable-v1')
RELATIONS={'adjacent','start-coincides','end-coincides','during'}
ARTICULATIONS={'left-hand-articulation','right-hand-articulation','two-hand-articulation'}

def validate_graph(graph):
 if not isinstance(graph,list) or not graph:raise ValueError('Expected nonempty author graph')
 ids={};edges=[]
 for triple in graph:
  if not isinstance(triple,list) or len(triple)!=3 or any(not isinstance(x,str) for x in triple):raise ValueError('Invalid triple')
  relation,a,b=triple
  if relation in ARTICULATIONS:
   if a in ids:raise ValueError('Repeated articulation ID')
   ids[a]=(relation,b)
  elif relation in RELATIONS:edges.append((relation,a,b))
  else:raise ValueError('Unknown relation')
 for relation,a,b in edges:
  if a not in ids or b not in ids:raise ValueError('Dangling temporal relation')
 return {'articulations':len(ids),'temporal_relations':len(edges)}

def normalize_query(query):return re.sub(r"'[^']*'","'ENTITY'",query)
def features(text,word=False):
 words=re.findall(r'\w+',text.casefold())
 if word:return collections.Counter(words+[' '.join(words[i:i+2]) for i in range(len(words)-1)])
 return collections.Counter(w[i:i+n] for w in [' '+x+' ' for x in words] for n in (3,4,5) for i in range(len(w)-n+1))

def retrieve(train,rows,word=False):
 fs=[features(r['french'],word) for r in train];vocab=sorted({t for f in fs for t in f});idf=np.array([math.log((1+len(train))/(1+sum(t in f for f in fs)))+1 for t in vocab])
 def vectorize(fs):
  x=np.array([[f[t] for t in vocab] for f in fs],dtype=float)*idf;return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 scores=vectorize([features(r['french'],word) for r in rows])@vectorize(fs).T
 return scores.argmax(1).tolist()

def score(refs,preds):
 if len(refs)!=len(preds) or not refs:raise ValueError('Prediction count mismatch')
 exact=graph_exact=tp=fp=fn=art_tp=art_fp=art_fn=invalid_refs=invalid_preds=0
 for ref,pred in zip(refs,preds):
  try:validate_graph(ref)
  except ValueError:invalid_refs+=1
  try:validate_graph(pred)
  except ValueError:invalid_preds+=1
  a={tuple(x) for x in ref};b={tuple(x) for x in pred};exact+=ref==pred;graph_exact+=a==b;tp+=len(a&b);fp+=len(b-a);fn+=len(a-b)
  aa={x for x in a if x[0] in ARTICULATIONS};bb={x for x in b if x[0] in ARTICULATIONS};art_tp+=len(aa&bb);art_fp+=len(bb-aa);art_fn+=len(aa-bb)
 return {'examples':len(refs),'serialized_graph_exact':exact/len(refs),'triple_set_exact':graph_exact/len(refs),'triple_micro_f1':2*tp/max(1,2*tp+fp+fn),'articulation_micro_f1':2*art_tp/max(1,2*art_tp+art_fp+art_fn),'invalid_reference_graphs':invalid_refs,'invalid_prediction_graphs':invalid_preds,'all_graphs_structurally_valid':invalid_refs+invalid_preds==0,'interpretation':'lexical/graph overlap only; not signing comprehension'}

def main():
 out=Path('research/private/geoquery-lsfb-baseline-v1');out.mkdir(exist_ok=True,parents=True)
 if (out/'AUDIT_AND_RESULTS.json').exists():raise ValueError('Completed baseline exists')
 provenance=json.loads((SOURCE/'PROVENANCE.json').read_text());prep=json.loads((PREPARED/'PREPARATION.json').read_text())
 for name,h in prep['files'].items():assert hashlib.sha256((PREPARED/name).read_bytes()).hexdigest()==h
 data={s:[json.loads(x) for x in (PREPARED/f'human_{s}.jsonl').read_text().splitlines()] for s in ['train','val']};train,val=data['train'],data['val']
 assert all(r['language']=='LSFB' and not r['synthetic'] for rs in data.values() for r in rs)
 assert not {r['template_type'] for r in train}&{r['template_type'] for r in val}
 assert not {normalize_query(r['geo_funql']) for r in train}&{normalize_query(r['geo_funql']) for r in val}
 graph_counts=collections.Counter();french_collisions=[];invalid_graphs=[]
 for s,rows in data.items():
  mapping=collections.defaultdict(set)
  for r in rows:
   try:q=validate_graph(r['lsfb_graph']);graph_counts.update(q)
   except ValueError as e:invalid_graphs.append({'split':s,'id':r['id'],'error':str(e)})
   mapping[' '.join(re.findall(r'\w+',r['french'].casefold()))].add(json.dumps(r['lsfb_graph']))
  french_collisions.append({'split':s,'conflicting_input_groups':sum(len(x)>1 for x in mapping.values())})
 references=[r['lsfb_graph'] for r in val];metrics={};predictions={}
 for name,word in [('char_tfidf',False),('word_tfidf',True)]:
  ids=retrieve(train,val,word);graphs=[train[i]['lsfb_graph'] for i in ids];metrics[name]=score(references,graphs);predictions[name]=[{'id':r['id'],'selected_train_id':train[i]['id'],'reference':r['lsfb_graph'],'prediction':g} for r,i,g in zip(val,ids,graphs)]
 train_hns={x[2] for r in train for x in r['lsfb_graph'] if x[0] in ARTICULATIONS};val_hns=[x[2] for r in val for x in r['lsfb_graph'] if x[0] in ARTICULATIONS]
 audit={'source':provenance,'counts':{s:len(rs) for s,rs in data.items()},'sign_language':'LSFB','source_french':'DeepL translation from original English per sourceREADME','signed_originals':'human native signer recording and HamNoSys transcription per source','local_review':'pending','augmentation_used':False,'split':'template-type disjoint; verified zero entity-normalizedFunQLstructure overlap','single_signer':'no signer-independent claim','train_hamnosys_types':len(train_hns),'validation_articulation_occurrences':len(val_hns),'validation_unseen_hamnosys_occurrences':sum(x not in train_hns for x in val_hns),'conflicting_french_groups':french_collisions,'graphs_checked':250,'invalid_graphs':invalid_graphs,'graphs_validated':250-len(invalid_graphs),'metrics':metrics,'production_ready':False}
 (out/'AUDIT_AND_RESULTS.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));(out/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2));Path('research/text-to-gloss/geoquery-lsfb/GEOQUERY_LSFB_BASELINE.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='source'},indent=2))
 # Correct interpretation of previously staged synthetic flag without overwriting that package.
 for s,rows in data.items():
  path=out/f'annotated_{s}.jsonl';path.write_text(''.join(json.dumps({**r,'french_transcription_source':'DeepL machine translation','signed_annotation_source':'original human recording/transcription','source_recording_group':r['elan_ref'],'local_linguistic_review':'pending'},ensure_ascii=False)+'\n' for r in rows))
if __name__=='__main__':main()
