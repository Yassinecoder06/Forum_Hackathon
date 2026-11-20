"""Audit feature loss caused by reducing UD FSL CoNLL-U to FORM-only targets."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/text_to_gloss/raw/stk/fsl_stkautogramm-ud-test.conllu'
OUT=ROOT/'reports/private/fsl-construction-audit-v1'
FEATURES=('NMM','Exp','Exp[lex]','Trs','Gaze','GazeHeld','Src','Tgt','Loc','PT','PTReal','ST','STReal','NDH','NDHHeld','DHHeld','TwoH','Rep','Reported')
def parse(path):
 sentences=[];current=None
 for line in path.read_text().splitlines()+['']:
  if line.startswith('# sent_id = '):
   if current:sentences.append(current)
   current={'id':line.split(' = ',1)[1],'tokens':[],'meta':{}}
  elif current and line.startswith('# ') and ' = ' in line:
   k,v=line[2:].split(' = ',1);current['meta'][k]=v
  elif current and line and not line.startswith('#'):
   c=line.split('\t')
   if c[0].isdigit():
    feats={} if c[5]=='_' else dict(x.split('=',1) for x in c[5].split('|'))
    misc={} if c[9]=='_' else dict(x.split('=',1) for x in c[9].split('|'))
    current['tokens'].append({'form':c[1],'lemma':c[2],'upos':c[3],'feats':feats,'misc':misc})
  elif current:
   sentences.append(current);current=None
 return sentences

def main():
 ss=parse(RAW);assert len(ss)==108
 split={}
 for s in ['train','val']:
  for r in map(json.loads,(ROOT/f'data/parallel/fsl_ud_feasibility_{s}.jsonl').read_text().splitlines()):split[r['id']]=s
 counts={s:collections.Counter() for s in ['train','val','test']};sentences={s:0 for s in counts};total={s:0 for s in counts}; examples=[]
 for sent in ss:
  key='stk:'+sent['id'];s=split.get(key,'test');sentences[s]+=1;total[s]+=len(sent['tokens'])
  form_features=[]
  for t in sent['tokens']:
   forms={k:v for k,v in t['feats'].items() if k in FEATURES};forms.update({k:v for k,v in t['misc'].items() if k in FEATURES});
   for k,v in forms.items():counts[s][k]+=1
   if forms:form_features.append({'form':t['form'],'features':forms})
  examples.append({'id':sent['id'],'split':s,'french':sent['meta'].get('text_fr',''),'token_count':len(sent['tokens']),'feature_tokens':form_features})
 train=set(counts['train']);val=set(counts['val']);result={'status':'complete','sign_language':'fsl','source_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'counts':{s:{'sentences':sentences[s],'tokens':total[s],'feature_occurrences':sum(counts[s].values()),'feature_types':dict(counts[s])} for s in counts},'validation_feature_types_unseen_in_train':sorted(val-train),'validation_feature_occurrences_by_type_unseen_in_train':{k:counts['val'][k] for k in sorted(val-train)},'target_projection':'FORM only; token FEATS and MISC modality/spatial markers excluded from data/parallel gloss_sequence','gold_translation_pairs':0,'test_evaluation_opened':False,'linguistic_review':False,'interpretation':'Reports annotation coverage and information discarded by the target projection; it does not define a new target convention or prove which cues are obligatory for each segment.'}
 (OUT/'source_feature_coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(OUT/'source_feature_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
