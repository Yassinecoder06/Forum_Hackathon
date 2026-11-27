"""Matched developmental context ablation; paired errors, no significance claim."""
import json,hashlib,shutil
from pathlib import Path
from urllib.parse import unquote
from .scoring import distance

def main():
 root=Path('.runtime/lsfb-narrow-context-data-v1');runs={'context':Path('.runtime/lsfb-narrow-context-run-v1'),'current':Path('.runtime/lsfb-narrow-current-run-v1')};rows=[json.loads(x) for x in (root/'val.jsonl').read_text().splitlines()];preds={};metrics={};manifests={}
 for name,path in runs.items():
  result=json.loads((path/'RESULT.json').read_text());assert result['status']=='complete';metrics[name]=result;manifests[name]=json.loads((path/'MANIFEST.json').read_text());preds[name]={r['id']:[unquote(t) for t in r['prediction'].split()] for r in json.loads((path/'validation_predictions.json').read_text())}
 for name in runs:assert set(preds[name])=={r['id'] for r in rows},'Prediction IDs differ from evaluation rows'
 # Same data, initialization seed, precision, hyperparameters, labels and length budgets.
 for key in ['model','revision','data_hash','seed','precision','train','val','epochs','batch_size','accumulation','planned_steps','source_limit','target_limit']:
  assert manifests['context'][key]==manifests['current'][key],key
 retrieval=json.loads((root/'RETRIEVAL.json').read_text());paired=[]
 for r in rows:
  record={'id':r['id'],'conversation_id':r['conversation_id'],'reference':r['target'],'length':len(r['target']),'current_french':r['current_french']}
  for name in runs:record[name]={'prediction':preds[name][r['id']],'edits':distance(r['target'],preds[name][r['id']]),'exact':r['target']==preds[name][r['id']]}
  paired.append(record)
 comparison={'task':'previously inspected narrow LSFB development; not independent test','matched_fields_checked':True,'context':metrics['context'],'current':metrics['current'],'retrieval':{k:{x:y for x,y in v.items() if x!='predictions'} for k,v in retrieval.items()},'paired_counts':{key:sum(r['context']['exact']==a and r['current']['exact']==b for r in paired) for key,a,b in [('both_correct',True,True),('context_only_correct',True,False),('current_only_correct',False,True),('both_wrong',False,False)]},'reviewed':False}
 out=Path('research/private/lsfb-context-ablation-v1');out.mkdir(exist_ok=False,parents=True);(out/'paired_predictions.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2));(out/'RESULT.json').write_text(json.dumps(comparison,indent=2));Path('research/text-to-gloss/lsfb/LSFB_CONTEXT_ABLATION_RESULT.json').write_text(json.dumps(comparison,indent=2))
 backup=Path('/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-narrow-context-v1/current-only-run');shutil.copytree(runs['current'],backup);(backup/'SHA256.json').write_text(json.dumps({str(p.relative_to(backup)):hashlib.sha256(p.read_bytes()).hexdigest() for p in backup.rglob('*') if p.is_file()},indent=2))
 p=Path('research/RESULTS.md')
 with p.open('a') as f:
  f.write('\n## Matched context ablation completed\n\nSame145train/23development examples,seed84,T5-base rank16LoRA,25epochs/475steps,BF16,batch2/accumulation4,LR2e-4schedule,plain lexical labels and validation-WER selection. Current-only removes preceding French dialogue;task prefix and current French remain. This is the only intended input change. Each arm selects its own best checkpoint under the same rule. No new official test labels opened.\n\n|Input|Exact|WER|Multi-sign exact|Multi-signWER|\n|---|---:|---:|---:|---:|\n')
  for name,r in metrics.items():
   v=r['validation'];multi=v['by_length']['multiple_signs'];f.write(f"|{name}|{round(v['exact']*23)}/23|{100*v['operation_wer']:.2f}%|{round(multi['exact']*6)}/6|{100*multi['wer']:.2f}%|\n")
  f.write('\nPaired exact counts:'+json.dumps(comparison['paired_counts'])+'. Small repeatedly inspected development set,one seed and six multi-sign examples prevent claims of robust or statistically established context benefit. Input lengths change padding and dropout random draws,so identical seed is a control rather than bitwise-identical optimization. Full paired predictions are private in research/private/lsfb-context-ablation-v1; no human semantic approval. Current-only adapter and result backup saved alongside previous context run under /home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-narrow-context-v1/.\n')
 print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()
