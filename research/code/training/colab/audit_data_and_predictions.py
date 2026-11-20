"""Read-only coverage and independent prediction audit; never modifies a checkpoint."""
import argparse,collections,json,unicodedata
from pathlib import Path
from rapidfuzz.distance import Levenshtein

def read(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def audit(splits,runs):
    data={name:read(splits/(name+'.jsonl')) for name in ['train','val','test']}
    vocabulary=collections.Counter(t for row in data['train'] for t in row['gloss_sequence'].split())
    report={'splits':{},'train_target_vocabulary':len(vocabulary),'train_target_singletons':sum(n==1 for n in vocabulary.values()),'total_target_vocabulary':len({t for rows in data.values() for row in rows for t in row['gloss_sequence'].split()}),'runs':{}}
    for name,rows in data.items():
        tokens=[r['gloss_sequence'].split() for r in rows];unseen=[[t for t in ts if t not in vocabulary] for ts in tokens]
        report['splits'][name]={'rows':len(rows),'reference_tokens':sum(map(len,tokens)),'target_vocabulary':len(set(t for ts in tokens for t in ts)),'unseen_tokens_vs_train':sum(map(len,unseen)),'sentences_with_unseen_target_tokens':sum(bool(ts) for ts in unseen),'signers':sorted({r['signer_id'] for r in rows}),'recordings':len({r['recording_id'] for r in rows})}
    normalized=lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
    expected=[(r['french_text'],r['gloss_sequence']) for r in data['test']]
    for name in ['poc-100','t5-base-300','flan-t5-base-300']:
        folder=runs/name;predictions=json.loads((folder/'predictions.json').read_text());original=json.loads((folder/'metrics.json').read_text())['model']
        assert [(r['french'],r['reference']) for r in predictions]==expected
        operations=collections.Counter()
        for row in predictions:operations.update(op.tag for op in Levenshtein.editops(row['reference'].split(),row['prediction'].split()))
        denominator=sum(len(r['reference'].split()) for r in predictions);wer=sum(operations.values())/denominator
        exact=sum(r['reference']==r['prediction'] for r in predictions)
        assert abs(wer-original['wer'])<1e-12 and exact/len(predictions)==original['exact_match']
        report['runs'][name]={'examples':len(predictions),'exact_matches':exact,'normalized_exact_matches':sum(normalized(r['reference'])==normalized(r['prediction']) for r in predictions),'independent_wer':wer,'edit_operations':dict(operations),'reference_tokens':denominator,'predicted_tokens':sum(len(r['prediction'].split()) for r in predictions),'unique_predictions':len({r['prediction'] for r in predictions}),'pretraining_marker_outputs':sum('<extra_id_' in r['prediction'] for r in predictions)}
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--splits',type=Path,default=Path('.runtime/lsfb-poc/short-splits'));p.add_argument('--runs',type=Path,default=Path('.runtime/colab-results'));p.add_argument('--output',type=Path,default=Path('research/text-to-gloss/lsfb/COLAB_ZERO_AUDIT.json'));args=p.parse_args()
    result=audit(args.splits,args.runs);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
