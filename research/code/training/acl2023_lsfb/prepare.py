"""Lossless LSFB-only parallel text export for controlled ACL-2023-inspired trials."""
import argparse, collections, hashlib, json, re
from pathlib import Path
from urllib.parse import quote, unquote

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path('/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont-usable-v1'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    rows={s:[json.loads(x) for x in (a.source/f'{s}.jsonl').read_text().splitlines()] for s in ('train','val')}
    norm=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
    for key in ('conversation_id','signer_id'):
        assert not {r[key] for r in rows['train']}&{r[key] for r in rows['val']}
    assert not {norm(r['current_french']) for r in rows['train']}&{norm(r['current_french']) for r in rows['val']}
    report={'language':'LSFB','linguistically_reviewed':False,'official_test_opened':False,'source_hashes':{},'splits':{},'target_encoding':'Percent-escape individual lemmas, then space delimit; reversible. No suffix stripping.'}
    for split,rs in rows.items():
        outputs=collections.defaultdict(list)
        for r in rs:
            assert r['language']=='LSFB' and r['target']
            # Spaces within a lexical label must never become additional sign tokens.
            target=' '.join(quote(t,safe="-'",encoding='utf-8') for t in r['target'])
            assert [unquote(t) for t in target.split()]==r['target']
            current=' '.join(r['current_french'].split());context=' '.join(r['input'].split())
            outputs['fr'].append('<2lsfb> '+current);outputs['context.fr'].append('<2lsfb> '+context);outputs['gloss'].append(target)
            outputs['metadata.jsonl'].append(json.dumps({k:r[k] for k in ('id','language','conversation_id','signer_id','target','pose_path','review_status')},ensure_ascii=False))
        for ext,lines in outputs.items():(a.output/f'{split}.{ext}').write_text('\n'.join(lines)+'\n')
        report['source_hashes'][f'{split}.jsonl']=digest(a.source/f'{split}.jsonl')
        report['splits'][split]={'rows':len(rs),'multi_sign':sum(len(r['target'])>1 for r in rs)}
    counts=collections.Counter(t for r in rows['train'] for t in r['target']);refs=[t for r in rows['val'] for t in r['target']]
    report.update(train_vocabulary=len(counts),singleton_training_labels=sum(v==1 for v in counts.values()),validation_operations=len(refs),unseen_validation_operations=sum(t not in counts for t in refs))
    report['files']={p.name:digest(p) for p in a.output.iterdir()}
    (a.output/'MANIFEST.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
