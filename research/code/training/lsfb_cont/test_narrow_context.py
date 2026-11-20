import json,collections
from pathlib import Path
from training.lsfb_cont.train_narrow_context import measure

def test_repetition_and_split_evidence():
 p=Path('.runtime/lsfb-narrow-context-data-v1');rs={s:[json.loads(x) for x in (p/f'{s}.jsonl').read_text().splitlines()] for s in ['train','val']};f=collections.Counter(t for r in rs['train'] for t in r['target']);g=collections.defaultdict(set)
 for r in rs['train']:
  for t in r['target']:g[t].add(r['conversation_id'])
 assert all(n>=3 and len(g[t])>=2 for t,n in f.items())
 for key in ['conversation_id','signer_id','current_french']:assert not {r[key] for r in rs['train']}&{r[key] for r in rs['val']}
 assert all(r['language']=='LSFB' and 'Contexte:' in r['input'] for rows in rs.values() for r in rows)

def test_empty_and_repetition_scoring():
 rows=[{'target':['a','b']}];assert measure(rows,[''])['operation_wer']==1
 assert measure(rows,['a a a a'])['operation_wer']==1.5
 assert measure(rows,['a b'])['by_length']['multiple_signs']['exact']==1
