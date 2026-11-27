"""Train-only recurrent vocabulary selection and context LSFB diagnostic."""
import json,collections,hashlib
from pathlib import Path
from urllib.parse import quote

def prepare():
 source=Path('/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont-usable-v1');out=Path('.runtime/lsfb-narrow-context-data-v1');out.mkdir(exist_ok=False)
 rows={s:[json.loads(x) for x in (source/f'{s}.jsonl').read_text().splitlines()] for s in ['train','val']}
 tr=rows['train'];v={t for t,n in collections.Counter(t for r in tr for t in r['target']).most_common(128)}
 for _ in range(100):
  eligible=[r for r in tr if 1<=len(r['target'])<=4 and set(r['target'])<=v];f=collections.Counter(t for r in eligible for t in r['target']);groups=collections.defaultdict(set)
  for r in eligible:
   for t in r['target']:groups[t].add(r['conversation_id'])
  nv={t for t in v if f[t]>=3 and len(groups[t])>=2}
  if nv==v:break
  v=nv
 selected={s:[dict(r) for r in rs if 1<=len(r['target'])<=4 and set(r['target'])<=v] for s,rs in rows.items()}
 # Keep original protected conversation/signer split; vocabulary selected from train only.
 for field in ['conversation_id','signer_id']:
  assert not {r[field] for r in selected['train']}&{r[field] for r in selected['val']}
 for rs in selected.values():
  for r in rs:r['target_text']=' '.join(quote(t,safe="-'") for t in r['target'])
 files={}
 for s,rs in selected.items():
  path=out/f'{s}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs));files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 report={'language':'LSFB','status':'complete','files':files,'vocabulary':sorted(v),'frequency':dict(f),'counts':{s:len(rs) for s,rs in selected.items()},'multi_sign_counts':{s:sum(len(r['target'])>1 for r in rs) for s,rs in selected.items()},'selection':'train-only top128 initial; fixed point >=3operations in >=2train conversations; target length1..4','linguistically_validated':False,'context':'original previous-completed French dialogue plus current text','holdout':'previously inspected development conversations, not a fresh test benchmark'}
 (out/'SCAN_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return out
if __name__=='__main__':prepare()
