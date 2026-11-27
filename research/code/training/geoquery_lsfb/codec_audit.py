import json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from .codec import GraphCodec

def main():
 root=Path('/home/jmal/Projects/Active/Forum_Hackathon/datasets/geoquery-lsfb-graph-feasibility-v1');out=Path('.runtime/geoquery-graph-codec-v1');out.mkdir(exist_ok=False);meta=json.loads((root/'MANIFEST.json').read_text());data={}
 for split in ['train','val']:
  path=root/f'{split}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['files'][path.name];data[split]=[json.loads(x) for x in path.read_text().splitlines()]
 codec=GraphCodec.fit([r['lsfb_graph'] for r in data['train']]);tok=AutoTokenizer.from_pretrained('.runtime/lsfb-cont-local-v3/tokenizer',local_files_only=True);report={'language':'LSFB','codebook_fit':'train only','input_source':'DeepL French','target':'full author graph, not flat gloss','counts':{},'tokenizer':'T5-base existing local','dictionary_entries':len(codec.hns),'no_local_linguistic_approval':True}
 (out/'hns_dictionary.json').write_text(json.dumps(codec.hns,ensure_ascii=False,indent=2))
 for split,rows in data.items():
  lengths=[];encoded=[];unknown=0;roundtrip=0
  for row in rows:
   text=codec.encode(row['lsfb_graph']);assert codec.decode(text)==row['lsfb_graph'];ids=tok(text)['input_ids'];decoded=tok.decode(ids,skip_special_tokens=True);roundtrip+=decoded!=text;unknown+=ids.count(tok.unk_token_id);lengths.append(len(ids));encoded.append({**row,'target_text':text,'target_tokens':len(ids),'input':'French to LSFB temporal graph: '+row['french']})
  report['counts'][split]={'examples':len(rows),'max_target_tokens':max(lengths),'mean_target_tokens':sum(lengths)/len(lengths),'above_1024':sum(n>1024 for n in lengths),'unknown_subtokens':unknown,'tokenizer_roundtrip_failures':roundtrip,'codec_roundtrip_failures':0}
  (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in encoded))
 report['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()};(out/'AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
