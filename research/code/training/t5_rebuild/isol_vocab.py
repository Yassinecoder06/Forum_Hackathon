"""Collect the provider's fixed 500-class inventory from ISOL TRAIN shards only."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
from prepare import dump
REV='947ee2209969d2fc12fa8f95503fe32906a64941'
def collect(folder):
    labels={};examples={};sources=[]
    for number in range(3,10):
        filename=f'shards/500/shard_{number:06d}.tar';path=folder/filename
        if not path.is_file():subprocess.run(['hf','download','ppoitier/lsfb-isol',filename,'--repo-type','dataset','--revision',REV,'--local-dir',str(folder)],check=True)
        records={}
        with tarfile.open(path) as tar:
            for m in tar:
                suffix=next((s for s in ['label.txt','label.idx','parent.txt','linguistic_metadata.json'] if m.name.endswith('.'+s)),None)
                if not suffix:continue
                text=tar.extractfile(m).read().decode();records.setdefault(m.name[:-len(suffix)-1],{})[suffix]=json.loads(text) if suffix.endswith('.json') else text.strip()
        for clip,row in records.items():
            idx=int(row['label.idx']);label=row['label.txt'];assert idx not in labels or labels[idx]==label
            assert row['linguistic_metadata.json']['lemma']==label
            labels[idx]=label;examples.setdefault(idx,{'clip_id':clip,'parent':row.get('parent.txt'),'metadata':row['linguistic_metadata.json']})
        sources.append({'file':filename,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'clips':len(records)})
        print('Observed',len(labels),'of 500 classes after train shard',number,flush=True)
        if set(labels)==set(range(500)):break
    if not set(labels)<=set(range(500)):raise ValueError('Unexpected class IDs')
    result={'source':'ppoitier/lsfb-isol','revision':REV,'variant':'500','license':'CC-BY-NC-SA-4.0','source_role':'independent class vocabulary; no French sentence supervision','source_split':'official train only','source_shards':sources,'classes':[{'class_id':i,'unit':'lexical:'+labels[i],'label':labels[i],'source_example':examples[i]} for i in sorted(labels)],'complete':set(labels)==set(range(500)),'missing_class_ids':sorted(set(range(500))-set(labels))}
    dump(folder/'vocabulary_500_complete.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();collect(a.folder)
