"""Retrieve only official CONT train/validation shards; persist thin annotations."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
from huggingface_hub import hf_hub_download
from prepare import REVISION,records_from_tar,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--local-shard2',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for number in [2,3,4,5,6,7,1]:
        out=a.output/f'annotations_{number:06d}.json.gz'
        if out.is_file():print('Cached annotation shard',number,flush=True);continue
        print('Collecting official '+('validation' if number==1 else 'train')+' shard '+str(number),flush=True)
        if number==2 and a.local_shard2 and a.local_shard2.exists():
            records=json.load(gzip.open(a.local_shard2,'rt',encoding='utf-8'));checksum='73c701e337621d658eac90aa785c942c08f36945d5c452414891e801cf6d6c17'
        else:
            filename=f'shards/annotated/shard_{number:06d}.tar'
            path=Path(hf_hub_download('ppoitier/lsfb-cont',filename,repo_type='dataset',revision=REVISION))
            checksum=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();records=records_from_tar(path)
        value={'source':'ppoitier/lsfb-cont','revision':REVISION,'source_shard':number,'official_role':'validation' if number==1 else 'train','tar_sha256':checksum,'records':records}
        temp=out.with_suffix(out.suffix+'.tmp')
        with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False)
        temp.replace(out)
        print(json.dumps({'shard':number,'recordings':len(records),'translations':sum(len(r.get('annotations.translation.json',[])) for r in records.values()),'output':str(out)}),flush=True)
    dump(a.output/'COLLECTION_COMPLETE.json',{'shards':[1,2,3,4,5,6,7],'test_shard_0_read':False})
if __name__=='__main__':main()
