from pathlib import Path
import requests,json,hashlib,csv,collections
from huggingface_hub import HfApi,hf_hub_download
out=Path('/content/fsl-audit');out.mkdir(exist_ok=True)
api=HfApi();info=api.dataset_info('SignerX/SignVerse-2M',files_metadata=True,token=False)
files=[{'name':x.rfilename,'size':x.size,'lfs':x.lfs} for x in info.siblings]
(out/'hub_inventory.json').write_text(json.dumps({'revision':info.sha,'gated':info.gated,'card':info.card_data.to_dict(),'files':files},default=str,indent=2))
print('REVISION',info.sha,'GATED',info.gated,'TOTAL_FILES',len(files),flush=True)
print('METADATA_FILES',[(x['name'],x['size']) for x in files if not x['name'].endswith(('.tar','.gz','.zip'))],flush=True)
for name in ['README.md','SignVerse-2M-metadata_split.csv','Sign-DWPose-2M-metadata_ori.csv']:
 f=next((f for f in files if f['name']==name),None)
 if not f:print('MISSING',name);continue
 assert f['size']<60_000_000
 path=Path(hf_hub_download('SignerX/SignVerse-2M',name,repo_type='dataset',revision=info.sha,token=False,local_dir=out))
 print('DOWNLOADED',name,path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
 if name.endswith('.csv'):
  with path.open() as fp:
   reader=csv.DictReader(fp);rows=list(reader)
  print('SCHEMA',name,reader.fieldnames,'ROWS',len(rows),flush=True)
  exact=[r for r in rows if 'fsl' in r.values()]
  (out/(name+'.fsl.json')).write_text(json.dumps(exact,ensure_ascii=False,indent=2))
  print('FSL_EXACT_ROWS',len(exact),'SAMPLE',exact[:2],flush=True)
j=requests.get('https://api.github.com/repos/SignerX/SignVerse-2M/git/trees/main?recursive=1',timeout=30);j.raise_for_status();(out/'github_tree.json').write_text(json.dumps(j.json(),indent=2));print('GITHUB_CODE',[x['path'] for x in j.json().get('tree',[]) if x['path'].endswith(('.py','.yaml','.yml','.ipynb','.sh','.md'))],flush=True)
for name,url in [('rosetta','https://hdl.handle.net/11403/rosetta-lsf/v3'),('dictasign','https://hdl.handle.net/11403/dicta-sign-lsf-v2/v1')]:
 try:
  r=requests.get(url,timeout=25);(out/(name+'.html')).write_text(r.text);print('RESOURCE',name,r.status_code,r.url,len(r.content),flush=True)
 except Exception as e:print('RESOURCE',name,type(e).__name__,flush=True)
