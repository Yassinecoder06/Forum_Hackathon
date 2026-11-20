from pathlib import Path,PurePosixPath
import requests,json,tarfile,io,collections,hashlib,time
p=Path('/content/fsl-audit');out=Path('/content/fsl-samples');out.mkdir(exist_ok=True)
inv=json.load(open(p/'hub_inventory.json'));rev=inv['revision'];selected=json.load(open(p/'selected.json'));wanted={r['video_id']:r for r in selected};shards=collections.defaultdict(set)
for r in selected:shards[r['archives'][0]].add(r['video_id'])
sess=requests.Session();budget=160*1024**2;used=0;requests_count=0;index=[];failures=[]
def get_range(name,offset,size):
 global used,requests_count
 if used+size>budget or requests_count>=500:raise ValueError('Download budget exhausted')
 url=f'https://huggingface.co/datasets/SignerX/SignVerse-2M/resolve/{rev}/dataset/{name}?offset={offset}'
 r=sess.get(url,headers={'Range':f'bytes={offset}-{offset+size-1}'},timeout=45,stream=True);requests_count+=1
 if r.status_code!=206:r.close();raise ValueError(f'Range unsupported HTTP {r.status_code}')
 if not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):r.close();raise ValueError('Wrong Content-Range')
 b=b''.join(r.iter_content(1024*1024));r.close();used+=len(b)
 if len(b)!=size:raise ValueError('Short range')
 return b
for shard,ids in shards.items():
 offset=0;found=set();print('SCAN',shard,flush=True)
 try:
  for n in range(400):
   raw=get_range(shard,offset,512)
   if raw==b'\0'*512:break
   ti=tarfile.TarInfo.frombuf(raw,'utf-8','strict');path=PurePosixPath(ti.name);body=offset+512
   index.append({'shard':shard,'name':ti.name,'size':ti.size,'offset':body})
   if path.is_absolute() or '..' in path.parts or ti.issym() or ti.islnk():raise ValueError('Unsafe TAR member')
   # Ordinary GNU/PAX metadata are skipped, never extracted or executed.
   parts=[x for x in path.parts if x!='.'];video=parts[-2] if len(parts)>1 else ''
   if ti.isfile() and video in ids and path.name in ('poses.npz','caption.json'):
    if ti.size>48*1024**2:print('SKIP oversized',video,path.name,ti.size,flush=True)
    else:
     dest=out/video;dest.mkdir(exist_ok=True);content=get_range(shard,body,ti.size);(dest/path.name).write_bytes(content);(dest/'manifest.json').write_text(json.dumps(wanted[video],indent=2));found.add((video,path.name));print('FETCH',video,path.name,ti.size,flush=True)
   offset=body+((ti.size+511)//512)*512
   if all((v,f) in found for v in ids for f in ('poses.npz','caption.json')):break
 except Exception as e:failures.append({'shard':shard,'error':str(e)});print('FAIL',shard,str(e),flush=True)
 (p/'range_index.json').write_text(json.dumps(index,indent=2))
(p/'range_fetch_report.json').write_text(json.dumps({'bytes':used,'requests':requests_count,'budget_bytes':budget,'failures':failures,'files':{str(f.relative_to(out)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in out.rglob('*') if f.is_file()}},indent=2))
print('COMPLETE',used,requests_count,flush=True)
