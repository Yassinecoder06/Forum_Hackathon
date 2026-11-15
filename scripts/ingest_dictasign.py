#!/usr/bin/env python3
"""Public Dicta-Sign France-LSF lexical videos -> measured motion/Animics data.
Local research only; this does not assert redistribution or human sign approval.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import gzip,hashlib,json,re,subprocess,sys,time,uuid
from pathlib import Path
from urllib.parse import urljoin,urlparse
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from threading import BoundedSemaphore
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.services import capture_library as store
BASE='https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/'
INDEX=BASE+'concepts_fre.html'
NETWORK=BoundedSemaphore(2)
ANIMICS_REVISION='daf09928fab05f6227d6d0836395d2bddbdebc4f'

def fetch(url):
    if urlparse(url).hostname!='www.sign-lang.uni-hamburg.de':raise ValueError('Unexpected source host')
    for attempt in range(4):
        try:
            with NETWORK,urlopen(Request(url,headers={'User-Agent':'SignaCulture local noncommercial lexical research; cached requests'}),timeout=45) as response:
                if urlparse(response.url).hostname!='www.sign-lang.uni-hamburg.de':raise ValueError('Unexpected redirect')
                return response.read()
        except HTTPError as exc:
            if exc.code not in [429,500,502,503,504] or attempt==3:raise
            pause=max(2,min(300,int(exc.headers.get('Retry-After','10')) if exc.headers.get('Retry-After','10').isdigit() else 30))
            print(f'Source cooldown {pause}s',flush=True);time.sleep(pause)
        except (TimeoutError,OSError):
            if attempt==3:raise
            time.sleep(2*(attempt+1))

def parse_index(html):
    soup=BeautifulSoup(html,'html.parser');result={}
    for a in soup.select('a[href]'):
        m=re.search(r'(?:^|/)cs_(\d+)\.html$',a['href'])
        if not m:continue
        identifier=m.group(1);label=a.get_text(' ',strip=True)
        if not label:continue
        entry=result.setdefault(identifier,{'concept_id':identifier,'labels':[],'page_url':urljoin(INDEX,a['href'])})
        if label not in entry['labels']:entry['labels'].append(label)
    return sorted(result.values(),key=lambda x:int(x['concept_id']))

def parse_lsf(html,page_url):
    soup=BeautifulSoup(html,'html.parser');entries=[]
    for box in soup.select('.cb-left,.cb-right'):
        lang=box.select_one('.title-l1')
        if not lang or lang.get_text(strip=True)!='LSF':continue
        label=box.select_one('.title-c');label=label.get_text(' ',strip=True) if label else ''
        for number,nc in enumerate(box.select('.nc')):
            gloss=nc.select_one('.gloss');gloss=gloss.get_text(' ',strip=True) if gloss else ''
            hns=nc.select_one('.hns');sources=nc.select('source[src]');video=next((urljoin(page_url,s['src']) for s in sources if s.get('type')=='video/mp4'),None)
            if video:
                if '/concepts/lsf/' not in urlparse(video).path:raise ValueError('Non-LSF video in LSF block')
                entries.append({'concept_fr':label,'canonical_gloss':gloss,'hamnosys':hns.get_text(strip=True) if hns else None,'video_url':video,'variant_index':number})
    return entries

def read_or_fetch(path,url):
    if path.exists():return path.read_bytes()
    data=fetch(url);temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path);return data

def inventory_item(entry,cache):
    raw=read_or_fetch(cache/f"cs_{entry['concept_id']}.html",entry['page_url'])
    return [{**entry,**item,'page_sha256':hashlib.sha256(raw).hexdigest(),'source':'Dicta-Sign public lexical portal','language':'fsl','rights_status':'pending','licence':'Not specified per portal asset; local research preview; redistribution pending'} for item in parse_lsf(raw.decode('utf-8'),entry['page_url'])]

def extract_entry(entry,register=True):
    job=str(uuid.uuid5(uuid.NAMESPACE_URL,entry['video_url']));folder=store.storage()/'captures'/job;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'source.mp4';motion_path=folder/'motion.json';raw_path=folder/'animics-landmarks.json.gz'
    raw=read_or_fetch(path,entry['video_url']);checksum=hashlib.sha256(raw).hexdigest()
    provenance={**entry,'source_sha256':checksum,'retrieved_at':datetime.now(timezone.utc).isoformat(),'animics_reference_revision':ANIMICS_REVISION,'extractor':'MediaPipe Holistic + Hands fallback; adapted Animics raw field/axis conventions, not the full browser engine'}
    store.write_json(folder/'source_provenance.json',provenance)
    if not motion_path.exists() or not raw_path.exists():
        result=subprocess.run([str(store.runtime_python('motion')),str(ROOT/'backend/app/workers/capture.py'),str(path),str(motion_path),str(folder/'progress.json'),'--raw-animics',str(raw_path)],capture_output=True,text=True,timeout=240,cwd=ROOT)
        if result.returncode:raise ValueError(result.stderr[-1200:])
    motion=json.loads(motion_path.read_text());q=motion['quality']
    motion['attribution']={'credit':'Dicta-Sign project / French LSF lexical portal','source_url':entry['page_url'],'license':entry['licence'],'changes':'Computer-vision landmarks and approximate 3D retargeting; local preview only'}
    store.write_json(motion_path,motion)
    store.write_json(folder/'status.json',{'job_id':job,'status':'ready','progress':100,'filename':'Dicta-Sign '+entry['canonical_gloss'],'duration_ms':motion['duration_ms'],'quality':q})
    result={'concept_id':entry['concept_id'],'concept':entry['concept_fr'],'gloss':entry['canonical_gloss'],'capture_id':job,'video_url':entry['video_url'],'source_sha256':checksum,'quality':q,'raw_animics_path':str(raw_path)}
    if q['pose_coverage']<.8 or max(q['left_hand_coverage'],q['right_hand_coverage'])<.35:
        return {**result,'status':'quality_review_needed'}
    if register:
        registered=store.register(job,entry['concept_fr'],entry['labels'],entry['page_url'],entry['licence'])
        with store.library_lock():
            rows=store.library()
            for row in rows:
                if row['motion']['motion_id']==registered['motion']['motion_id']:
                    row.update(source_id='DICTASIGN_PORTAL_'+entry['concept_id'],attribution=motion['attribution'],canonical_gloss=entry['canonical_gloss'],provenance=provenance,raw_animics_path=str(raw_path))
                    row['gloss'].update(gloss=entry['canonical_gloss'],source_id='DICTASIGN_PORTAL_'+entry['concept_id'],variant='France LSF · Dicta-Sign lexical citation form')
                    row['motion']['quality_status']='captured_pending_review'
            store.write_json(store.storage()/'library.json',rows)
        result['motion_id']=registered['motion']['motion_id']
    return {**result,'status':'imported' if register else 'captured'}

def retry_failed(workers):
    status=store.storage()/'dictasign_status.json'
    state=json.loads(status.read_text())
    if state['status']!='complete':raise SystemExit('Wait for the current ingestion to complete before retrying failures')
    inventory=json.loads((ROOT/'data/source_inventory/dictasign_lsf.json').read_text())['entries']
    entries={e['concept_id']:e for e in inventory}
    failed=[(i,r) for i,r in enumerate(state['results']) if r['status']=='failed']
    state['status']='retrying';store.write_json(status,state)
    with ThreadPoolExecutor(max_workers=max(1,min(4,workers))) as pool:
        pending={pool.submit(extract_entry,entries[r['concept_id']]):(i,r) for i,r in failed}
        for future in as_completed(pending):
            i,old=pending[future]
            try:result=future.result()
            except Exception as exc:result={**old,'error':str(exc)}
            if result['status']!='failed':
                state.setdefault('recovered_failures',[]).append({'concept_id':old['concept_id'],'previous_error':old['error']})
            state['results'][i]=result
            for kind in ['imported','quality_review_needed','failed']:
                state[kind]=sum(r['status']==kind for r in state['results'])
            store.write_json(status,state)
            print(f"Retried {old['concept_id']}: {result['status']}",flush=True)
    state['status']='complete';store.write_json(status,state)
    print(json.dumps({k:v for k,v in state.items() if k not in ['results','recovered_failures']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--limit',type=int,default=0);p.add_argument('--concept-ids',default='');p.add_argument('--workers',type=int,default=2);p.add_argument('--inventory-only',action='store_true');p.add_argument('--capture-workers',type=int,default=2);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if a.retry_failed:return retry_failed(a.capture_workers)
    cache=store.storage()/'dictasign/source-pages';cache.mkdir(parents=True,exist_ok=True)
    entries=parse_index(read_or_fetch(cache/'concepts_fre.html',INDEX).decode('utf-8'))
    if a.concept_ids:
        ids=set(a.concept_ids.split(','));entries=[e for e in entries if e['concept_id'] in ids]
    if not a.all and a.limit:entries=entries[:a.limit]
    if not a.all and not a.limit and not a.concept_ids:raise SystemExit('Choose --all, --limit or --concept-ids explicitly')
    state={'status':'inventory','indexed_concepts':len(entries),'inventory_processed':0,'total':0,'processed':0,'imported':0,'quality_review_needed':0,'failed':0,'results':[]};status=store.storage()/('dictasign_status.json' if a.all else 'dictasign_sample_status.json');items=[];start=time.monotonic()
    store.write_json(status,state)
    # Read only public citation-form pages/videos; two workers and a small delay.
    def cached(e):
        result=inventory_item(e,cache);time.sleep(.15);return result
    dedup={};captures={}
    def collect(future):
        e=captures.pop(future)
        try:r=future.result()
        except Exception as exc:r={'concept':e['concept_fr'],'concept_id':e['concept_id'],'status':'failed','error':str(exc)}
        state['results'].append(r);state['processed']+=1
        if r['status'] in ['imported','quality_review_needed']:state[r['status']]+=1
        if r['status']=='failed':state['failed']+=1
        store.write_json(status,state);print(f"[{state['processed']}/{state['total']}] {e['concept_fr']}: {r['status']}",flush=True)
    with ThreadPoolExecutor(max_workers=max(1,min(2,a.workers))) as pool, ThreadPoolExecutor(max_workers=max(1,min(4,a.capture_workers))) as video_pool:
        for future in as_completed({pool.submit(cached,e):e for e in entries}):
            try:page_items=future.result()
            except Exception as exc:
                page_items=[];state['results'].append({'status':'inventory_failed','error':str(exc)});state['failed']+=1
            for e in page_items:
                if e['video_url'] in dedup:
                    dedup[e['video_url']]['labels']=list(dict.fromkeys(dedup[e['video_url']]['labels']+e['labels']))
                else:
                    dedup[e['video_url']]=e
                    if not a.inventory_only:captures[video_pool.submit(extract_entry,e)]=e
            state['inventory_processed']+=1;state['total']=len(dedup)
            for done in [f for f in captures if f.done()]:collect(done)
            store.write_json(status,state)
            if state['inventory_processed']%50==0:print(f"Inventory {state['inventory_processed']}/{len(entries)}",flush=True)
        items=sorted(dedup.values(),key=lambda x:int(x['concept_id']))
        out=ROOT/'data/source_inventory';out.mkdir(exist_ok=True)
        name='dictasign_lsf.json' if a.all else 'dictasign_lsf_sample.json'
        store.write_json(out/name,{'source_url':INDEX,'rights_status':'pending','scope':'Public France-LSF lexical citation forms, not sentence pairs','entries':items})
        state['status']='capturing';store.write_json(status,state)
        for future in as_completed(list(captures)):collect(future)
    state.update(status='complete',seconds=round(time.monotonic()-start,2));store.write_json(status,state)
    print(json.dumps({k:v for k,v in state.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
