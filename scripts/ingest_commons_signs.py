#!/usr/bin/env python3
"""Inventory licensed France-LSF clips, then explicitly ingest selected/all clips.

Uses the documented Wikimedia Commons API, not webpage scraping. Every file is
checked independently for its reported licence/restrictions before retrieval.
Source videos, captures and generated library live in ignored .runtime storage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlencode,quote
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.app.services import capture_library as store

API='https://commons.wikimedia.org/w/api.php'
CATEGORY='Category:Vidéos LSF par Laura Jauvert'
ALLOWED={'CC BY 3.0','CC BY 4.0','CC BY-SA 3.0','CC BY-SA 4.0','CC0'}

def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url,headers={'User-Agent':'SignaCultureAI/0.2 (local LSF research prototype)'}),timeout=45) as response:
                return response.read()
        except HTTPError as exc:
            if attempt==3: raise
            if exc.code==429:
                wait=max(60,min(300,int(exc.headers.get('Retry-After','60'))))
                print(f'Commons rate limit: waiting {wait}s before retry.',flush=True)
                time.sleep(wait)
            else: time.sleep(2**attempt)
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)

def api(**params):
    return json.loads(fetch(API+'?'+urlencode({'format':'json',**params})))

def plain(value):
    return html.unescape(re.sub('<[^>]+>',' ',value or '')).strip()

def inventory():
    pages=[];continuation={}
    while True:
        response=api(action='query',list='categorymembers',cmtitle=CATEGORY,cmtype='file',cmlimit=500,**continuation)
        pages.extend(response['query']['categorymembers'])
        if 'continue' not in response: break
        continuation=response['continue']
    entries=[]
    for start in range(0,len(pages),40):
        response=api(action='query',pageids='|'.join(str(x['pageid']) for x in pages[start:start+40]),prop='imageinfo',iiprop='url|size|sha1|extmetadata')
        for page in response['query']['pages'].values():
            match=re.search(r'\(fsl\)-Laura Jauvert-(.+)\.webm$',page['title'])
            if not match or not page.get('imageinfo'): continue
            info=page['imageinfo'][0];meta=info.get('extmetadata',{})
            def field(key): return meta.get(key,{}).get('value','')
            licence=field('LicenseShortName').strip()
            allowed=licence in ALLOWED and not field('Restrictions').strip()
            entries.append({'source_id':f"WMC_{page['pageid']}",'page_id':page['pageid'],'title':page['title'],
                'concept_fr':match[1].strip().lower(),'language_variant':'LSF_FR','kind':'isolated_sign_video',
                'source_url':'https://commons.wikimedia.org/wiki/'+quote(page['title'].replace(' ','_')),
                'download_url':info['url'].split('?')[0],'license':licence,'license_url':field('LicenseUrl'),
                'credit':plain(field('Artist')),'restrictions':plain(field('Restrictions')),'eligible_for_motion':allowed,
                'size_bytes':info.get('size'),'source_sha1':info.get('sha1'),'linguistic_review':'pending',
                'verified_at':datetime.now(timezone.utc).isoformat()})
        checkpoint=ROOT/'.runtime/commons_inventory_partial.json'
        checkpoint.parent.mkdir(exist_ok=True)
        store.write_json(checkpoint,{'entries':entries,'processed_pages':start+40})
        print(f'Inspected {len(entries)} clip licences.',flush=True)
        time.sleep(2)
    entries.sort(key=lambda x:x['concept_fr'])
    target=ROOT/'data/source_inventory';target.mkdir(parents=True,exist_ok=True)
    output=target/'commons_lsf.json';store.write_json(output,{'category':CATEGORY,'api':API,'entries':entries})
    print(f'Inventory: {len(entries)} France-LSF files, {sum(x["eligible_for_motion"] for x in entries)} with compatible reported licences.',flush=True)
    return entries

def prepare_author_mirrors(entries):
    revision='701563bb9de3a5e7d64d52632bb40973f17388bf'
    tree_path=ROOT/'research/parlr_tree.json'
    if not tree_path.exists(): return entries
    tree={item['path']:item for item in json.loads(tree_path.read_text())['tree']}
    segments={}
    for part in (1,2):
        filename=f'data/partie-{part}:-Apprendre-300-mots-du-quotidien-en-LSF.jauvert-laura.hd.ass'
        url='https://raw.githubusercontent.com/parlr/lsf-data/'+revision+'/'+quote(filename)
        raw=fetch(url)
        for line in raw.decode('utf-8-sig').splitlines():
            if line.startswith('Dialogue:'):
                fields=line.split(',',9)
                label=re.sub(r'\{[^}]*\}','',fields[-1]).strip().casefold()
                segments[label]={'part':part,'start':fields[1],'end':fields[2],'subtitle_source':url}
    matched=0
    for entry in entries:
        path='videos/jauvert/'+entry['concept_fr']+'.webm'
        # Never generalize the repository MIT licence to videos. A candidate
        # must match BOTH an individually licensed Commons concept and the
        # subtitle of the same identified original work by this author.
        if entry['eligible_for_motion'] and entry['concept_fr'] in segments and path in tree:
            entry['author_mirror']={'url':'https://raw.githubusercontent.com/parlr/lsf-data/'+revision+'/'+quote(path),
                'git_revision':revision,'git_blob_sha1':tree[path]['sha'],'path':path,
                'original_segment':segments[entry['concept_fr']],
                'note':'Author repository rendition of the matched licensed original work; resolution/cut may differ from Commons.'}
            matched+=1
    store.write_json(ROOT/'data/source_inventory/commons_lsf.json',{'category':CATEGORY,'api':API,'entries':entries})
    print(f'Author source matched for {matched} individually licensed concepts.',flush=True)
    return entries

def ingest(entry):
    job_id=str(uuid.uuid5(uuid.NAMESPACE_URL,entry['source_url']))
    folder=store.storage()/'captures'/job_id;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'source.mp4';motion_path=folder/'motion.json'
    if not path.exists():
        mirror=entry.get('author_mirror')
        url=mirror['url'] if mirror else entry['download_url']
        raw=fetch(url)
        if mirror:
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()
            if blob!=mirror['git_blob_sha1']: raise ValueError('Author repository blob checksum mismatch')
        elif entry['source_sha1'] and hashlib.sha1(raw).hexdigest()!=entry['source_sha1']:
            raise ValueError('Source SHA1 does not match Commons metadata')
        path.write_bytes(raw)
        store.write_json(folder/'source_provenance.json',{'retrieved_from':url,'sha256':hashlib.sha256(raw).hexdigest(),'canonical_licence_source':entry['source_url'],'author_mirror':mirror})
    if not motion_path.exists():
        result=subprocess.run([str(store.runtime_python('motion')),str(ROOT/'backend/app/workers/capture.py'),str(path),str(motion_path),str(folder/'progress.json')],capture_output=True,text=True,timeout=240,cwd=ROOT)
        if result.returncode:
            error=folder/'motion.error';raise ValueError(error.read_text() if error.exists() else result.stderr[-500:])
    motion=json.loads(motion_path.read_text());quality=motion['quality']
    if quality['pose_coverage']<.8 or max(quality['left_hand_coverage'],quality['right_hand_coverage'])<.35:
        return {'source_id':entry['source_id'],'status':'quality_review_needed','quality':quality}
    motion['attribution']={k:entry[k] for k in ('credit','source_url','license','license_url')}
    motion['attribution']['changes']='MediaPipe landmarks, approximate depth and procedural avatar retargeting'
    store.write_json(motion_path,motion)
    store.write_json(folder/'status.json',{'job_id':job_id,'status':'ready','progress':100,'filename':entry['title'],'duration_ms':motion['duration_ms'],'quality':quality})
    registered=store.register(job_id,entry['concept_fr'],[],entry['credit']+' · '+entry['source_url'],entry['license'])
    with store.library_lock():
        entries=store.library()
        for item in entries:
            if item['motion']['motion_id']==registered['motion']['motion_id']:
                item['motion']['rights_status']='cleared'
                item['attribution']=motion['attribution'];item['source_id']=entry['source_id']
        store.write_json(store.storage()/'library.json',entries)
    return {'source_id':entry['source_id'],'concept':entry['concept_fr'],'status':'imported','motion_id':registered['motion']['motion_id'],'quality':quality}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true');parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--all',action='store_true',help='Explicitly retrieve and extract all eligible inventory clips')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--author-mirror',action='store_true',help='Use only exact concept/original-subtitle matched author repository renditions')
    args=parser.parse_args();saved=ROOT/'data/source_inventory/commons_lsf.json'
    entries=inventory() if args.refresh or not saved.exists() else json.loads(saved.read_text())['entries']
    if args.author_mirror: entries=prepare_author_mirrors(entries)
    if not args.all and not args.limit: return
    eligible=[e for e in entries if e['eligible_for_motion'] and e['concept_fr']!='bonjour']
    if args.author_mirror: eligible.sort(key=lambda e:(not bool(e.get('author_mirror')),e['concept_fr']))
    if not args.all: eligible=eligible[:args.limit]
    results=[];state={'status':'running','total':len(eligible),'processed':0,'imported':0,'failed':0,'results':results}
    store.write_json(store.storage()/'corpus_status.json',state)
    with ThreadPoolExecutor(max_workers=max(1,min(4,args.workers))) as pool:
        pending={pool.submit(ingest,entry):entry for entry in eligible}
        for future in as_completed(pending):
            entry=pending[future]
            try: outcome=future.result()
            except Exception as exc: outcome={'source_id':entry['source_id'],'concept':entry['concept_fr'],'status':'failed','error':str(exc)}
            results.append(outcome);state['processed']+=1
            if outcome['status']=='imported': state['imported']+=1
            if outcome['status']=='failed': state['failed']+=1
            store.write_json(store.storage()/'corpus_status.json',state)
            print(f"[{state['processed']}/{state['total']}] {entry['concept_fr']}: {outcome['status']}",flush=True)
    state['status']='complete';store.write_json(store.storage()/'corpus_status.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
