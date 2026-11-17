"""Save actual API plans/native captured assets for repeatable renderer diagnostics."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PHRASES=["Je veux boire de l’eau.","Je visite Paris avec ma famille.","Bonjour, comment ça va ?","Le musée est fermé demain.","Je veux apprendre la langue des signes.","bibliothèque"]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--api',default='http://127.0.0.1:8320');args=parser.parse_args()
    out=ROOT/'.runtime/motion-quality/current-cases';out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'.runtime/library.json').read_text());byid={r['motion']['motion_id']:r for r in rows}
    manifest=[]
    for i,text in enumerate(PHRASES,1):
        req=urllib.request.Request(args.api+'/api/compose-animation',data=json.dumps({'text':text}).encode(),headers={'Content-Type':'application/json'})
        plan=json.load(urllib.request.urlopen(req,timeout=60));assets=[];hashes={}
        if not plan['segments']:raise ValueError('No actual captured signs for '+text)
        for segment in plan['segments']:
            identifier=segment['motion_id'];p=ROOT/byid[identifier]['motion']['path'];data=p.read_bytes();motion=json.loads(data)
            if motion.get('synthetic'):raise ValueError('Synthetic diagnostic input is forbidden')
            assets.append([identifier,motion]);hashes[identifier]=hashlib.sha256(data).hexdigest()
        (out/f'{i:02d}-input.json').write_text(json.dumps({'text':text,'plan':plan,'assets':assets},ensure_ascii=False))
        manifest.append({'id':i,'text':text,'asset_sha256':hashes,'duration_ms':plan['duration_ms']})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':main()
