#!/usr/bin/env python3
"""Relocate the included Linux x86-64 runtimes; no pip/network/CV regeneration."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]


def relocate():
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
        raise SystemExit('This ready-made runtime targets Linux x86-64 (Windows: use WSL2). Source is included for other platforms; these binary environments are not cross-platform.')
    meta=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())
    mappings={meta['source_root']:str(ROOT),meta['build_root']:str(ROOT)}
    mappings.update({k:str(ROOT/v) for k,v in meta['python_roots'].items()})
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(mappings,key=len,reverse=True)))
    replace=lambda value:pattern.sub(lambda m:mappings[m.group()],value)
    links=json.loads((ROOT/'SYMLINKS.json').read_text())
    for record in links:
        path=ROOT/record['path'];target=record['target']
        target=replace(target)
        resolved=(path.parent/target).resolve()
        if not resolved.is_relative_to(ROOT):raise ValueError('External symlink refused: '+record['path'])
        if path.is_symlink():path.unlink()
        elif path.exists():raise ValueError('Cannot overwrite a regular file with a link: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        path.symlink_to(os.path.relpath(resolved,path.parent))
    for name in ['.venv','.venv-motion','.venv-audio','integration/spoken_pipeline/.venv']:
        env=ROOT/name
        paths=[env/'pyvenv.cfg',*(env/'bin').glob('*'),*env.glob('lib/python*/site-packages/*.pth'),*env.glob('lib/python*/site-packages/__editable__*.py')]
        for path in paths:
            if not path.is_file() or path.is_symlink():continue
            try:text=path.read_text()
            except (UnicodeError,OSError):continue
            updated=replace(text)
            if updated!=text:path.write_text(updated)
        for path in (env/'bin').glob('*'):
            if path.is_file():path.chmod(path.stat().st_mode|0o111)
    for loader in (ROOT/'runtimes/ffprobe/lib').glob('ld-linux*'):
        if loader.is_file():loader.chmod(loader.stat().st_mode|0o111)
    for folder in ['runtimes/python311/bin','runtimes/python313/bin','runtimes/ffprobe','portable/bin']:
        for path in (ROOT/folder).glob('*'):
            if path.is_file():path.chmod(path.stat().st_mode|0o111)
    for path in (ROOT/'integration/spoken_pipeline/.venv').glob('lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'):
        path.chmod(path.stat().st_mode|0o111)
    lexicon=ROOT/'.runtime/lexical_pipeline/lexicon/manifest.json'
    if lexicon.is_file():
        def rebase(value):
            if isinstance(value,str):
                for old,new in sorted(mappings.items(),key=lambda item:len(item[0]),reverse=True):
                    if value.startswith(old):return new+value[len(old):]
            if isinstance(value,list):return [rebase(v) for v in value]
            if isinstance(value,dict):return {k:rebase(v) for k,v in value.items()}
            return value
        manifest=rebase(json.loads(lexicon.read_text()))
        # Record paths change; the pose content does not. Refresh the actual canonical fingerprint.
        sys.path.insert(0,str(ROOT/'integration/spoken_pipeline'))
        payload={k:manifest[k] for k in ['upstream_revision','language','spoken_language','allow_local_preview','source_library_sha256','records','blocked_pending_ids','export_errors']}
        import hashlib
        manifest['fingerprint']=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        lexicon.write_text(json.dumps(manifest,ensure_ascii=False))
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--start',action='store_true');a=p.parse_args()
    meta=relocate()
    env=os.environ.copy();env['PATH']=str(ROOT/'portable/bin')+os.pathsep+env.get('PATH','');env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    checks=[('.venv','import fastapi,uvicorn,spacy;spacy.load("fr_core_news_sm");print("API + French model ready")'),('.venv-motion','import mediapipe,cv2;print("MediaPipe ready")'),('.venv-audio','import torch,torchaudio,funasr;print("CPU audio ready")'),('integration/spoken_pipeline/.venv','import spoken_to_signed,pose_format;print("Pinned pose pipeline ready")')]
    if a.check:
        for folder,code in checks:subprocess.run([str(ROOT/folder/'bin/python'),'-c',code],cwd=ROOT,env=env,check=True)
        subprocess.run([str(ROOT/'portable/bin/ffmpeg'),'-version'],stdout=subprocess.DEVNULL,env=env,check=True)
        subprocess.run([str(ROOT/'portable/bin/ffprobe'),'-version'],stdout=subprocess.DEVNULL,env=env,check=True)
        print('Ready:',meta['capture_count'],'captures; zero extraction or model download requested.')
    if a.start:
        raise SystemExit(subprocess.call([str(ROOT/'.venv/bin/python'),str(ROOT/'integration/portable/serve.py')],cwd=ROOT,env=env))
    print('Launch: python3 integration/portable/bootstrap.py --start   then http://127.0.0.1:8080')


if __name__=='__main__':main()
