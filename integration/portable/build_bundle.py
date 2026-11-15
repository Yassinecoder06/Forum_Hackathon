#!/usr/bin/env python3
"""Snapshot only the live app's resources into a private, relocatable offline ZIP."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target,follow_symlinks=False)


def tree(source,target,extra=()):
    shutil.copytree(source,target,dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git','.cache',*extra))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare(stage):
    stage.mkdir(parents=True,exist_ok=True)
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for relative in files:
        source=ROOT/relative
        if not relative or not source.is_file() or relative.startswith(('artifacts/','research/archive/')):continue
        copy_file(source,stage/relative)
    tree(ROOT/'integration/portable',stage/'integration/portable')
    tree(ROOT/'frontend/dist',stage/'frontend/dist')
    runtime=stage/'.runtime';runtime.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'.runtime/library.json').read_text())
    raw_map={};raw_count=0
    for i,row in enumerate(rows):
        motion=Path(row['motion']['path']);motion=motion if motion.is_absolute() else ROOT/motion
        target=runtime/'motions'/motion.name;copy_file(motion,target)
        row['motion']['path']=str(target.relative_to(stage))
        raw=Path(row['raw_animics_path']);raw=raw if raw.is_absolute() else ROOT/raw
        if raw not in raw_map:
            value=digest(raw);raw_target=runtime/'active_raw'/(value+'.json.gz')
            if not raw_target.exists():copy_file(raw,raw_target)
            raw_map[raw]=str(raw_target.relative_to(stage));raw_count+=1
        row['raw_animics_path']=raw_map[raw]
        source=ROOT/'.runtime/captures'/row['capture_id'];folder=runtime/'captures'/row['capture_id'];folder.mkdir(parents=True,exist_ok=True)
        for name in ['source.mp4','source_provenance.json','status.json']:
            if (source/name).is_file():copy_file(source/name,folder/name)
        if i%250==0:print('Snapshot captures',i,'/',len(rows),flush=True)
    (runtime/'library.json').write_text(json.dumps(rows,ensure_ascii=False))
    for name in ['avatar-candidates','dictasign']:
        if (ROOT/'.runtime'/name).exists():tree(ROOT/'.runtime'/name,runtime/name)
    for name in ['corpus_status.json','dictasign_status.json']:
        if (ROOT/'.runtime'/name).is_file():copy_file(ROOT/'.runtime'/name,runtime/name)
    tree(ROOT/'.models/emotion2vec-seed',stage/'.models/emotion2vec-seed')
    for name in ['.venv','.venv-motion','.venv-audio','integration/spoken_pipeline/.venv']:
        print('Snapshot runtime',name,flush=True);tree(ROOT/name,stage/name)
    python_roots={}
    for version,label in [('3.11','python311'),('3.13','python313')]:
        interpreter=(ROOT/('.venv-motion' if version=='3.11' else '.venv')/'bin/python').resolve()
        source=interpreter.parent.parent;target=stage/'runtimes'/label
        tree(source,target);python_roots[str(source)]=str(target.relative_to(stage))
    tree(ROOT/'integration/spoken_pipeline/runtime/upstream',stage/'integration/spoken_pipeline/runtime/upstream',('*.pose',))
    (stage/'integration/spoken_pipeline/runtime/upstream/UPSTREAM_REVISION.txt').write_text('fec7ac393989f4b6541ab5f860819262a54f27c7\n')
    portable=stage/'portable';(portable/'bin').mkdir(parents=True,exist_ok=True)
    for kind,name in [('api','.venv'),('motion','.venv-motion'),('audio','.venv-audio'),('pose','integration/spoken_pipeline/.venv')]:
        result=subprocess.run(['uv','pip','freeze','--python',str(ROOT/name/'bin/python')],capture_output=True,text=True,check=True)
        (portable/f'requirements-{kind}.txt').write_text(result.stdout)
    # FFmpeg is the installed imageio static build. FFprobe brings its exact loader/DSOs.
    binaries=list((stage/'integration/spoken_pipeline/.venv').glob('lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*'))
    assert len(binaries)==1
    ffmpeg_relative=os.path.relpath(binaries[0],portable/'bin')
    (portable/'bin/ffmpeg').symlink_to(ffmpeg_relative)
    dependencies=subprocess.check_output(['ldd','/usr/bin/ffprobe'],text=True)
    ffdir=stage/'runtimes/ffprobe';(ffdir/'lib').mkdir(parents=True,exist_ok=True)
    loader=None
    for name in sorted(set(re.findall(r'/[^\s()]+',dependencies))):
        source=Path(name)
        if not source.is_file():continue
        resolved=source.resolve();copy_file(resolved,ffdir/'lib'/source.name)
        if 'ld-linux' in source.name:loader=source.name
    assert loader
    copy_file(Path('/usr/bin/ffprobe'),ffdir/'ffprobe.bin')
    wrapper='#!/usr/bin/env bash\nset -euo pipefail\nroot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"\nexec "$root/runtimes/ffprobe/lib/'+loader+'" --library-path "$root/runtimes/ffprobe/lib" "$root/runtimes/ffprobe/ffprobe.bin" "$@"\n'
    (portable/'bin/ffprobe').write_text(wrapper);(portable/'bin/ffprobe').chmod(0o755)
    for copyright in Path('/usr/share/doc').glob('*/copyright'):
        if copyright.is_file():copy_file(copyright,portable/'system-library-notices'/copyright.parent.name/'copyright')
    db=Path.home()/'.local/state/signaculture/plans.sqlite3'
    if db.is_file():
        a=sqlite3.connect(db);b=sqlite3.connect(runtime/'plans.sqlite3');a.backup(b);b.close();a.close()
    manifest={'format':'SignaCulture-private-offline-bundle-v1','source_root':str(ROOT),'build_root':str(stage),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'working_copy_changes_included':True,'platform':'Linux x86-64 GNU; tested Ubuntu 24.04','capture_count':len(rows),'unique_active_raw_paths':raw_count,'python_roots':python_roots,'launch':'python3 integration/portable/bootstrap.py --start','url':'http://127.0.0.1:8080','training_excluded':True,'qwen_in_pipeline':False,'linguistic_review':'pending','keypoint_accuracy_validated':False}
    (stage/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    omitted={'qwen':'Removed from application by explicit user request. No GGUF/server/startup runtime included. Historical reports are archival only.','training':'No training environments, checkpoints, Colab state or dataset shard caches; user asked not to work on training. Tracked source/report text remains.','duplicate_history':'Old raw/retargeted capture copies, CV experiment backups and duplicate model download caches omitted. Active raw + native assets included.','credentials':'No .git, SSH keys, account tokens, home caches or .env files.','platform':'No native Windows/macOS/ARM binaries; use Linux x86-64 or recreate platform dependencies from included requirements.'}
    (stage/'OMISSIONS.json').write_text(json.dumps(omitted,indent=2))
    resources=[{'resource':'emotion2vec/emotion2vec_plus_seed','revision':'7be9a30d52bdc793cc24e6f7d25358d6648927c1','path':'.models/emotion2vec-seed','model_sha256':digest(stage/'.models/emotion2vec-seed/model.pt'),'license':'Original model README/terms retained; no broader licence inferred','source':'https://huggingface.co/emotion2vec/emotion2vec_plus_seed'},{'resource':'spoken-to-signed-translation','revision':'fec7ac393989f4b6541ab5f860819262a54f27c7','source':'https://github.com/sign-language-processing/spoken-to-signed-translation','license':'UPSTREAM_LICENSE retained; no Swiss/German pretrained runtime/pose corpus included'},{'resource':'registered sign videos/raw/native motions','count':len(rows),'source':'per-entry .runtime/library.json and source_provenance.json','license':'Per-source licences retained; Dicta-Sign and Noa public redistribution remains pending. Private local research transfer, not public publication.'},{'resource':'Python/package/native runtimes','source':'Installed isolated runtimes and exact requirements snapshots','license':'Bundled installed licence notices and distro copyright files retained.'}]
    (stage/'THIRD_PARTY_RESOURCES.json').write_text(json.dumps(resources,indent=2))
    copy_file(ROOT/'integration/portable/agent_prompt.md',stage/'agent_prompt.md')
    (stage/'START_HERE.txt').write_text('Extract this ZIP onto a local Linux x86-64 computer (8+ GB free), not inside MTP.\nRead agent_prompt.md. Run: python3 integration/portable/bootstrap.py --check\nThen: python3 integration/portable/bootstrap.py --start\nOpen http://127.0.0.1:8080 . No CV extraction, model download, npm or training is required.\n')
    links=[{'path':str(p.relative_to(stage)),'target':os.readlink(p)} for p in sorted(stage.rglob('*')) if p.is_symlink()]
    (stage/'SYMLINKS.json').write_text(json.dumps(links,indent=2))
    print('STAGE_READY',stage,flush=True)


def archive(stage,output,wait_pose=False):
    links=[]
    for p in sorted(stage.rglob('*')):
        if p.is_symlink():links.append({'path':str(p.relative_to(stage)),'target':os.readlink(p)})
    (stage/'SYMLINKS.json').write_text(json.dumps(links,indent=2))
    paths=[p for p in sorted(stage.rglob('*')) if p.is_file() and not p.is_symlink() and (not wait_pose or not p.is_relative_to(stage/'.runtime/lexical_pipeline'))]
    total=sum(p.stat().st_size for p in paths);done=0
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for i,p in enumerate(paths):
            suffix=p.suffix.lower();method=zipfile.ZIP_STORED if suffix in ['.gz','.mp4','.webm','.jpg','.png','.zip'] else zipfile.ZIP_DEFLATED
            z.write(p,str(Path('SignaCulture')/p.relative_to(stage)),compress_type=method)
            done+=p.stat().st_size
            if i%500==0:print('ZIP',i,'/',len(paths),round(done/1e9,2),'/',round(total/1e9,2),'GB',flush=True)
    if wait_pose:
        manifest=stage/'.runtime/lexical_pipeline/lexicon/manifest.json'
        while not manifest.is_file():
            print('Waiting for final precomputed pose-cache manifest',flush=True);time.sleep(5)
        cache=json.loads(manifest.read_text());assert not cache['export_errors'],cache['export_errors']
        with zipfile.ZipFile(output,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
            for p in sorted((stage/'.runtime/lexical_pipeline').rglob('*')):
                if p.is_file() and not p.is_symlink():z.write(p,str(Path('SignaCulture')/p.relative_to(stage)))
    with zipfile.ZipFile(output) as check:
        paths=check.infolist();total=sum(info.file_size for info in paths)
    result={'zip':str(output),'bytes':output.stat().st_size,'sha256':digest(output),'files':len(paths),'uncompressed_bytes':total}
    output.with_suffix('.sha256').write_text(result['sha256']+'  '+output.name+'\n')
    output.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print('ARCHIVE_READY',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--zip',type=Path);p.add_argument('--archive-only',action='store_true');p.add_argument('--wait-pose-manifest',action='store_true');a=p.parse_args()
    if not a.archive_only:prepare(a.stage.resolve())
    if a.zip:archive(a.stage.resolve(),a.zip.resolve(),a.wait_pose_manifest)


if __name__=='__main__':main()
