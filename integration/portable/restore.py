#!/usr/bin/env python3
"""Keep this BOOTSTRAP.py alongside the supplied ZIP and checksum on the destination."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import platform
import shutil
import sys
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--start',action='store_true');p.add_argument('--archive',type=Path);a=p.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
        raise SystemExit('Run on Linux x86-64 (Windows: WSL2). The phone is transfer storage; native macOS/Windows/ARM need other runtimes.')
    here=Path(__file__).resolve().parent
    archive=a.archive or here/'SignaCulture_offline_linux_x86_64.zip'
    app=here/'SignaCulture'
    if not (app/'BUNDLE_MANIFEST.json').is_file():
        if app.exists() and any(app.iterdir()):raise SystemExit('Existing non-bundle SignaCulture folder: choose an empty destination.')
        if shutil.disk_usage(here).free < 8_000_000_000:raise SystemExit('Need at least 8 GB additional free space after copying the ZIP, about 12 GB before copying it.')
        expected=archive.with_suffix('.sha256').read_text().split()[0]
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=expected:raise SystemExit('ZIP checksum mismatch; do not extract.')
        print('ZIP checksum verified; extracting the ready-made app.',flush=True)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not (here/info.filename).resolve().is_relative_to(here):raise SystemExit('Unsafe archive path')
            z.extractall(here)
    # Python's standard ZIP extraction does not retain Unix executable bits.
    # The included bootstrap repairs interpreters; repair this ELF loader too.
    for loader in (app/'runtimes/ffprobe/lib').glob('ld-linux*'):
        if loader.is_file():loader.chmod(loader.stat().st_mode|0o111)
    command=[sys.executable,str(app/'integration/portable/bootstrap.py'),'--check' if a.check else '--start']
    raise SystemExit(subprocess.call(command,cwd=app))


if __name__=='__main__':main()
