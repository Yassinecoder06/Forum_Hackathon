#!/usr/bin/env python3
"""Download checksum-pinned public UPF demo avatars for local review only."""
import hashlib
import os
from pathlib import Path
import urllib.request
root=Path(__file__).resolve().parents[1]
folder=Path(os.getenv('SIGNACULTURE_STORAGE',root/'.runtime'))/'avatar-candidates'
folder.mkdir(parents=True,exist_ok=True)
# ReadyEva is the default studio character; Noa remains the measured reference rig
# used by frontend/scripts/measure-*.ts. Both share the same skeleton and ARKit morphs.
MODELS={
    'ReadyEva':('https://resources.gti.upf.edu/3Dcharacters/ReadyEva/ReadyEva.glb',
                'b6dc3c8e34df93aad378d24daa38f305c753f45fac147608fc0149218812bbf0'),
    'Noa':('https://resources.gti.upf.edu/3Dcharacters/Noa/Noa.glb',
           '610f0b246a3239a0bc8bc8989454ce03c87b7a47fda723bd3a0266e141ca3cbe'),
}
for name,(url,expected) in MODELS.items():
    target=folder/f'{name}.glb'
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
        print(f'{name}: already installed.')
        continue
    raw=urllib.request.urlopen(url,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError(f'Upstream model {name} changed: review before use.')
    target.write_bytes(raw)
    print(f'{name}: installed.')
print('Local review models installed. See docs/avatar/AVATAR_PROVENANCE.md; redistribution/publication rights remain pending.')
