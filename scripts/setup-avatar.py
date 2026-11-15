#!/usr/bin/env python3
"""Download a checksum-pinned public UPF demo avatar for local review only."""
import hashlib
import os
from pathlib import Path
import urllib.request
root=Path(__file__).resolve().parents[1]
folder=Path(os.getenv('SIGNACULTURE_STORAGE',root/'.runtime'))/'avatar-candidates'
folder.mkdir(parents=True,exist_ok=True)
url='https://resources.gti.upf.edu/3Dcharacters/Noa/Noa.glb'
expected='610f0b246a3239a0bc8bc8989454ce03c87b7a47fda723bd3a0266e141ca3cbe'
raw=urllib.request.urlopen(url,timeout=90).read()
if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Upstream model changed: review before use.')
(folder/'Noa.glb').write_bytes(raw)
print('Local review model installed. See docs/AVATAR_PROVENANCE.md; redistribution/publication rights remain pending.')
