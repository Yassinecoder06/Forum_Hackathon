"""Run on Colab after Google Drive authorization; refuses an absent mount."""
from pathlib import Path
import hashlib, shutil
source=Path('/content/signaculture-acl2023-lsfb')
drive=Path('/content/drive/MyDrive')
if not drive.is_dir():raise SystemExit('Google Drive is not mounted; authorize it before saving.')
target=drive/'SignaCulture/acl2023-lsfb-v1'
for p in source.rglob('*'):
 if not p.is_file():continue
 dest=target/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(p.read_bytes()).digest():raise SystemExit(f'Refusing to overwrite different backup: {dest}')
 shutil.copy2(p,dest)
 assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest()
print('Verified Drive backup:',target)
