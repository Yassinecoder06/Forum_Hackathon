from pathlib import Path
import json,csv,collections
from huggingface_hub import hf_hub_download
p=Path('/content/fsl-audit');rev=json.load(open(p/'hub_inventory.json'))['revision']
for name in ['runtime_state/archive_upload_progress.json','runtime_state/SignVerse-2M-metadata_processed.csv','debug/croissant_shard_manifest.csv','scripts/pipeline03_upload_to_huggingface.py','scripts/pipeline02_extract_dwpose_from_video.py','scripts/pipeline01_download_video_fix_caption.py','scripts/visualize_dwpose_npz.py']:
 path=hf_hub_download('SignerX/SignVerse-2M',name,repo_type='dataset',revision=rev,token=False,local_dir=p)
 print('FETCHED',name,Path(path).stat().st_size,flush=True)
j=json.load(open(p/'runtime_state/archive_upload_progress.json'));print('INDEX_TYPE',type(j).__name__);print('INDEX_SAMPLE',str(j)[:3000])
r=list(csv.DictReader(open(p/'runtime_state/SignVerse-2M-metadata_processed.csv')));print('PROCESSED_SCHEMA',list(r[0]),'SAMPLE',r[0]);print('FSL_PROCESSED',sum(x.get('sign_language')=='fsl' for x in r))
f=json.load(open(p/'hub_inventory.json'))['files'];sh=[x for x in f if x['name'].endswith('.tar')];print('SHARDS',len(sh),'TOTAL_BYTES',sum(x['size'] for x in sh),'MIN',min(x['size'] for x in sh),'MAX',max(x['size'] for x in sh))
