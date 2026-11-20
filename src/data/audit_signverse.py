import csv,json,collections,hashlib,zipfile
from pathlib import Path
p=Path('/content/fsl-audit');inv=json.load(open(p/'hub_inventory.json'));sizes={x['name'].split('/')[-1]:x['size'] for x in inv['files'] if x['name'].endswith('.tar')};split=list(csv.DictReader(open(p/'SignVerse-2M-metadata_split.csv')));raw=list(csv.reader(open(p/'Sign-DWPose-2M-metadata_ori.csv')));ori={r[0]:r[1] for r in raw};processed={x['video_id']:x for x in csv.DictReader(open(p/'runtime_state/SignVerse-2M-metadata_processed.csv'))};idx=json.load(open(p/'runtime_state/archive_upload_progress.json'))['archives'];mapping=collections.defaultdict(list)
for name,obj in idx.items():
 for video in obj['folders']:mapping[video].append(name)
manifest=[]
for row in split:
 if row['sign_language']!='fsl':continue
 assert ori[row['video_id']]=='fsl'
 extra=processed.get(row['video_id'],{});assert extra.get('sign_language')=='fsl'
 archives=[name for name in mapping[row['video_id']] if name in sizes];declared=extra.get('archive_name');
 if declared in sizes and declared not in archives:archives.append(declared)
 manifest.append(dict(sign_language='fsl',video_id=row['video_id'],source='SignerX/SignVerse-2M',revision=inv['revision'],split=row['split'],source_video_id=row['video_id'],subtitle_language=extra.get('subtitle_languages') or None,transcription_source=extra.get('subtitle_en_source') or None,signer_id=None,recording_start_s=extra.get('start_sec') or None,recording_end_s=extra.get('end_sec') or None,duration_s=extra.get('duration_sec') or None,license='CC-BY-NC-4.0',permission_status='derived_pose_metadata_noncommercial_research_only; original_video_rights_not_conveyed',archives=archives,archive_bytes=[sizes[x] for x in archives],download_status=extra.get('download_status'),pose_status=extra.get('process_status'),subtitle_status=extra.get('subtitle_status') or None,record_language_verified=False,gloss_status='not_provided',deletion_status='no_local_request_received'))
(p/'fsl_manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in manifest))
summary={'revision':inv['revision'],'metadata_rows':len(split),'original_rows_headerless':len(raw),'fsl_videos':len(manifest),'splits':dict(collections.Counter(x['split'] for x in manifest)),'mapped_videos':sum(bool(x['archives']) for x in manifest),'missing_archive_videos':sum(not x['archives'] for x in manifest),'duration_known':sum(x['duration_s'] is not None for x in manifest),'subtitle_language_known':sum(x['subtitle_language'] is not None for x in manifest),'pose_status':dict(collections.Counter(x['pose_status'] for x in manifest)),'shard_count':len(sizes),'total_shard_bytes':sum(sizes.values()),'fsl_shards':len(set(x for r in manifest for x in r['archives'])),'schema_original':'headerless two-column video_id,sign_language; first row retained','segments':None,'usable_duration_s':None}
(p/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
selected=[]
for part in sorted(set(r['split'] for r in manifest)):
 selected.extend(sorted([r for r in manifest if r['split']==part and r['archives']],key=lambda r:min(r['archive_bytes']))[:3])
(p/'selected.json').write_text(json.dumps(selected,indent=2));print('SELECTED',[(r['split'],r['video_id'],r['archives'],r['archive_bytes']) for r in selected])
with zipfile.ZipFile('/content/fsl-audit.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in p.rglob('*'):
  if f.is_file() and '.cache' not in f.parts:z.write(f,str(f.relative_to(p)))
