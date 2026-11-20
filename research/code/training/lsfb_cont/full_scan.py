"""Scan local LSFB source integrity, extract text and paired Belgian pose crops."""
import argparse,collections,hashlib,io,json,tarfile,time,gzip
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .prepare import lexical_projection_events,make_rows,render_input,conversation_id
from .experiments import normalize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/jmal/Dataset/lsfb-cont'))
    ap.add_argument('--output',type=Path,default=Path('/home/jmal/Dataset/lsfb-cont-usable-v1'))
    ap.add_argument('--resume-verified-scan',action='store_true');args=ap.parse_args()
    root,out=args.root,args.output
    if out.exists() and not args.resume_verified_scan:raise ValueError('Output exists; choose a new frozen directory')
    out.mkdir(exist_ok=True);(out/'poses').mkdir(exist_ok=True);started=time.monotonic()
    def save(name,x):
        p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
    def rows(name,x):
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
    revision='64b0c112f59d3710367fb736ef1808f37ff15e28'
    expected=json.load(open(root/'.cache/huggingface/trees'/f'{revision}.json'))['files']
    inventory=[];pose_index={};bad_records=set();all_records=collections.defaultdict(dict)
    if args.resume_verified_scan:
        assert not (out/'SCAN_REPORT.json').exists(), 'Frozen extraction already exists'
        inventory=json.load(open(out/'inventory.json'));pose_index=json.load(open(out/'pose_index.json'))
        assert len(inventory)==len(expected) and all(i['status']=='verified' and (root/i['path']).stat().st_size==i['size'] and (not expected[i['path']].get('lfs_sha256') or i['sha256']==expected[i['path']]['lfs_sha256']) for i in inventory)
        bad_records={rid for rid,parts in pose_index.items() if any(not p['structurally_valid'] for p in parts.values())}
    for relative,spec in ([] if args.resume_verified_scan else expected.items()):
        path=root/relative
        if not path.is_file():inventory.append(dict(path=relative,status='missing'));continue
        sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        ok=path.stat().st_size==spec['size'] and (not spec.get('lfs_sha256') or sha==spec['lfs_sha256'])
        result=dict(path=relative,size=path.stat().st_size,sha256=sha,status='verified' if ok else 'mismatch',expected_sha256=spec.get('lfs_sha256'))
        inventory.append(result);save('inventory.json',inventory)
        save('progress.json',dict(status='scanning',file=relative,completed_files=len(inventory),total_files=len(expected),seconds=time.monotonic()-started))
        print('verified',relative,ok,flush=True)
        if not ok:raise ValueError('Source integrity mismatch: '+relative)
        if not relative.startswith('shards/') or not relative.endswith('.tar'):continue
        # Test labels remain unopened. Pose structure/integrity is inventoried only.
        with tarfile.open(path) as tar:
            for member in tar:
                if '.pose.' not in member.name or not member.name.endswith('.npy'):continue
                rid,part=member.name.split('.pose.',1);part=part[:-4]
                try:
                    payload=tar.extractfile(member).read();stream=io.BytesIO(payload)
                    version=np.lib.format.read_magic(stream)
                    shape,fortran,dtype=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(stream)
                    offset=member.offset_data+stream.tell();a=np.load(io.BytesIO(payload),allow_pickle=False)
                    finite=float(np.isfinite(a).mean());nonzero=float(np.any(a!=0,axis=tuple(range(1,a.ndim))).mean())
                    valid=a.ndim==3 and shape[-1] in (2,3,4) and shape[0]>0 and dtype.kind=='f'
                    info=dict(tar=relative,member=member.name,offset=offset,shape=list(shape),dtype=str(dtype),fortran=fortran,finite_fraction=finite,nonzero_frame_fraction=nonzero,structurally_valid=valid)
                # Annotated shards sort first; preserve that authoritative copy
                # if the same recording also appears in pose-only archives.
                pose_index.setdefault(rid,{}).setdefault(part,info)
                    if not valid:bad_records.add(rid)
                except Exception as error:
                    bad_records.add(rid);all_records[rid]['pose_error']=str(error)
        save('pose_index.json',pose_index)
    # Reconstruct every source candidate from previously verified thin snapshots.
    records={}
    for path in sorted(Path('.runtime/lsfb-cont-task-v3/sources').glob('source_*.json.gz')):
        x=json.load(gzip.open(path,'rt'))
        rel=f"shards/annotated/shard_{x['shard']:06d}.tar"
        assert x['tar_sha256']==next(i['sha256'] for i in inventory if i['path']==rel)
        for rid,r in x['records'].items():
            assert r['language.txt']=='lsfb'
            records[rid]={**r,'source_shard':x['shard'],'official_role':'validation' if x['shard']==1 else 'train','conversation_id':conversation_id(rid)}
    candidates=make_rows(records);tok=AutoTokenizer.from_pretrained('.runtime/lsfb-cont-local-v3/tokenizer',local_files_only=True)
    kept=[];rejected=[];removed=0
    for r in candidates:
        reasons=[x for x in r['technical_exclusions'] if x in ['conversation_present_in_official_validation','no_aligned_sign_events','lexical_event_missing_author_lemma']]
        events,duplicates=lexical_projection_events(r['aligned_events']);removed+=len(duplicates)
        if not events:reasons.append('no_lexical_events')
        if any(e.get('specifier')=='fs' for e in events):reasons.append('fingerspelling_requires_separate_task')
        if any(a['end_ms']>b['start_ms'] for a,b in zip(events,events[1:])):reasons.append('simultaneous_lexical_events_require_review')
        if r['boundary_excluded_events']:reasons.append('uncertain_segment_boundary')
        if not 1<=len(events)<=16:reasons.append('lexical_length_outside_1_to_16')
        if len(r['french_text'].split())>60 or r['interval_ms'][1]-r['interval_ms'][0]>30000:reasons.append('long_segment')
        if 'ellipsis_or_fragment' in r['review_flags'] or '[' in r['french_text']:reasons.append('fragment_or_editorial_annotation')
        pose=pose_index.get(r['recording_id'],{})
        if r['recording_id'] in bad_records or not all(k in pose for k in ['upper_pose','left_hand','right_hand']):reasons.append('missing_or_invalid_pose')
        rawtr=records[r['recording_id']]['annotations.translation.json'][int(r['id'].rsplit(':',1)[-1])]
        a,b=rawtr.get('start_frame'),rawtr.get('end_frame')
        if not isinstance(a,int) or not isinstance(b,int) or b<=a or any(b>p['shape'][0] for p in pose.values()):reasons.append('pose_frame_bounds')
        target=[e['lemma'].strip() for e in events]
        target_text=json.dumps(target,ensure_ascii=False)
        context=list(r['context']);text=render_input(r['french_text'],r['signer_id'],context)
        while context and len(tok.encode(text))>384:
            context.pop(0);text=render_input(r['french_text'],r['signer_id'],context)
        ids=tok.encode(target_text)
        try:roundtrip=json.loads(tok.decode(ids,skip_special_tokens=True))==target
        except ValueError:roundtrip=False
        if not roundtrip or tok.unk_token_id in ids or len(ids)>192 or len(tok.encode(text))>384:reasons.append('tokenizer_or_length')
        item=dict(id=r['id'],language='LSFB',split=r['split'],conversation_id=r['conversation_id'],signer_id=r['signer_id'],
                  input=text,current_french=r['french_text'],target=target,target_text=target_text,
                  interval_ms=r['interval_ms'],frames=[a,b],recording_id=r['recording_id'],
                  source_shard=r['source_shard'],review_status='pending',review_flags=r['review_flags'],
                  raw_events=r['aligned_events'],collapsed_simultaneous_events=duplicates,
                  license='CC-BY-NC-SA-4.0',exclusions=sorted(set(reasons)))
        (rejected if reasons else kept).append(item)
    # Remove training examples whose current French is present in validation, regardless of context.
    val_inputs={normalize(r['current_french']) for r in kept if r['split']=='val'}
    final=[];seen=set()
    for r in kept:
        key=(r['input'],r['target_text'])
        if r['split']=='train' and normalize(r['current_french']) in val_inputs:r['exclusions'].append('current_french_shared_with_validation')
        elif r['split']=='train' and key in seen:r['exclusions'].append('duplicate_training_input_target')
        seen.add(key)
        (rejected if r['exclusions'] else final).append(r)
    train=[r for r in final if r['split']=='train'];val=[r for r in final if r['split']=='val']
    assert train and val
    assert not {r['conversation_id'] for r in train}&{r['conversation_id'] for r in val}
    assert not {r['signer_id'] for r in train}&{r['signer_id'] for r in val}
    assert not {normalize(r['current_french']) for r in train}&{normalize(r['current_french']) for r in val}
    # Copy actual Belgian pose segments, not any app motion library.
    for i,r in enumerate(final):
        arrays={};a,b=r['frames'];quality={}
        for part,spec in pose_index[r['recording_id']].items():
            mm=np.memmap(root/spec['tar'],dtype=spec['dtype'],mode='r',offset=spec['offset'],shape=tuple(spec['shape']),order='F' if spec['fortran'] else 'C')
            crop=np.array(mm[a:b]);arrays[part]=crop
            quality[part]=dict(finite=float(np.isfinite(crop).mean()),nonzero_frames=float(np.any(crop!=0,axis=(1,2)).mean()))
            del mm
        filename=f'poses/{i:06d}.npz';np.savez_compressed(out/filename,**arrays)
        r['pose_path']=filename;r['pose_quality']=quality
        if i%100==0:save('progress.json',dict(status='extracting_lsfb_pose_crops',completed=i,total=len(final)))
    rows('train.jsonl',train);rows('val.jsonl',val);rows('quarantine.jsonl',rejected)
    vocab=collections.Counter(t for r in train for t in r['target'])
    report=dict(language='LSFB',status='complete',linguistically_validated=False,source_root=str(root),source_revision=revision,
                inventory_files=len(inventory),pose_recordings=len(pose_index),bad_pose_recordings=sorted(bad_records),
                candidates=len(candidates),train=len(train),val=len(val),quarantined=len(rejected),
                exact_simultaneous_duplicates_removed=removed,exclusion_counts=dict(collections.Counter(x for r in rejected for x in r['exclusions'])),
                train_vocabulary=len(vocab),val_oov=sum(t not in vocab for r in val for t in r['target']),
                val_operations=sum(len(r['target']) for r in val),official_test_labels_opened=False,
                pose_only_recordings_not_used_as_text_supervision=sum(rid not in records for rid in pose_index),
                seconds=time.monotonic()-started,files={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['train.jsonl','val.jsonl','quarantine.jsonl']})
    save('SCAN_REPORT.json',report);save('progress.json',report)
    (out/'README.md').write_text('# Belgian LSFB usable extraction\n\nTechnically eligible, linguistically unreviewed lexical projection. train.jsonl and val.jsonl contain French context, Belgian LSFB author lemmas and relative paths to actual paired Belgian pose crops. quarantine.jsonl retains excluded candidates and reasons. Original raw events are retained in each row. No France-LSF assets are used. Official test labels remain unopened. Full videos were SHA-256 verified, not decoded frame by frame. Nonlexical features remain source annotations; this is not full signing grammar.\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
