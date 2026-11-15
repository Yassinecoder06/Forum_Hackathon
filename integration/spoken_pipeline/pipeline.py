"""Pinned upstream France-LSF lexical lookup -> masked poses -> 2D diagnostic."""
import argparse,contextlib,copy,hashlib,io,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import simplemma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import normalize_pose_size
from spoken_to_signed.gloss_to_pose.lookup import CSVPoseLookup
from spoken_to_signed.gloss_to_pose.concatenate import normalize_pose,join_poses
from spoken_to_signed.text_to_gloss.types import GlossItem
from pose_adapter import BASE,DEFAULT_MAIN,UPSTREAM_REV,export_lexicon,mask_missing,fold,dump,sha

TOKEN=re.compile(r'[^\W_]+',re.UNICODE)
def tokenize(s):
    return [(m.group(),fold(simplemma.lemmatize(m.group().casefold(),lang='fr'))) for m in TOKEN.finditer(s)]
def select(records):
    best_priority=min(r['priority'] for r in records)
    records=[r for r in records if r['priority']==best_priority]
    canonical={fold(r['gloss']) for r in records}
    if len(canonical)>1:return None,sorted({r['gloss'] for r in records})
    return sorted(records,key=lambda r:(r['priority'],r['id']))[0],[]
def resolve(manifest,text=None,glosses=None):
    records=manifest['records'];selected=[];unknown=[];ambiguous=[]
    if glosses is not None:
        for token in glosses:
            matches=[r for r in records if fold(r['gloss'])==fold(token)]
            if not matches:unknown.append(token);continue
            best,choices=select(matches)
            if choices:ambiguous.append(dict(token=token,canonical_glosses=choices))
            else:selected.append(best)
    else:
        tokens=tokenize(text);i=0
        while i<len(tokens):
            matches=[]
            for r in records:
                for phrase in dict.fromkeys([r['concept'],*r.get('aliases',[])]):
                    pt=tokenize(phrase)
                    if not pt:continue
                    if [x[0].casefold() for x in tokens[i:i+len(pt)]]==[x[0].casefold() for x in pt] or [x[1] for x in tokens[i:i+len(pt)]]==[x[1] for x in pt]:matches.append((len(pt),r))
            if not matches:unknown.append(tokens[i][0]);i+=1;continue
            length=max(n for n,r in matches);best,choices=select([r for n,r in matches if n==length])
            if choices:ambiguous.append(dict(token=' '.join(t[0] for t in tokens[i:i+length]),canonical_glosses=choices))
            else:selected.append(best)
            i+=length
    return selected,unknown,ambiguous

def nearest_resample(pose,fps):
    # Repeat nearest observed frames; NEVER fill a missing point through interpolation.
    n=max(1,round(len(pose.body.data)*fps/pose.body.fps))
    indexes=np.clip(np.round(np.arange(n)*pose.body.fps/fps).astype(int),0,len(pose.body.data)-1)
    result=Pose(copy.deepcopy(pose.header),NumPyPoseBody(fps,pose.body.data[indexes].copy(),pose.body.confidence[indexes].copy()))
    return mask_missing(result),indexes.tolist()

def run(lexicon,out,text=None,glosses=None,fps=25.,video=True):
    start=time.monotonic();lexicon=Path(lexicon);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((lexicon/'manifest.json').read_text());selected,unknown,ambiguous=resolve(manifest,text,glosses)
    report=dict(text=text,input_glosses=glosses,spoken_language='fr',signed_language='fsl',language_variant='France LSF, not LSF-CH',candidate=True,translation_validated=False,linguistic_review='pending',unknowns=unknown,ambiguities=ambiguous,glosses=[r['gloss'] for r in selected],selected_ids=[r['id'] for r in selected],sources=[dict(id=r['id'],source_id=r['source_id'],source_url=r['source_url'],license=r['license'],rights_status=r['rights_status']) for r in selected],local_preview_only=any(r['local_preview_only'] for r in selected),upstream_revision=UPSTREAM_REV,lexical_prototype=True,lexicon_fingerprint=manifest.get('fingerprint',manifest['source_library_sha256']),warnings=['Lexical source-order lookup is not linguistic LSF translation','No fingerspelling, synthetic fixture or language fallback','Diagnostic 2D skeleton; not a validated avatar or complete paper reproduction'])
    if unknown or ambiguous or not selected:
        report.update(status='blocked_unknown_or_ambiguous',seconds=time.monotonic()-start);dump(out/'result.json',report);return report
    lookup=CSVPoseLookup(str(lexicon),backup=None);poses=[];segments=[]
    # Use the actual upstream lexical API, constrained to the selected canonical/priority variant.
    # Per-entry lookup rows avoid silently choosing a different ambiguity or variant.
    for r in selected:
        rows=[dict(words=r['concept'],glosses=r['gloss'],spoken_language='fr',signed_language='fsl',path=r['pose_path'],start=0,end=int(np.ceil(r['frames']/r['fps']*1000)),priority=0)]
        from spoken_to_signed.gloss_to_pose.lookup.lookup import PoseLookup
        specific=PoseLookup(rows,directory=str(lexicon),backup=None)
        result=specific.lookup_sequence([GlossItem(r['concept'],r['gloss'])],'fr','fsl')[0]
        pose=copy.deepcopy(result.pose);before=len(pose.body.data);native_conf=pose.body.confidence.copy()
        # Real upstream shoulder normalization; does not invent absent landmarks.
        pose=normalize_pose(pose);pose,indexes=nearest_resample(pose,fps);poses.append(pose)
        segments.append(dict(id=r['id'],native_frames=before,native_fps=r['fps'],output_frames=len(indexes),source_frame_indexes=indexes,source_pose_sha256=r['pose_sha256'],origin=r['origin']['format']))
    # Actual upstream NumPy masked-array concatenation, with no smoothing/imputation.
    combined=join_poses(poses);expected_conf=np.concatenate([p.body.confidence for p in poses])
    assert np.array_equal(combined.body.confidence,expected_conf)
    normalize_pose_size(combined,target_width=512);combined=mask_missing(combined)
    pose_path=out/'sequence.pose'
    with pose_path.open('wb') as f:combined.write(f)
    with pose_path.open('rb') as f:readback=Pose.read(f.read())
    assert np.array_equal(readback.body.confidence==0,combined.body.confidence==0)
    report.update(status='rendered' if video else 'pose_exported',pose_path=str(pose_path.resolve()),pose_sha256=sha(pose_path),frames=len(combined.body.data),fps=float(combined.body.fps),segments=segments,confidence_semantics='Original visibility where available, otherwise source-presence flag; no per-point confidence fabrication',masked_samples=int((combined.body.confidence==0).sum()),tracked_samples=int((combined.body.confidence>0).sum()),methods_reused=['PoseLookup.lookup_sequence','normalize_pose','join_poses','pose_format.normalize_pose_size','Pose.write/read','PoseVisualizer.draw/save_video/save_gif'],custom_adaptations=['Native JSON/raw-Animics mapping and masks','French lexical matcher with ambiguity/unknown rejection','Explicit Dicta local-preview gate and source preference','Nearest-observation frame-rate alignment preserving masks'])
    if video:
        visualizer=PoseVisualizer(readback,thickness=2)
        mp4=out/'diagnostic.mp4';gif=out/'diagnostic.gif'
        visualizer.save_video(str(mp4),visualizer.draw(),custom_ffmpeg=os.getenv('SIGNACULTURE_FFMPEG') or shutil.which('ffmpeg') or '/usr/bin/ffmpeg')
        visualizer.save_gif(str(gif),visualizer.draw())
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,nb_read_frames,r_frame_rate','-show_entries','format=duration','-of','json',str(mp4)],text=True))
        report.update(video_path=str(mp4.resolve()),gif_path=str(gif.resolve()),video_sha256=sha(mp4),gif_sha256=sha(gif),ffprobe=probe)
    report['seconds']=time.monotonic()-start;dump(out/'result.json',report);return report

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--output',type=Path,required=True);e.add_argument('--library-root',type=Path,default=DEFAULT_MAIN);e.add_argument('--allow-local-preview',action='store_true')
    a=sub.add_parser('render');a.add_argument('--lexicon',type=Path,required=True);a.add_argument('--output',type=Path,required=True);group=a.add_mutually_exclusive_group(required=True);group.add_argument('--text');group.add_argument('--glosses',nargs='+');a.add_argument('--fps',type=float,default=25.);a.add_argument('--pose-only',action='store_true')
    args=p.parse_args()
    if args.command=='export':
        r=export_lexicon(args.output,args.library_root,args.allow_local_preview);print(json.dumps({k:r[k] for k in ['counts','export_errors','allow_local_preview','seconds']},ensure_ascii=False));return 0
    if not 1<=args.fps<=60:raise ValueError('Output fps must be1..60')
    with contextlib.redirect_stdout(sys.stderr):r=run(args.lexicon,args.output,args.text,args.glosses,args.fps,not args.pose_only)
    print(json.dumps(r,ensure_ascii=False));return 0 if r['status'] in ['rendered','pose_exported'] else 2
if __name__=='__main__':sys.exit(main())
