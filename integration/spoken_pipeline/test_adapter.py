"""Real-source geometry/masks, fsl lookup, source preference and encoded video checks."""
import csv,copy,gzip,json,os
from pathlib import Path
import numpy as np
import pytest
from pose_format import Pose
from spoken_to_signed.gloss_to_pose.lookup import CSVPoseLookup
from spoken_to_signed.gloss_to_pose.lookup.lookup import CoverageType
from spoken_to_signed.text_to_gloss.types import GlossItem
from pose_adapter import BASE,BODY,TOTAL,DEFAULT_MAIN,UPSTREAM_REV,motion_to_pose,raw_to_pose,sha
from pipeline import resolve,run
DEFAULT=BASE/'runtime/lexicon'
PREVIEW=Path(os.getenv('SIGNACULTURE_PREVIEW_LEXICON',str(DEFAULT_MAIN/'.runtime/lexical_pipeline/lexicon')))
def load(p):return json.loads(Path(p).read_text())
def read_pose(p):return Pose.read(Path(p).read_bytes())

@pytest.fixture(scope='session')
def rendered(tmp_path_factory):
    """Exercise the CLI engine on real exports; do not depend on old ignored demos."""
    output=tmp_path_factory.mktemp('real-lexical-renders')
    results={}
    for name,lexicon,text in [('cleared',DEFAULT,'Bonjour merci'),('preview',PREVIEW,'Bonjour merci'),('unknown',PREVIEW,'Bonjour motinconnu')]:
        folder=output/name
        run(lexicon,folder,text=text,fps=25,video=True)
        results[name]=load(folder/'result.json')
    return results

def test_all_cleared_assets_are_real_and_fsl_only():
    m=load(DEFAULT/'manifest.json');assert m['counts']['entries']==440 and m['export_errors']==[]
    rows=list(csv.DictReader((DEFAULT/'index.csv').open()))
    assert len(rows)==440 and all(r['signed_language']=='fsl' and r['spoken_language']=='fr' for r in rows)
    assert all(r['rights_status']=='cleared' for r in m['records'])
    assert m['upstream_revision']==UPSTREAM_REV

def test_measured_motion_points_preserved_and_absent_face_hips_masked():
    m=load(DEFAULT/'manifest.json');r=next(r for r in m['records'] if r['concept']=='merci')
    source=load(r['motion_path']);p=read_pose(DEFAULT/r['pose_path'])
    assert len(p.body.data)==len(source['frames'])
    for i,f in enumerate(source['frames']):
        for name,key in [('NOSE','head'),('LEFT_SHOULDER','left_shoulder'),('RIGHT_WRIST','right_wrist')]:
            if key in f['joints']:
                x,y,z=f['joints'][key];j=BODY.index(name)
                assert np.allclose(p.body.data[i,0,j],[320+240*x,560-240*y,-240*z],atol=1e-4)
    # Never turn face ratios or procedural hip roots into tracked landmark arrays.
    assert np.all(p.body.confidence[:,:,33:501]==0)
    for name in ['LEFT_HIP','RIGHT_HIP','LEFT_EYE','LEFT_ANKLE']:
        assert np.all(p.body.confidence[:,:,BODY.index(name)]==0)
    assert np.ma.getmaskarray(p.body.data)[:,:,33:501].all()

def test_raw_animics_exact_coordinates_visibility_and_missing_hands():
    m=load(PREVIEW/'manifest.json');r=next(r for r in m['records'] if r['source_id']=='DICTASIGN_PORTAL_380')
    raw=DEFAULT_MAIN/'.runtime/captures'/r['capture_id']/'animics-landmarks.json.gz'
    data=json.load(gzip.open(raw,'rt'));p=read_pose(PREVIEW/r['pose_path'])
    assert r['origin']['format']=='raw-animics-image-landmarks'
    for i,f in enumerate(data):
        if f.get('PLM'):
            point=f['PLM'][0];width=r['origin']['image_width'];height=r['origin']['image_height']
            assert np.allclose(p.body.data[i,0,0],[point['x']*width,point['y']*height,point['z']*width],atol=1e-4)
            assert np.isclose(p.body.confidence[i,0,0],point['visibility'])
        for key,offset,n in [('FLM',33,468),('LLM',501,21),('RLM',522,21)]:
            if not f.get(key):assert np.all(p.body.confidence[i,0,offset:offset+n]==0)
    assert p.header.total_points()==TOTAL==543
    assert not any('WORLD_HAND' in c.name for c in p.header.components)
    assert p.header.dimensions.width==width and p.header.dimensions.height==height

def test_dicta_explicit_preview_gate_preference_and_true_ambiguity():
    safe=load(DEFAULT/'manifest.json');preview=load(PREVIEW/'manifest.json')
    assert not any(r['local_preview_only'] for r in safe['records'])
    records,unknown,amb=resolve(preview,text='Bonjour merci')
    assert unknown==amb==[] and records[0]['source_id']=='DICTASIGN_PORTAL_380'
    assert records[0]['local_preview_only']
    fake=copy.deepcopy(preview);base=copy.deepcopy(records[0]);base['gloss']='DIFFERENT_CANONICAL';base['id']='ONLY_AMBIGUITY_SOFTWARE_FIXTURE';fake['records'].append(base)
    selected,unknown,amb=resolve(fake,text='Bonjour')
    assert selected==[] and amb and set(amb[0]['canonical_glosses'])=={'BONJOUR','DIFFERENT_CANONICAL'}

def test_unknown_stops_video_instead_of_fingerspelling_or_fixture(rendered):
    r=rendered['unknown']
    assert r['status']=='blocked_unknown_or_ambiguous' and r['unknowns']==['motinconnu']
    assert 'video_path' not in r and 'pose_path' not in r

def test_actual_upstream_csv_lookup_roundtrip():
    lookup=CSVPoseLookup(str(DEFAULT),backup=None)
    rows=lookup.lookup_sequence([GlossItem('bonjour','BONJOUR'),GlossItem('merci','MERCI')],'fr','fsl')
    assert len(rows)==2 and all(r.coverage==CoverageType.LEXICON for r in rows)
    with pytest.raises(FileNotFoundError):lookup.lookup('missing-entry','missing-entry','fr','fsl')

def test_upstream_join_never_creates_new_tracked_samples(rendered):
    result=rendered['cleared'];combined=read_pose(result['pose_path']);m=load(DEFAULT/'manifest.json');records={r['id']:r for r in m['records']};masks=[]
    for segment in result['segments']:
        p=read_pose(DEFAULT/records[segment['id']]['pose_path']);masks.append(p.body.confidence[segment['source_frame_indexes']])
    expected=np.concatenate(masks)
    assert np.array_equal(combined.body.confidence,expected)
    assert np.array_equal(np.ma.getmaskarray(combined.body.data).all(-1),combined.body.confidence==0)

def test_actual_video_and_local_preview_attribution(rendered):
    for name in ['cleared','preview']:
        r=rendered[name];s=r['ffprobe']['streams'][0]
        assert s['codec_name']=='h264' and int(s['nb_read_frames'])==r['frames']
        assert s['width']==s['height']==512
        assert Path(r['gif_path']).stat().st_size>1000 and Path(r['pose_path']).stat().st_size>1000
        assert r['translation_validated'] is False
    r=rendered['preview'];assert r['local_preview_only'] is True
    assert r['sources'][0]['rights_status']=='pending' and r['sources'][0]['source_id']=='DICTASIGN_PORTAL_380'


def test_priority_is_applied_before_canonical_ambiguity():
    from pipeline import select
    commons=dict(id='commons-age',gloss='ÂGE',priority=10)
    dicta=dict(id='dicta-age',gloss='AGE',priority=0)
    selected,ambiguity=select([commons,dicta])
    assert selected['id']=='dicta-age' and ambiguity==[]
    # Actual ambiguity among equally preferred source entries still blocks.
    conflict=dict(id='dicta-other',gloss='AGE_AUTRE',priority=0)
    selected,ambiguity=select([commons,dicta,conflict])
    assert selected is None and set(ambiguity)=={'AGE','AGE_AUTRE'}

def test_manifest_fingerprint_is_stable_but_sensitive_to_entry_changes():
    from pose_adapter import manifest_fingerprint
    manifest=load(DEFAULT/'manifest.json');expected=manifest_fingerprint(manifest)
    assert len(expected)==64
    again=copy.deepcopy(manifest);again['seconds']=123456
    assert manifest_fingerprint(again)==expected
    again['records'][0]['gloss']='CHANGED_ONLY_SOFTWARE_TEST'
    assert manifest_fingerprint(again)!=expected
