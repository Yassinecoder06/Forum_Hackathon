import copy
import pytest
from training.lsfb_cont.alignment_audit import row_evidence
from training.lsfb_cont.prepare import digest


def example():
    source=dict(id='fixture',source_revision='64b0c112f59d3710367fb736ef1808f37ff15e28',
                interval_ms=[100,200],recording_id='fixture',
                aligned_events=[dict(start_ms=100,end_ms=200,start_frame=5,end_frame=10,lemma='oui',sign_type='lexical'),
                                dict(start_ms=120,end_ms=180,sign_type='pointing')],boundary_excluded_events=[])
    source['content_sha256']=digest(source)
    row=dict(id='fixture',language='LSFB',split='train',target=['oui'],interval_ms=[100,200],recording_id='fixture')
    return row,source


def test_lexical_match_does_not_hide_nonlexical_omission():
    row,source=example();before=copy.deepcopy(source);r=row_evidence(row,source)
    assert r['target_matches_source']
    assert r['nonlexical_event_count']==1
    assert r['max_frame_timestamp_deviation_ms']==0
    assert not r['linguistically_validated']
    assert source==before


def test_changed_target_reported_without_repair():
    row,source=example();row['target']=['non']
    assert not row_evidence(row,source)['target_matches_source']
    assert row['target']==['non']


def test_stale_source_hash_rejected():
    row,source=example();source['interval_ms'][1]=201
    with pytest.raises(ValueError,match='hash'):
        row_evidence(row,source)
