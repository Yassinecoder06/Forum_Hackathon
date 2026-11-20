"""Exercise cached ingestion against the real registration service without network."""
import gzip
import json
from pathlib import Path
import uuid
import pytest
from scripts import ingest_commons_signs as commons, ingest_dictasign as dicta
from backend.app.services import capture_library as store


def no_network(*args,**kwargs):
    raise AssertionError('Ingestion regression tests must not download')


def cached_capture(tmp_path, job_id):
    folder=tmp_path/'captures'/job_id;folder.mkdir(parents=True)
    (folder/'source.mp4').write_bytes(b'cached test source, never decoded')
    motion=json.loads(Path('data/approved_motion_assets/LSF_BONJOUR_COMMONS.json').read_text())
    (folder/'motion.json').write_text(json.dumps(motion))
    with gzip.open(folder/'animics-landmarks.json.gz','wt') as stream:json.dump([],stream)


def test_cached_commons_uses_explicit_fsl_without_approval(tmp_path,monkeypatch):
    monkeypatch.setenv('SIGNACULTURE_STORAGE',str(tmp_path));monkeypatch.setattr(commons,'fetch',no_network)
    entry=dict(language_variant='LSF_FR',eligible_for_motion=True,source_id='WMC_fixture',source_url='https://example.org/test',concept_fr='fixture',credit='fixture',license='CC BY 3.0',license_url='https://creativecommons.org/licenses/by/3.0',title='fixture')
    job=str(uuid.uuid5(uuid.NAMESPACE_URL,entry['source_url']));cached_capture(tmp_path,job)
    outcome=commons.ingest(entry);assert outcome['status']=='imported'
    row=store.library()[0];assert row['sign_language']=='fsl'
    assert row['gloss']['review_status']=='pending'
    assert row['motion']['animation_review']=='pending'


def test_cached_dicta_uses_explicit_fsl_and_keeps_rights_pending(tmp_path,monkeypatch):
    monkeypatch.setenv('SIGNACULTURE_STORAGE',str(tmp_path));monkeypatch.setattr(dicta,'fetch',no_network)
    entry=dict(language='fsl',video_url='https://example.org/lsf/test.mp4',concept_fr='fixture',concept_id='fixture',canonical_gloss='FIXTURE',labels=[],page_url='https://example.org/test',licence='rights pending')
    job=str(uuid.uuid5(uuid.NAMESPACE_URL,entry['video_url']));cached_capture(tmp_path,job)
    outcome=dicta.extract_entry(entry);assert outcome['status']=='imported'
    row=store.library()[0];assert row['sign_language']=='fsl'
    assert row['motion']['rights_status']=='pending'
    assert row['gloss']['review_status']=='pending'


@pytest.mark.parametrize('language',[None,'sfb','ssr'])
def test_wrong_or_unknown_language_rejected_before_io(language,monkeypatch):
    monkeypatch.setattr(commons,'fetch',no_network);monkeypatch.setattr(dicta,'fetch',no_network)
    with pytest.raises(ValueError,match='France-LSF'):commons.ingest({'language_variant':language})
    with pytest.raises(ValueError,match='France-LSF'):dicta.extract_entry({'language':language})


def test_ineligible_commons_source_rejected_before_io():
    with pytest.raises(ValueError,match='not eligible'):
        commons.ingest(dict(language_variant='LSF_FR',eligible_for_motion=False))
