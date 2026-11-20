import hashlib
import json
import uuid
import pytest
from scripts.migrate_capture_languages import identify


def fixture(tmp_path,sidecar):
    url='https://example.org/commons/fixture';job=str(uuid.uuid5(uuid.NAMESPACE_URL,url))
    folder=tmp_path/'captures'/job;folder.mkdir(parents=True);raw=b'original fixture'
    (folder/'source.mp4').write_bytes(raw)
    entry=dict(language_variant='LSF_FR',eligible_for_motion=True,source_url=url,source_sha1=hashlib.sha1(raw).hexdigest())
    if sidecar:(folder/'source_provenance.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),canonical_licence_source=url)))
    return dict(source_id='fixture',capture_id=job),{'fixture':(entry,'inventory-digest','commons')},folder


@pytest.mark.parametrize('sidecar',[False,True])
def test_source_bytes_identify_language_without_linguistic_approval(tmp_path,sidecar):
    row,inventory,folder=fixture(tmp_path,sidecar)
    result=identify(row,inventory,tmp_path)
    assert result['source_id']=='fixture'
    assert result['linguistic_review'].startswith('pending')
    assert 'sign_language' not in row


@pytest.mark.parametrize('sidecar',[False,True])
def test_changed_source_cannot_receive_language_evidence(tmp_path,sidecar):
    row,inventory,folder=fixture(tmp_path,sidecar);(folder/'source.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum|hash'):identify(row,inventory,tmp_path)


def test_capture_identity_mismatch_rejected(tmp_path):
    row,inventory,folder=fixture(tmp_path,True);row['capture_id']='other'
    with pytest.raises(ValueError,match='capture ID'):identify(row,inventory,tmp_path)
