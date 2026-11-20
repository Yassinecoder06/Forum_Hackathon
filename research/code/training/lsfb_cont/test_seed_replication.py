import json
from pathlib import Path
import pytest
from training.lsfb_cont.summarize_seed_replication import summarize


def fixture_files(tmp_path):
    paths = []
    for seed, wer in [(84, .4), (85, .6)]:
        p = tmp_path / str(seed)
        p.mkdir()
        (p / 'MANIFEST.json').write_text(json.dumps(dict(seed=seed, data_hash='same', input_mode='current')))
        (p / 'RESULT.json').write_text(json.dumps(dict(status='complete', best=dict(epoch=11), seconds=2,
            validation=dict(exact=.5, operation_wer=wer, by_length=dict(multiple_signs=dict(examples=1, exact=0, wer=1)), cap_hits=0))))
        (p / 'validation_predictions.json').write_text(json.dumps([dict(id='dev1', reference=['a','b'])]))
        paths.append(p)
    retrieval = tmp_path / 'retrieval.json'
    retrieval.write_text(json.dumps(dict(current_french=dict(examples=1, operation_wer=.5, predictions=[dict(id='dev1')]))))
    return paths, retrieval


def test_all_seeds_kept_without_winner_selection(tmp_path):
    paths, retrieval = fixture_files(tmp_path)
    result = summarize(paths, retrieval)
    assert len(result['runs']) == 2
    assert result['aggregate']['operation_wer']['mean'] == .5
    assert result['seeds_lower_wer_than_retrieval'] == 1
    assert not result['independent_test']


def test_mismatched_protocol_rejected(tmp_path):
    paths, retrieval = fixture_files(tmp_path)
    p = paths[-1] / 'MANIFEST.json'
    m = json.loads(p.read_text());m['data_hash'] = 'changed';p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='Protocol mismatch'):
        summarize(paths, retrieval)


def test_duplicate_seed_rejected(tmp_path):
    paths, retrieval = fixture_files(tmp_path)
    with pytest.raises(ValueError, match='Duplicate seed'):
        summarize([paths[0], paths[0]], retrieval)
