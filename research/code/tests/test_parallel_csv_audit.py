import pytest
from src.data.audit_parallel_csv import audit


def test_empty_gloss_is_not_supervision(tmp_path):
    p = tmp_path / 'source.csv'
    p.write_text('text,gloss,split\nbonjour,,train\nmerci, ,dev\n')
    r = audit(p, sign_language='sfb')
    assert r['counts'] == dict(rows=2, nonempty_text=2, nonempty_gloss=0, nonempty_pairs=0)
    assert not r['gold_supervision_verified']


def test_nonempty_gloss_does_not_imply_review(tmp_path):
    p = tmp_path / 'source.csv'
    p.write_text('text,gloss\nbonjour,BONJOUR\n')
    r = audit(p, sign_language='fsl')
    assert r['counts']['nonempty_pairs'] == 1
    assert not r['gold_supervision_verified']


def test_conflicting_record_language_rejected(tmp_path):
    p = tmp_path / 'source.csv'
    p.write_text('text,gloss,sign_language\nbonjour,BONJOUR,sfb\n')
    with pytest.raises(ValueError, match='language'):
        audit(p, sign_language='fsl')
