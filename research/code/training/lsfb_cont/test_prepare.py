import copy
from pathlib import Path

import pytest

from training.lsfb_cont.prepare import (TASK, digest, lexical_operation, make_rows, pair_events,
                                       parse_plan, serialize_plan, token_preflight)
from training.lsfb_cont.release import apply_decisions
from training.lsfb_cont.review import recording_from_index


def event(start=100, end=200, gloss='oui', lemma='oui', **extra):
    return {'start_ms': start, 'end_ms': end, 'gloss': gloss, 'lemma': lemma,
            'sign_type': 'lexical', **extra}


def test_projection_merges_simultaneous_variant_but_preserves_sequential_repetition():
    from training.lsfb_cont.prepare import lexical_projection_events
    source = [event(gloss='oui(7)'), event(), event(220,290), event(gloss='non',lemma='non')]
    before = copy.deepcopy(source)
    retained, removed = lexical_projection_events(source)
    assert len(retained) == 3 and len(removed) == 1
    assert retained[1]['start_ms'] == 220
    assert retained[-1]['lemma'] == 'non'
    assert source == before


def record(signer, role='train', turns=None, events=None, group='CLSFBI9901'):
    return {'signer.txt': signer, 'official_role': role,
            'source_shard': 1 if role == 'validation' else 2,
            'conversation_id': group, 'annotations.translation.json': turns or [],
            'annotations.both_hands.json': events or []}


def test_fingerspelling_preserves_operation_and_original_hint():
    e = event(gloss='fs:fk(ok)', lemma='fk', specifier='fs')
    assert lexical_operation(e) == {'op': 'fingerspell', 'label': 'fk', 'annotation': 'fs:fk(ok)'}
    assert lexical_operation({**e, 'sign_type': 'pointing'}) is None


def test_plan_grammar_handles_punctuation_and_quoted_separators():
    operations = [{'op': 'sign', 'lemma': 'a ; b'},
                  {'op': 'fingerspell', 'label': 'ok', 'annotation': 'fs:fk("ok")'}]
    assert parse_plan(serialize_plan(operations)) == operations
    with pytest.raises(ValueError):
        parse_plan('SIGN "oui" ;')


def test_boundary_review_retains_dropped_events_and_real_repetition():
    e = event()
    repeated = event(220, 290)
    crossing = event(290, 350)
    source = [e, copy.deepcopy(e), repeated, crossing]
    before = copy.deepcopy(source)
    selected, dropped = pair_events(source, 100, 300)
    assert selected == [e, repeated] and dropped == [crossing]
    assert source == before


def test_context_has_only_completed_french_turns_and_no_glosses():
    records = {'CLSFBI9901A_S001_B': record('S001', turns=[
        {'start_ms': 50, 'end_ms': 90, 'text': 'Avant.'},
        {'start_ms': 100, 'end_ms': 200, 'text': 'Oui.'},
        {'start_ms': 201, 'end_ms': 300, 'text': 'Futur.'}], events=[event()]),
        'CLSFBI9901A_S002_B': record('S002', turns=[
            {'start_ms': 70, 'end_ms': 150, 'text': 'Tour encore en cours.'}])}
    row = next(r for r in make_rows(records) if r['french_text'] == 'Oui.')
    assert [c['text'] for c in row['context']] == ['Avant.']
    assert 'Futur.' not in row['input_text'] and 'Tour encore' not in row['input_text']
    assert 'S001' not in row['input_text'] and 'target_operations' not in row['input_text']


def test_validation_conversation_purges_other_training_participant():
    tr = [{'start_ms': 100, 'end_ms': 200, 'text': 'Oui.'}]
    records = {'CLSFBI9901A_S001_B': record('S001', turns=tr, events=[event()]),
               'CLSFBI9901A_S002_B': record('S002', 'validation', turns=tr, events=[event()])}
    rows = make_rows(records)
    assert [r['split'] for r in rows] == ['quarantine', 'val']
    assert 'conversation_present_in_official_validation' in rows[0]['technical_exclusions']


def reviewable_row():
    row = make_rows({'CLSFBI9901A_S001_B': record('S001', turns=[
        {'start_ms': 100, 'end_ms': 200, 'text': 'Oui.'}], events=[event()])})[0]
    decision = {'id': row['id'], 'content_sha256': row['content_sha256'],
                'task': TASK, 'decision': 'accept', 'reviewer': 'test-reviewer',
                'reviewed_at': '2026-11-15T00:00:00Z',
                'checks': dict.fromkeys(['alignment', 'meaning', 'context', 'representation'], True)}
    return row, decision


def test_no_pending_rows_automatically_promoted():
    row, _ = reviewable_row()
    accepted, _ = apply_decisions([row], [])
    assert accepted == {'train': [], 'val': []}


def test_stale_or_mutated_source_rejects_review():
    row, decision = reviewable_row()
    row['french_text'] = 'Changed.'
    with pytest.raises(ValueError, match='content changed'):
        apply_decisions([row], [decision])


def test_acceptance_requires_review_checks_and_excludes_quarantine():
    row, decision = reviewable_row()
    decision['checks']['meaning'] = False
    with pytest.raises(ValueError, match='four'):
        apply_decisions([row], [decision])
    decision['checks']['meaning'] = True
    row['technical_exclusions'] = ['no_lexical_plan']
    row.pop('content_sha256')
    row['content_sha256'] = digest(row)
    decision['content_sha256'] = row['content_sha256']
    with pytest.raises(ValueError, match='technically excluded'):
        apply_decisions([row], [decision])


def test_accepted_correction_preserves_source_and_changes_only_release_target():
    row, decision = reviewable_row()
    before = copy.deepcopy(row)
    decision['corrected_target'] = [{'op': 'sign', 'lemma': 'accord'}]
    accepted, _ = apply_decisions([row], [decision])
    assert accepted['train'][0]['target_text'] == 'SIGN "accord"'
    assert row == before


def test_real_t5_tokenizer_preserves_plan_and_quarantines_unknown_label():
    pytest.importorskip('tokenizers')
    tokenizer = Path(__file__).parents[4]/'.runtime/lsfb-cont-tokenizer/tokenizer.json'
    if not tokenizer.exists():
        pytest.skip('Local pinned tokenizer not downloaded')
    row, _ = reviewable_row()
    row['target_operations'] = [{'op': 'sign', 'lemma': 'bonjour'}]
    row['target_text'] = serialize_plan(row['target_operations'])
    token_preflight([row], tokenizer)
    assert row['token_counts']['target_unknowns'] == 0
    assert not row['technical_exclusions']
    row['target_operations'] = [{'op': 'sign', 'lemma': '{unrepresentable}'}]
    row['target_text'] = serialize_plan(row['target_operations'])
    token_preflight([row], tokenizer)
    assert 'target_not_representable_by_t5_tokenizer' in row['technical_exclusions']


def test_indexed_video_read_verifies_actual_tar_header(tmp_path):
    import io
    import json
    import tarfile
    rid = 'CLSFBI9901A_S001_B'
    content = b'example media payload'
    with tarfile.open(tmp_path/'videos.tar', 'w') as archive:
        info = tarfile.TarInfo(rid+'.mp4')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    index = {rid+'.mp4': [512, len(content)]}
    (tmp_path/'videos.tar.index.json').write_text(json.dumps(index))
    recording_from_index(tmp_path, rid, tmp_path/'result.mp4')
    assert (tmp_path/'result.mp4').read_bytes() == content
    index[rid+'.mp4'][1] -= 1
    (tmp_path/'videos.tar.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='TAR member'):
        recording_from_index(tmp_path, rid, tmp_path/'invalid.mp4')
