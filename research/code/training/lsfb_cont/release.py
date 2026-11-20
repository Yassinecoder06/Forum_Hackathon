"""Import provenance-bound review decisions and export reviewed training rows."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .prepare import TASK, digest, parse_plan, serialize_plan, write_json, write_rows
else:
    from prepare import TASK, digest, parse_plan, serialize_plan, write_json, write_rows


def validate_target(operations):
    if not isinstance(operations, list) or not operations:
        raise ValueError('Accepted lexical plan must contain operations')
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError('Expected a lexical operation object')
        fields = {'op', 'lemma'} if op.get('op') == 'sign' else {'op', 'label', 'annotation'} if op.get('op') == 'fingerspell' else set()
        if not fields or set(op) != fields or any(not isinstance(v, str) or not v.strip() for v in op.values()):
            raise ValueError('Invalid lexical operation schema')
    if parse_plan(serialize_plan(operations)) != operations:
        raise ValueError('Target representation does not roundtrip')


def apply_decisions(rows, decisions):
    lookup = {r['id']: r for r in rows}
    seen = set()
    accepted = {'train': [], 'val': []}
    audit = []
    for decision in decisions:
        rid = decision['id']
        if rid in seen:
            raise ValueError('Multiple decisions for one row: '+rid)
        seen.add(rid)
        if rid not in lookup:
            raise ValueError('Unknown reviewed row: '+rid)
        row = lookup[rid]
        if decision.get('task') != TASK or decision.get('content_sha256') != row['content_sha256']:
            raise ValueError('Stale review or task mismatch: '+rid)
        core = {k: v for k, v in row.items() if k != 'content_sha256'}
        if digest(core) != row['content_sha256']:
            raise ValueError('Candidate content changed: '+rid)
        if not isinstance(decision.get('reviewer'), str) or not decision['reviewer'].strip() or not decision.get('reviewed_at'):
            raise ValueError('Reviewer and review time required')
        if decision.get('decision') not in ['accept', 'reject', 'needs_context', 'needs_alignment']:
            raise ValueError('Invalid decision')
        audit.append(decision)
        if decision['decision'] != 'accept':
            continue
        if row['split'] not in accepted or row['technical_exclusions']:
            raise ValueError('Cannot approve a quarantined or technically excluded row; repair preparation first: '+rid)
        if any(decision.get('checks', {}).get(k) is not True for k in ['alignment', 'meaning', 'context', 'representation']):
            raise ValueError('Acceptance requires all four linguistic/source checks')
        target = decision.get('corrected_target', row['target_operations'])
        validate_target(target)
        accepted[row['split']].append({
            'id': rid, 'content_sha256': row['content_sha256'],
            'conversation_id': row['conversation_id'], 'signer_id': row['signer_id'],
            'input_text': row['input_text'], 'target_text': serialize_plan(target),
            'review_status': 'accepted', 'reviewer': decision['reviewer'],
            'reviewed_at': decision['reviewed_at'], 'review_notes': decision.get('notes', '')})
    return accepted, audit


def release(dataset, decisions_path, output):
    from tokenizers import Tokenizer
    if output.exists():
        raise ValueError('Use a fresh release directory')
    manifest = json.loads((dataset/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((dataset/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen input changed: '+name)
    rows = [json.loads(l) for l in (dataset/'candidates.jsonl').read_text().splitlines()]
    decisions = [json.loads(l) for l in decisions_path.read_text().splitlines() if l.strip()]
    accepted, audit = apply_decisions(rows, decisions)
    tokenizer_path = Path(manifest['tokenizer']['path'])
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != manifest['tokenizer']['sha256']:
        raise ValueError('Tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for subset in accepted.values():
        for row in subset:
            encoded = tokenizer.encode(row['target_text'])
            if len(encoded.ids) > 512 or tokenizer.token_to_id('<unk>') in encoded.ids or parse_plan(tokenizer.decode(encoded.ids)) != parse_plan(row['target_text']):
                raise ValueError('Corrected target failed tokenizer preflight: '+row['id'])
    if not accepted['train'] or not accepted['val']:
        raise ValueError('Reviewed train AND validation rows required; no training release created')
    assert not {r['conversation_id'] for r in accepted['train']} & {r['conversation_id'] for r in accepted['val']}
    for name, subset in accepted.items():
        write_rows(output/(name+'.jsonl'), subset)
    write_rows(output/'review_audit.jsonl', audit)
    write_json(output/'manifest.json', {'task': TASK, 'mode': 'reviewed', 'counts': {k: len(v) for k, v in accepted.items()},
                                      'source_manifest_sha256': hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
                                      'decisions_sha256': hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
                                      'policy': manifest['policy'], 'tokenizer': manifest['tokenizer'],
                                      'test_used': False, 'requires_learning_gate_before_full_training': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--decisions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    release(a.dataset, a.decisions, a.output)
