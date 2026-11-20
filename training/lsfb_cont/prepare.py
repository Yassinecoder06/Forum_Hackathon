"""Prepare local LSFB-CONT for context-conditioned lexical planning.

No source files are changed. Automatic checks never constitute linguistic approval.
The official test shard is intentionally not opened by this command.
"""
import argparse
import collections
import copy
import csv
import gzip
import hashlib
import json
import re
import tarfile
import unicodedata
from pathlib import Path

REPO = 'ppoitier/lsfb-cont'
REVISION = '64b0c112f59d3710367fb736ef1808f37ff15e28'
TASK = 'lsfb-cont-context-lexical-plan-v1'
SHARDS = [1, 2, 3, 4, 5, 6, 7]
TIERS = ['left_hand', 'right_hand', 'both_hands', 'translation']
POLICY = {
    'task': TASK, 'source_revision': REVISION,
    'input': 'current French translation plus at most four completed preceding turns within 60s, across both participants; no gloss context or future text',
    'target': 'readable SIGN / FINGERSPELL operations with quoted labels; lexical author lemma and raw fingerspelling annotation; genuine repetitions retained',
    'annotation_projection': 'nonlexical events and hand/timing metadata retained in sidecar; not full signing or an avatar motion plan',
    'alignment': '80% of event duration inside translation; intersecting excluded events retained for review',
    'split': 'official validation targets retained; quarantine every training conversation present anywhere in validation; retain official train/val signer separation',
    'vocabulary': 'statistics from technically eligible training targets only; no new atomic tokenizer tokens',
    'review': 'every row pending until a provenance-bound human linguistic review is imported',
    'technical_limits': {'max_french_words': 80, 'max_lexical_operations': 32, 'max_duration_seconds': 30},
    'token_limits': {'source': 512, 'target': 512, 'context_overflow': 'remove oldest completed context turns; never truncate current text or target'},
    'official_test': 'shard 0 excluded; previously used test is not a fresh development holdout',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def conversation_id(rid):
    # Keep A/B camera suffix conservative: even those variants stay in one group.
    match = re.fullmatch(r'(CLSFBI\d+)[A-Za-z]?_S\d+_[A-Za-z]+', rid)
    if not match:
        raise ValueError('Unrecognized recording ID; specify grouping rather than guessing: '+rid)
    return match[1]


def validate_interval(value):
    return (isinstance(value.get('start_ms'), (int, float)) and
            isinstance(value.get('end_ms'), (int, float)) and
            value['end_ms'] > value['start_ms'])


def load_local(root, cache):
    """Verify local shard hashes against pinned HF metadata; retain all hand tiers."""
    tree = root/'.cache/huggingface/trees'/f'{REVISION}.json'
    if not tree.exists():
        raise ValueError('Pinned HF tree metadata missing: '+str(tree))
    expected = json.loads(tree.read_text())['files']
    records = {}
    provenance = []
    cache.mkdir(parents=True, exist_ok=True)
    for n in SHARDS:
        relative = f'shards/annotated/shard_{n:06d}.tar'
        path = root/relative
        spec = expected[relative]
        if path.stat().st_size != spec['size']:
            raise ValueError('Shard size mismatch: '+str(path))
        checksum = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if checksum != spec['lfs_sha256']:
            raise ValueError('Shard checksum mismatch: '+str(path))
        saved = cache/f'source_{n:06d}.json.gz'
        value = None
        if saved.exists():
            with gzip.open(saved, 'rt', encoding='utf-8') as f:
                cached = json.load(f)
            if cached.get('tar_sha256') == checksum and cached.get('tiers') == TIERS:
                value = cached
        if value is None:
            entries = {}
            suffixes = [f'annotations.{tier}.json' for tier in TIERS]+['signer.txt', 'language.txt']
            with tarfile.open(path) as tar:
                for member in tar:
                    suffix = next((s for s in suffixes if member.name.endswith('.'+s)), None)
                    if not suffix or not member.isfile():
                        continue
                    if member.size > 20_000_000:
                        raise ValueError('Unexpected annotation member size')
                    rid = member.name[:-len(suffix)-1]
                    content = tar.extractfile(member).read().decode('utf-8')
                    entries.setdefault(rid, {})[suffix] = json.loads(content) if suffix.endswith('.json') else content.strip()
            value = {'source': REPO, 'revision': REVISION, 'shard': n,
                     'tar_sha256': checksum, 'tiers': TIERS, 'records': entries}
            with gzip.open(saved, 'wt', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False)
        role = 'validation' if n == 1 else 'train'
        for rid, content in value['records'].items():
            if rid in records:
                raise ValueError('Recording duplicated across source shards: '+rid)
            if content.get('language.txt') != 'lsfb':
                raise ValueError('Unexpected language: '+rid)
            if not content.get('signer.txt') or content['signer.txt'] not in rid:
                raise ValueError('Signer/recording mismatch: '+rid)
            records[rid] = {**content, 'source_shard': n, 'official_role': role,
                            'conversation_id': conversation_id(rid)}
        provenance.append({'shard': n, 'official_role': role, 'bytes': path.stat().st_size,
                           'sha256': checksum, 'recordings': len(value['records'])})
        print(f'Verified source shard {n}: {len(value["records"])} recordings', flush=True)
    return records, provenance


def lexical_operation(event):
    if event.get('sign_type') != 'lexical':
        return None
    lemma = event.get('lemma')
    if not isinstance(lemma, str) or not lemma.strip():
        return None
    if event.get('specifier') == 'fs':
        # Never guess letters from a parsed lemma or discard the raw spelling hint.
        return {'op': 'fingerspell', 'label': lemma.strip(), 'annotation': event['gloss']}
    return {'op': 'sign', 'lemma': lemma.strip()}


def serialize_plan(operations):
    chunks = []
    quote = lambda s: json.dumps(s, ensure_ascii=False)
    for op in operations:
        if op['op'] == 'sign':
            chunks.append('SIGN '+quote(op['lemma']))
        elif op['op'] == 'fingerspell':
            chunks.append('FINGERSPELL '+quote(op['label'])+' '+quote(op['annotation']))
        else:
            raise ValueError('Unsupported lexical operation')
    return ' ; '.join(chunks)


def lexical_projection_events(events):
    """Collapse only identical-time identical operations; retain raw events elsewhere."""
    retained, simultaneous_duplicates, seen = [], [], set()
    for event in events:
        operation = lexical_operation(event)
        if operation is None:
            continue
        key = (event['start_ms'], event['end_ms'],
               json.dumps(operation, sort_keys=True, ensure_ascii=False))
        if key in seen:
            simultaneous_duplicates.append(copy.deepcopy(event))
        else:
            seen.add(key)
            retained.append(copy.deepcopy(event))
    return retained, simultaneous_duplicates


def parse_plan(text):
    decoder = json.JSONDecoder()
    remaining = text.strip()
    result = []
    while remaining:
        if remaining.startswith('SIGN '):
            op = {'op': 'sign'}
            fields = ['lemma']
            remaining = remaining[5:].lstrip()
        elif remaining.startswith('FINGERSPELL '):
            op = {'op': 'fingerspell'}
            fields = ['label', 'annotation']
            remaining = remaining[12:].lstrip()
        else:
            raise ValueError('Invalid lexical plan operation')
        for field in fields:
            value, end = decoder.raw_decode(remaining)
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Expected nonempty quoted lexical label')
            op[field] = value
            remaining = remaining[end:].lstrip()
        result.append(op)
        if remaining:
            if not remaining.startswith(';'):
                raise ValueError('Expected operation separator')
            remaining = remaining[1:].lstrip()
            if not remaining:
                raise ValueError('Trailing operation separator')
    return result


def pair_events(events, start, end):
    selected = []
    dropped = []
    seen = set()
    for event in events:
        if not validate_interval(event) or not isinstance(event.get('gloss'), str) or not event['gloss'].strip():
            continue
        overlap = max(0, min(end, event['end_ms'])-max(start, event['start_ms']))
        if not overlap:
            continue
        key = (event['start_ms'], event['end_ms'], event['gloss'])
        if key in seen:
            continue
        seen.add(key)
        if overlap/(event['end_ms']-event['start_ms']) >= .8:
            selected.append(copy.deepcopy(event))
        else:
            dropped.append(copy.deepcopy(event))
    order = lambda e: (e['start_ms'], e['end_ms'], e['gloss'])
    return sorted(selected, key=order), sorted(dropped, key=order)


def render_input(text, signer, context):
    lines = [('Moi' if t['speaker'] == signer else 'Autre')+': '+t['text'] for t in context]
    return 'Planifier les signes lexicaux en LSFB.\nContexte:\n'+'\n'.join(lines)+'\nTexte actuel: '+text


def make_rows(records):
    heldout_groups = {r['conversation_id'] for r in records.values() if r['official_role'] == 'validation'}
    timelines = collections.defaultdict(list)
    for rid, record in sorted(records.items()):
        for index, tr in enumerate(record.get('annotations.translation.json', [])):
            if validate_interval(tr) and isinstance(tr.get('text'), str) and tr['text'].strip():
                # Keep exact session clocks separate even when split grouping is broader.
                timelines[rid.split('_')[0]].append({
                    'recording_id': rid, 'translation_index': index,
                    'speaker': record['signer.txt'], 'text': ' '.join(tr['text'].split()),
                    'start_ms': tr['start_ms'], 'end_ms': tr['end_ms']})
    rows = []
    for clock_id, timeline in sorted(timelines.items()):
        timeline.sort(key=lambda t: (t['start_ms'], t['end_ms'], t['recording_id'], t['translation_index']))
        for turn in timeline:
            rid = turn['recording_id']
            record = records[rid]
            group = record['conversation_id']
            a, b = turn['start_ms'], turn['end_ms']
            events, dropped = pair_events(record.get('annotations.both_hands.json', []), a, b)
            operations = [o for e in events if (o := lexical_operation(e)) is not None]
            past = [t for t in timeline if t['end_ms'] <= a and t['end_ms'] >= a-60_000]
            context = past[-4:]
            # Role labels are relative to the current signer, avoiding global identity features.
            input_text = render_input(turn['text'], turn['speaker'], context)
            review_flags = []
            reasons = []
            if not events:
                reasons.append('no_aligned_sign_events')
            if not operations:
                reasons.append('no_lexical_plan')
            if len(turn['text'].split()) > POLICY['technical_limits']['max_french_words']:
                reasons.append('long_french_segment')
            if len(operations) > POLICY['technical_limits']['max_lexical_operations']:
                reasons.append('long_lexical_plan')
            if b-a > POLICY['technical_limits']['max_duration_seconds']*1000:
                reasons.append('long_signing_segment')
            if any(e.get('sign_type') == 'lexical' and lexical_operation(e) is None for e in events):
                reasons.append('lexical_event_missing_author_lemma')
            if any(o['op'] == 'fingerspell' for o in operations):
                review_flags.append('fingerspelling_content_and_operation')
            if any(e.get('sign_type') != 'lexical' for e in events):
                review_flags.append('nonlexical_events_preserved_outside_projection')
            if dropped:
                review_flags.append('boundary_excluded_events')
            if any(e['end_ms'] > other['start_ms'] for e, other in zip(events, events[1:])):
                review_flags.append('simultaneous_hand_events')
            if any(x == y for x, y in zip(operations, operations[1:])):
                review_flags.append('genuine_repetition_requires_review')
            if not context:
                review_flags.append('no_preceding_context')
            if re.search(r'…|\.\.\.', turn['text']):
                review_flags.append('ellipsis_or_fragment')
            split = 'val' if record['official_role'] == 'validation' else 'train'
            if split == 'train' and group in heldout_groups:
                reasons.append('conversation_present_in_official_validation')
                split = 'quarantine'
            tiers = {}
            for tier in ['left_hand', 'right_hand']:
                tiers[tier] = [copy.deepcopy(e) for e in record.get(f'annotations.{tier}.json', [])
                               if validate_interval(e) and e['end_ms'] > a and e['start_ms'] < b]
            core = {'task': TASK, 'source_revision': REVISION,
                    'id': f'{rid}:{a}:{b}:{turn["translation_index"]}',
                    'recording_id': rid, 'conversation_id': group,
                    'signer_id': turn['speaker'], 'official_role': record['official_role'],
                    'source_shard': record['source_shard'], 'split': split,
                    'interval_ms': [a, b], 'french_text': turn['text'],
                    'context': context, 'input_text': input_text,
                    'target_operations': operations,
                    'target_text': serialize_plan(operations),
                    'aligned_events': events, 'boundary_excluded_events': dropped,
                    'hand_tiers': tiers,
                    'review_flags': review_flags, 'technical_exclusions': reasons,
                    'review_status': 'pending', 'license': 'CC-BY-NC-SA-4.0',
                    'video': {'archive_member': rid+'.mp4', 'start_ms': a, 'end_ms': b}}
            core['content_sha256'] = digest(core)
            rows.append(core)
    # Deduplicate within train; remove input collisions with validation, preserving every source row.
    val_inputs = {normalize(r['input_text']) for r in rows if r['split'] == 'val'}
    seen = set()
    for row in rows:
        if row['split'] != 'train':
            continue
        key = (normalize(row['input_text']), row['target_text'])
        if key in seen:
            row['technical_exclusions'].append('duplicate_training_input_target')
        seen.add(key)
        if normalize(row['input_text']) in val_inputs:
            row['technical_exclusions'].append('input_also_in_validation')
        # Hash reflects the final row including duplicate/exclusion decisions.
        row.pop('content_sha256')
        row['content_sha256'] = digest(row)
    return rows


def token_preflight(rows, tokenizer_path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unk = tokenizer.token_to_id('<unk>')
    for row in rows:
        context = list(row['context'])
        def render():
            return render_input(row['french_text'], row['signer_id'], context)
        text = render()
        while context and len(tokenizer.encode(text).ids) > POLICY['token_limits']['source']:
            context.pop(0)
            text = render()
        row['input_text'] = text
        row['model_context'] = context
        row['omitted_context_turns'] = len(row['context'])-len(context)
        if row['omitted_context_turns']:
            row['review_flags'].append('oldest_context_removed_for_token_budget')
        encoded = tokenizer.encode(row['target_text'])
        source_tokens = len(tokenizer.encode(text).ids)
        row['token_counts'] = {'source': source_tokens, 'target': len(encoded.ids), 'target_unknowns': encoded.ids.count(unk)}
        if source_tokens > POLICY['token_limits']['source']:
            row['technical_exclusions'].append('source_token_limit')
        if len(encoded.ids) > POLICY['token_limits']['target']:
            row['technical_exclusions'].append('target_token_limit')
        if unk in encoded.ids:
            row['technical_exclusions'].append('target_not_representable_by_t5_tokenizer')
        try:
            recovered = parse_plan(tokenizer.decode(encoded.ids))
        except (ValueError, json.JSONDecodeError):
            recovered = None
        if recovered != row['target_operations']:
            row['technical_exclusions'].append('target_tokenizer_roundtrip_mismatch')
        row.pop('content_sha256')
        row['content_sha256'] = digest(row)
    # Context pruning must not introduce identical model inputs across the split.
    val_inputs = {normalize(r['input_text']) for r in rows if r['split'] == 'val'}
    seen = set()
    for row in rows:
        if row['split'] == 'train':
            key = (normalize(row['input_text']), row['target_text'])
            if key in seen and 'duplicate_training_input_target' not in row['technical_exclusions']:
                row['technical_exclusions'].append('duplicate_training_input_target')
                row.pop('content_sha256')
                row['content_sha256'] = digest(row)
            seen.add(key)
        if row['split'] == 'train' and normalize(row['input_text']) in val_inputs:
            if 'input_also_in_validation' not in row['technical_exclusions']:
                row['technical_exclusions'].append('input_also_in_validation')
                row.pop('content_sha256')
                row['content_sha256'] = digest(row)


def build(root, output, tokenizer_path):
    if (output/'manifest.json').exists():
        raise ValueError('Output already frozen; use a new output directory')
    records, source = load_local(root, output/'sources')
    rows = make_rows(records)
    token_preflight(rows, tokenizer_path)
    train = [r for r in rows if r['split'] == 'train' and not r['technical_exclusions']]
    val = [r for r in rows if r['split'] == 'val' and not r['technical_exclusions']]
    assert not {r['conversation_id'] for r in train} & {r['conversation_id'] for r in val}
    assert not {r['signer_id'] for r in train} & {r['signer_id'] for r in val}
    write_rows(output/'candidates.jsonl', rows)
    for split, subset in [('train', train), ('val', val)]:
        write_rows(output/'exploratory'/f'{split}.jsonl', [
            {k: r[k] for k in ['id', 'content_sha256', 'conversation_id', 'input_text', 'target_text', 'review_status']}
            for r in subset])
    frequency = collections.Counter(o['lemma'] for r in train for o in r['target_operations'] if o['op'] == 'sign')
    write_json(output/'train_vocabulary.json', {'basis': 'technically eligible train targets only; no added model tokens',
                                              'sign_lemma_counts': dict(frequency.most_common()),
                                              'fingerspelling_operations': sum(o['op'] == 'fingerspell' for r in train for o in r['target_operations'])})
    # A diagnostic sample mixes typical and difficult cases from every shard. It is not a benchmark.
    sample = []
    selected_ids = set()
    for shard in SHARDS:
        pool = [r for r in rows if r['source_shard'] == shard and r['split'] != 'quarantine']
        order = sorted(pool, key=lambda r: (len(r['technical_exclusions']), len(r['target_operations']), r['id']))
        for r in order[:3]:
            if r['id'] not in selected_ids:
                sample.append(r); selected_ids.add(r['id'])
        for flag in ['fingerspelling_content_and_operation', 'boundary_excluded_events', 'simultaneous_hand_events']:
            matches = [r for r in pool if flag in r['review_flags']]
            if matches:
                r = min(matches, key=lambda r: (len(r['target_operations']), r['id']))
                if r['id'] not in selected_ids:
                    sample.append(r); selected_ids.add(r['id'])
    for fragment in ['CLSFBI1603A_S037_B:129292:129762:', 'CLSFBI0211A_S004_B:265166:270426:']:
        for r in rows:
            if r['id'].startswith(fragment) and r['id'] not in selected_ids:
                sample.append(r); selected_ids.add(r['id'])
    write_rows(output/'review_sample.jsonl', sample)
    with (output/'review_queue.csv').open('w', newline='') as f:
        fields = ['id', 'content_sha256', 'split', 'french_text', 'target_text', 'review_flags', 'technical_exclusions', 'decision', 'reviewer', 'notes']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in fields[:5]}, 'review_flags': '|'.join(r['review_flags']),
                             'technical_exclusions': '|'.join(r['technical_exclusions']), 'decision': '', 'reviewer': '', 'notes': ''})
    manifest = {'task': TASK, 'policy': POLICY, 'policy_sha256': digest(POLICY),
                'tokenizer': {'path': str(tokenizer_path.resolve()), 'sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                              'model': 'google-t5/t5-base', 'revision': 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1', 'added_tokens': 0},
                'local_root': str(root.resolve()), 'sources': source,
                'counts': {'source_records': len(records), 'all_candidate_translations': len(rows),
                           'exploratory_train': len(train), 'exploratory_val': len(val),
                           'quarantined_conversation_rows': sum(r['split'] == 'quarantine' for r in rows),
                           'technically_excluded_rows': sum(bool(r['technical_exclusions']) for r in rows),
                           'review_sample': len(sample), 'approved_train': 0, 'approved_val': 0},
                'exclusion_counts': dict(collections.Counter(reason for r in rows for reason in r['technical_exclusions'])),
                'readiness': {'linguistically_validated': False, 'production_training_ready': False,
                              'exploratory_files': 'weakly source-aligned lexical projection only; all rows pending review',
                              'reason': 'source videos and annotation meaning must be reviewed; data preparation is not translation validation'},
                'test_shard_opened': False,
                'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.rglob('*') if p.is_file() and p.suffix != '.gz'}}
    write_json(output/'manifest.json', manifest)
    write_json(output/'readiness.json', manifest['readiness'])
    print(json.dumps(manifest['counts'], indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output, args.tokenizer)
