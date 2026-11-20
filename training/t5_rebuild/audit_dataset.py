"""Read-only audit of frozen LSFB pairs, thin sources and generated predictions.

Heuristic review flags are not linguistic errors and never filter the dataset.
Only Python's standard library is needed. Source payloads are local backups.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path

from prepare import canonical_event


def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def load(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text())


def summary(values):
    values = sorted(values)
    if not values:
        return {}
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'p95': values[int((len(values)-1)*.95)],
            'max': values[-1]}


def edit_counts(ref, pred):
    """Deterministic minimum-edit alignment; ties prefer match/substitution."""
    previous = [(j, 0, 0, j) for j in range(len(pred)+1)]
    for i, r in enumerate(ref, 1):
        row = [(i, 0, i, 0)]
        for j, p in enumerate(pred, 1):
            cost, s, d, ins = previous[j-1]
            diagonal = (cost+(r != p), s+(r != p), d, ins)
            cost, s, d, ins = previous[j]
            deletion = (cost+1, s, d+1, ins)
            cost, s, d, ins = row[-1]
            insertion = (cost+1, s, d, ins+1)
            row.append(min([diagonal, deletion, insertion], key=lambda x: x[0]))
        previous = row
    return previous[-1]


def event_key(e):
    return e['start_ms'], e['end_ms'], e['gloss']


def audit(data, raw, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = load(data/'manifest.json')
    hashes = {name: hashlib.sha256((data/name).read_bytes()).hexdigest() == digest
              for name, digest in manifest['files'].items()}
    assert all(hashes.values()), 'Frozen dataset checksum mismatch'
    splits = {name: [json.loads(l) for l in (data/(name+'.jsonl')).read_text().splitlines()]
              for name in ['train', 'val']}
    sources = {}
    shard_stats = []
    raw_issues = collections.Counter()
    for p in sorted(raw.glob('annotations_*.json.gz')):
        payload = load(p)
        stats = collections.Counter()
        for rid, record in payload['records'].items():
            sources[rid] = record
            translations = record.get('annotations.translation.json', [])
            events = record.get('annotations.both_hands.json', [])
            stats['recordings'] += 1
            stats['translations'] += len(translations)
            stats['events'] += len(events)
            stats['recordings_with_translations'] += bool(translations)
            stats['recordings_without_translations'] += not bool(translations)
            for tr in translations:
                if tr['end_ms'] <= tr['start_ms']:
                    raw_issues['nonpositive_translation_duration'] += 1
                    continue
                intersecting = [e for e in events if e['end_ms'] > e['start_ms']
                                and min(tr['end_ms'], e['end_ms']) > max(tr['start_ms'], e['start_ms'])]
                selected = [e for e in intersecting if
                            (min(tr['end_ms'], e['end_ms'])-max(tr['start_ms'], e['start_ms'])) /
                            (e['end_ms']-e['start_ms']) >= .8]
                stats['translations_without_selected_events'] += not bool(selected)
                stats['translations_with_boundary_dropped_events'] += len(selected) < len(intersecting)
                stats['intersecting_event_assignments'] += len(intersecting)
                stats['selected_event_assignments'] += len(selected)
            for e in events:
                raw_issues['nonpositive_event_duration'] += e['end_ms'] <= e['start_ms']
                raw_issues['empty_event_gloss'] += not bool(e.get('gloss', '').strip())
                for bound in ['start', 'end']:
                    if bound+'_frame' in e:
                        raw_issues['event_frame_ms_disagreement_over_40ms_at_50fps'] += abs(e[bound+'_frame']*20-e[bound+'_ms']) > 40
        shard_stats.append({'shard': payload['source_shard'], 'role': payload['official_role'], **stats})

    profiles = {}
    all_flags = []
    integrity = collections.Counter()
    for name, rows in splits.items():
        units = collections.Counter(u for r in rows for u in r['units'])
        flags = collections.Counter()
        rawforms = collections.defaultdict(collections.Counter)
        target_by_text = collections.defaultdict(set)
        event_owners = collections.Counter()
        for r in rows:
            record = sources[r['recording_id']]
            a, b = r['translation_interval_ms']
            matches = [tr for tr in record.get('annotations.translation.json', [])
                       if tr['start_ms'] == a and tr['end_ms'] == b and
                       ' '.join(tr['text'].split()) == r['french_text']]
            integrity['rows_without_exact_source_translation'] += not bool(matches)
            integrity['signer_id_mismatch'] += r['signer_id'] != record.get('signer.txt')
            raw_events = {event_key(e): e for e in record.get('annotations.both_hands.json', [])}
            integrity['events_absent_from_source'] += sum(event_key(e) not in raw_events for e in r['events'])
            integrity['canonical_target_mismatch'] += r['units'] != [canonical_event(e) for e in r['events']]
            target_by_text[norm(r['french_text'])].add(tuple(r['units']))
            rowflags = []
            nonlex = sum(e.get('sign_type') != 'lexical' for e in r['events'])
            fs = [e for e in r['events'] if e.get('specifier') == 'fs']
            if nonlex:
                rowflags.append('nonlexical_context_in_target')
            if fs:
                rowflags.append('fingerspelling_marker_removed_from_target')
            if re.search(r'…|\.\.\.', r['french_text']):
                rowflags.append('ellipsis_in_french')
            if len(r['units']) > 2*max(1, len(r['french_text'].split())):
                rowflags.append('target_over_twice_french_word_count')
            if any(x == y for x, y in zip(r['units'], r['units'][1:])):
                rowflags.append('consecutive_repeated_target_units')
            if any(e['start_ms'] < a or e['end_ms'] > b for e in r['events']):
                rowflags.append('selected_event_crosses_translation_boundary')
            overlap_pairs = 0
            for i, e in enumerate(r['events']):
                unit = canonical_event(e)
                rawforms[unit][e['gloss']] += 1
                event_owners[(r['recording_id'], *event_key(e))] += 1
                for other in r['events'][i+1:]:
                    if other['start_ms'] < e['end_ms'] and e['start_ms'] < other['end_ms']:
                        overlap_pairs += 1
            if overlap_pairs:
                rowflags.append('simultaneous_events_serialized')
            flags.update(rowflags)
            all_flags.append({'split': name, 'id': r['id'], 'french_text': r['french_text'],
                              'target_units': ' '.join(r['units']), 'flags': '|'.join(rowflags),
                              'target_length': len(r['units']), 'nonlexical_events': nonlex,
                              'overlapping_event_pairs': overlap_pairs,
                              'review_status': 'unreviewed; flags are not error labels'})
        profiles[name] = {'pairs': len(rows), 'recordings': len({r['recording_id'] for r in rows}),
                          'signers': len({r['signer_id'] for r in rows}),
                          'target_tokens': sum(units.values()), 'target_types': len(units),
                          'singleton_types': sum(v == 1 for v in units.values()),
                          'types_occurring_at_most_5_times': sum(v <= 5 for v in units.values()),
                          'target_length': summary([len(r['units']) for r in rows]),
                          'french_word_count': summary([len(r['french_text'].split()) for r in rows]),
                          'duration_seconds': summary([(r['translation_interval_ms'][1]-r['translation_interval_ms'][0])/1000 for r in rows]),
                          'event_types': dict(collections.Counter(e.get('sign_type') for r in rows for e in r['events'])),
                          'review_flags': dict(flags),
                          'events_assigned_to_multiple_translations': sum(n > 1 for n in event_owners.values()),
                          'texts_with_multiple_targets': sum(len(v) > 1 for v in target_by_text.values()),
                          'normalization_units_merging_multiple_raw_labels': sum(len(v) > 1 for v in rawforms.values()),
                          'top_targets': units.most_common(20),
                          'fingerspelling_examples': [{'id': r['id'], 'french': r['french_text'],
                                                      'events': [e for e in r['events'] if e.get('specifier') == 'fs']}
                                                     for r in rows if any(e.get('specifier') == 'fs' for e in r['events'])][:8]}

    train_units = collections.Counter(u for r in splits['train'] for u in r['units'])
    vocabulary = load(data/'vocabulary.json')
    isol = {u for u, origins in vocabulary['origins'].items() if 'ISOL_TRAIN' in origins}
    val_units = [u for r in splits['val'] for u in r['units']]
    tr_rec = {r['recording_id'] for r in splits['train']}
    va_rec = {r['recording_id'] for r in splits['val']}
    # Last letter may encode a camera/participant; report group overlap, do not assume its semantics.
    group = lambda rid: re.sub('[A-Za-z]$', '', rid.split('_')[0])
    overlaps = {'recordings': sorted(tr_rec & va_rec),
                'signers': sorted({r['signer_id'] for r in splits['train']} & {r['signer_id'] for r in splits['val']}),
                'exact_conversation_prefixes': sorted({r.split('_')[0] for r in tr_rec} & {r.split('_')[0] for r in va_rec}),
                'conversation_prefixes_ignoring_final_letter': sorted({group(r) for r in tr_rec} & {group(r) for r in va_rec}),
                'normalized_french_inputs': sorted({norm(r['french_text']) for r in splits['train']} & {norm(r['french_text']) for r in splits['val']})}
    prediction_metrics = {}
    predpath = raw/'test_predictions.json.gz'
    if predpath.exists():
        predictions = load(predpath)
        for constrained in [True, False]:
            part = [p for p in predictions if p['constrained'] == constrained]
            edits = [edit_counts(p['reference'], p['prediction']) for p in part]
            ref_tokens = sum(len(p['reference']) for p in part)
            ngrams = lambda seq: list(zip(seq, seq[1:], seq[2:]))
            target_counts = collections.Counter(tuple(p['prediction']) for p in part)
            prediction_metrics['constrained' if constrained else 'raw'] = {
                'examples': len(part), 'reference_tokens': ref_tokens,
                'predicted_tokens': sum(len(p['prediction']) for p in part),
                'substitutions': sum(e[1] for e in edits), 'deletions': sum(e[2] for e in edits),
                'insertions': sum(e[3] for e in edits),
                'wer': sum(e[0] for e in edits)/max(1, ref_tokens),
                'mean_prediction_length': statistics.mean(len(p['prediction']) for p in part),
                'mean_reference_length': ref_tokens/len(part),
                'sequences_with_repeated_trigrams': sum(len(ngrams(p['prediction'])) > len(set(ngrams(p['prediction']))) for p in part),
                'reference_sequences_with_repeated_trigrams': sum(len(ngrams(p['reference'])) > len(set(ngrams(p['reference']))) for p in part),
                'without_eos': sum(not p['eos'] for p in part),
                'invalid_sequences': sum(bool(p['invalid_tokens']) for p in part),
                'most_common_predictions': [{'units': list(k), 'count': v} for k, v in target_counts.most_common(3)],
                'top_predicted_units': collections.Counter(u for p in part for u in p['prediction']).most_common(15),
                'top_reference_units': collections.Counter(u for p in part for u in p['reference']).most_common(15)}
    history = load(raw/'history.json.gz') if (raw/'history.json.gz').exists() else []
    result = {'scope': 'All frozen train/validation rows and seven thin source shards; official test predictions diagnostic only. No training or filtering performed.',
              'frozen_hashes_match': hashes, 'source_shards': shard_stats,
              'source_integrity_checks': dict(raw_issues), 'derived_integrity_checks': dict(integrity),
              'splits': profiles, 'split_overlaps': overlaps,
              'vocabulary': {'inventory': len(vocabulary['units']), 'paired_train_types': len(train_units),
                             'never_targeted_inventory_units': len(set(vocabulary['units'])-set(train_units)),
                             'isol_train_units': len(isol), 'train_tokens_with_isol_assets': sum(v for u, v in train_units.items() if u in isol),
                             'train_pairs_fully_inside_isol_inventory': sum(all(u in isol for u in r['units']) for r in splits['train']),
                             'val_tokens_unseen_in_paired_train': sum(u not in train_units for u in val_units),
                             'val_tokens_outside_inventory': sum(u not in vocabulary['origins'] for u in val_units)},
              'generated_test_diagnostics': prediction_metrics,
              'validation_history': [h for h in history if 'eval_wer' in h],
              'late_training_losses': [h for h in history if 'loss' in h][-10:],
              'limitations': ['No automatic flag proves linguistic invalidity.',
                              'Thin source snapshots omit separate left/right hand tiers; one original local shard can be audited separately.',
                              'No free-running training predictions available: train-fit vs generalization is not resolved.',
                              'Full source video/context and LSFB expertise needed to verify suspected semantic mismatches.',
                              'Vocabulary inventory coverage is not evidence of learned translation or motion coverage.']}
    (output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (output/'review_queue.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_flags[0]))
        writer.writeheader()
        writer.writerows(all_flags)
    print(json.dumps({k: result[k] for k in ['derived_integrity_checks', 'split_overlaps', 'vocabulary', 'generated_test_diagnostics']}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data', type=Path)
    p.add_argument('raw', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    audit(a.data, a.raw, a.output)
