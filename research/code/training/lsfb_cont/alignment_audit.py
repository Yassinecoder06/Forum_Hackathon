"""Trace extracted LSFB lexical targets to source events; never grant linguistic approval."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from .prepare import digest, lexical_projection_events, write_rows


def row_evidence(row, source):
    if row['id'] != source['id']:
        raise ValueError('Source ID mismatch')
    if digest({k:v for k,v in source.items() if k != 'content_sha256'}) != source['content_sha256']:
        raise ValueError('Source content hash mismatch')
    if row['language'] != 'LSFB' or source['source_revision'] != '64b0c112f59d3710367fb736ef1808f37ff15e28':
        raise ValueError('Wrong language or source revision')
    events, duplicates = lexical_projection_events(source['aligned_events'])
    projected = [e['lemma'].strip() for e in events]
    start, end = row['interval_ms']
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('Invalid segment bounds')
    frame_deviations = []
    outside = 0
    for event in source['aligned_events']:
        a,b = event['start_ms'], event['end_ms']
        if not math.isfinite(a) or not math.isfinite(b) or b <= a:
            raise ValueError('Invalid event bounds')
        inside = max(0, min(end,b)-max(start,a)) / (b-a)
        outside += inside < .8-1e-9
        for field, timestamp in [('start_frame', a), ('end_frame', b)]:
            if field in event:
                frame_deviations.append(abs(event[field]*20-timestamp))
    return dict(id=row['id'], split=row['split'], source_sha256=source['content_sha256'],
                target_matches_source=projected == row['target'],
                interval_matches_source=row['interval_ms'] == source['interval_ms'],
                recording_matches_source=row['recording_id'] == source['recording_id'],
                below_80_percent_overlap_events=outside,
                max_frame_timestamp_deviation_ms=max(frame_deviations,default=None),
                nonlexical_event_count=sum(e.get('sign_type')!='lexical' for e in source['aligned_events']),
                collapsed_duplicate_events=len(duplicates),
                boundary_excluded_events=len(source['boundary_excluded_events']),
                review_status='pending', linguistically_validated=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=Path('.runtime/lsfb-cont-task-v3/candidates.jsonl'))
    ap.add_argument('--data',type=Path,default=Path('/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont-usable-v1'))
    ap.add_argument('--narrow',type=Path,default=Path('.runtime/lsfb-narrow-context-data-v1'))
    ap.add_argument('--output',type=Path,default=Path('research/private/lsfb-alignment-audit'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
    sources={r['id']:r for r in read(args.source)}
    rows=[r for split in ['train','val'] for r in read(args.data/(split+'.jsonl'))]
    evidence=[row_evidence(r,sources[r['id']]) for r in rows]
    narrow=[r for split in ['train','val'] for r in read(args.narrow/(split+'.jsonl')) if len(r['target'])>1]
    candidates=[]
    for r in narrow:
        s=dict(sources[r['id']]);s['experimental_lexical_target']=r['target']
        candidates.append(s)
    # Each original content hash is retained; experimental view fields do not mutate its canonical source.
    write_rows(args.output/'source_trace.jsonl',evidence)
    write_rows(args.output/'review_candidates.jsonl',candidates)
    counts=collections.Counter()
    for e in evidence:
        counts['rows']+=1
        for k in ['target_matches_source','interval_matches_source','recording_matches_source']:
            counts[k]+=e[k]
        counts['rows_with_nonlexical_events']+=e['nonlexical_event_count']>0
        counts['rows_with_below_80_percent_events']+=e['below_80_percent_overlap_events']>0
        counts['rows_with_frame_timestamp_deviation_above40ms']+=(e['max_frame_timestamp_deviation_ms'] or 0)>40
        counts['rows_with_excluded_boundary_events']+=e['boundary_excluded_events']>0
    result=dict(status='complete',sign_language='sfb',counts=dict(counts),
                source_candidates_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                review_pack_counts=dict(collections.Counter(r['split'] for r in candidates)),
                qualified_review=False,official_test_opened=False,
                limitations=['Source-event consistency does not prove French/sign semantic alignment.',
                             'Lexical targets omit pointing and other events; no nonmanual grammar completeness is established.',
                             'Frame consistency assumes the source 50 FPS annotation convention, not independent video synchronization.'])
    Path('research/text-to-gloss/lsfb/LSFB_SOURCE_ALIGNMENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
