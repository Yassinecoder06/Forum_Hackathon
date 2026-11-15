"""Summarize completed real extraction outputs; never convert QC into approval."""
import argparse,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=Path('.runtime/cv-v4-all'));p.add_argument('--output',type=Path,default=Path('data/source_inventory/cv_v4_all_report.json'));a=p.parse_args()
d=json.loads((a.batch/'results.json').read_text());s=d['summary'];assert s['processed']==s['total_unique_videos'],'Batch is incomplete'
counts=collections.Counter();comparisons=[];frames=0;coverage=collections.Counter();recovered=lost=0
for r in d['results']:
 if r['status']!='complete':continue
 q=r['quality'];frames+=q['sampled_frames']
 for side in ['left','right']:
  if q[side+'_hand_coverage']<.8:coverage[side+'_below_80_percent']+=1
 for c in r['comparisons']:
  counts[c['gate']]+=1
  recovered+=sum(x['recovered'] for x in c.get('hands',{}).values());lost+=sum(x['lost'] for x in c.get('hands',{}).values())
  comparisons.append({'source_sha256':r['source_sha256'],'source_entries':r['aliases'],**c})
report={'kind':'Completed local CV-v4 extraction; engineering triage only','summary':s,'sampled_native_frames':frames,'comparison_statuses':dict(counts),'coverage_flags_unique_videos':dict(coverage),'recovered_hand_observations_common_timestamps_alias_weighted':recovered,'lost_hand_observations_common_timestamps_alias_weighted':lost,'worker_fingerprints':sorted({r.get('worker_fingerprint') for r in d['results'] if r.get('worker_fingerprint')}),'accuracy_validated':False,'active_assets_replaced':False,'comparisons':comparisons,'failures':[r for r in d['results'] if r['status']!='complete']}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['comparisons','failures']},indent=2))
