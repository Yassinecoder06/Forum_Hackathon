"""Verify local published UD FORM targets against original CoNLL-U; keep SUD separate."""
from pathlib import Path
import json,hashlib,collections

def main():
 raw=Path('data/text_to_gloss/raw/stk/fsl_stkautogramm-ud-test.conllu');segments={}
 for block in raw.read_text().strip().split('\n\n'):
  meta={};forms=[]
  for line in block.splitlines():
   if line.startswith('# ') and ' = ' in line:k,v=line[2:].split(' = ',1);meta[k]=v
   elif line and not line.startswith('#'):
    cells=line.split('\t')
    if cells[0].isdigit():forms.append(cells[1])
  if 'sent_id' in meta:segments[meta['sent_id']]=(meta,forms)
 report={'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'language':'fsl','local_reviewed_pairs':0,'mode':'published_annotated_feasibility_not_verified_translation','counts':{},'files':{}}
 seen={}
 for split in ['train','val','test']:
  rows=[]
  for r in map(json.loads,Path(f'data/text_to_gloss/v002/{split}.jsonl').read_text().splitlines()):
   if r['annotation_scheme']!='UD-FSL-STK-FORM':continue
   assert r['language_variant']=='LSF_FR' and r['license_id']=='CC-BY-SA-4.0'
   forms=[]
   for sid in r['source_segment_ids']:
    meta,fs=segments[sid];assert meta['text_fr']==r['french_text'];forms.extend(fs)
   assert ' '.join(forms)==r['gloss_sequence']
   group=r['split_group'];assert group not in seen or seen[group]==split;seen[group]=split
   rows.append({'id':r['pair_id'],'sign_language':'fsl','french_text':r['french_text'],'gloss_sequence':forms,'annotation_convention':'UD-FSL-STK-FORM','source_dataset':'UD_French_Sign_Language-STKAutogramm','source_recording_id':r['recording'],'source_segment_ids':r['source_segment_ids'],'split_group':group,'signer_id':r['speaker_group'],'split':split,'original_assignment':'upstream UD test; local recording-group research split','alignment_status':'source_free_translation; not formal parallel','review_status':'published_expert_annotation; pending_local_LSF_review','license':'CC-BY-SA-4.0','source_revision':r['source_revision'],'gold_annotation':True,'gold_translation_pair':False})
  p=Path(f'data/parallel/fsl_ud_feasibility_{split}.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));report['counts'][split]=len(rows);report['files'][split]=hashlib.sha256(p.read_bytes()).hexdigest()
 Path('data/parallel/manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
