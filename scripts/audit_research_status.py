"""Collect current evidence without equating passing software checks with signing quality."""
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence={}
    for name in ['data/parallel/manifest.json','research/private/fsl-mt5-small-comparison-v1/RESULT.json',
                 'research/private/fsl-t5-base-comparison-v3/RESULT.json','research/private/fsl-motion-v1/RESULTS.json',
                 'research/text-to-gloss/lsfb/LSFB_SEED_REPLICATION_RESULT.json','research/text-to-gloss/lsfb/LSFB_SOURCE_ALIGNMENT_AUDIT.json',
                 'research/audio/EMOTION2VEC_WEIGHT_AUDIT.json','docs/verification/CAPTURE_LANGUAGE_MIGRATION.json',
                 'docs/verification/CAPTURE_LANGUAGE_ASSET_RECHECK.json','configs/asr_model.json',
                 'research/audio/ASR_SMOKE_RESULT.json','docs/verification/VIDEO_API_LANDMARK_RECHECK.json',
                 'docs/verification/STUDIO_PLAYBACK_RECHECK.md','research/text-to-gloss/fsl/FSL_TRAINING_REVIEW_PACKET.md',
                 'research/motion/FSL_MOTION_INVENTORY_FOLLOWUP.json','research/motion/FSL_VISIBLE_ENDPOINTS_LEARNED.json',
                 'research/motion/FSL_CAPTION_POSE_BASELINE.json','research/text-to-gloss/lsfb/LSFB_FREQUENT_CONTEXT_RESULT.md']:
        evidence[name]={'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size}
    manifest=read('data/parallel/manifest.json')
    for split,digest in manifest['files'].items():
        assert sha(ROOT/f'data/parallel/fsl_ud_feasibility_{split}.jsonl')==digest
    comparisons={}
    for name,folder in [('mt5-small','fsl-mt5-small-comparison-v1'),('t5-base','fsl-t5-base-comparison-v3')]:
        prefix='research/private/'+folder+'/';result=read(prefix+'RESULT.json');config=read(prefix+'CONFIG.json')
        assert result['status']=='complete' and result['examples']==len(read(prefix+'predictions.json'))
        for split,digest in config['data_sha256'].items():assert manifest['files'][split]==digest
        assert (ROOT/prefix/'adapter/adapter_model.safetensors').is_file()
        comparisons[name]={k:result[k] for k in ['examples','exact','wer','bleu','chrf','cap_hits','peak_cuda_mb']}
    seeds=read('research/text-to-gloss/lsfb/LSFB_SEED_REPLICATION_RESULT.json')
    for r in seeds['runs']:
        folder='lsfb-narrow-current-run-v1' if r['seed']==84 else f"lsfb-narrow-current-seed{r['seed']}-v1"
        assert sha(ROOT/'.runtime'/folder/'RESULT.json')==r['result_sha256']
    with sqlite3.connect('file:'+str(ROOT/'data/catalog/catalog.sqlite')+'?mode=ro',uri=True) as db:
        assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
        reviews=db.execute('select count(*) from reviews').fetchone()[0]
    library=read('.runtime/library.json');migration=read('docs/verification/CAPTURE_LANGUAGE_MIGRATION.json')
    migration_snapshot_matches_current=sha(ROOT/'.runtime/library.json')==migration['after_sha256']
    assert all(r.get('sign_language')=='fsl' for r in library)
    small=comparisons['mt5-small'];base=comparisons['t5-base']
    partial=ROOT/'.runtime/mt5-base/pytorch_model.bin.part'
    asr_config=read('configs/asr_model.json')
    asr_files={name: (ROOT/'.models/faster-whisper-small'/name).is_file()
               and sha(ROOT/'.models/faster-whisper-small'/name)==digest
               for name,digest in asr_config['file_hashes'].items()}
    asr_smoke=read('research/audio/ASR_SMOKE_RESULT.json')
    landmark_result = read('docs/verification/VIDEO_API_LANDMARK_RECHECK.json')
    assert landmark_result['source_sha256'] == sha(ROOT/'data/public_examples/bonjour.webm')
    coverage = read('research/motion/FSL_MOTION_INVENTORY_FOLLOWUP.json')
    motion_experiment = read('research/motion/FSL_VISIBLE_ENDPOINTS_LEARNED.json')
    caption_experiment = read('research/motion/FSL_CAPTION_POSE_BASELINE.json')
    gates={
        'qualified_linguistic_reviews_present':reviews>0,
        'france_lsf_parallel_pairs_locally_reviewed':manifest['local_reviewed_pairs']>0,
        'france_lsf_comparison_has_any_development_exact':any(r['exact']>0 for r in comparisons.values()),
        'belgian_seed_study_has_any_multisign_exact':any(r['multi_sign']['exact']>0 for r in seeds['runs']),
        'validated_2d_to_3d_lifting_available':False,
        'fluent_signer_comprehension_evaluation_available':False,
    }
    result=dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                objective_complete=False,scientific_gates=gates,
                france_lsf=dict(manifest_counts=manifest['counts'],local_reviewed_pairs=manifest['local_reviewed_pairs'],completed_model_results=comparisons),
                video_landmark_roundtrip=landmark_result,
                motion_coverage=coverage,
                motion_inbetweening=dict(counts=motion_experiment['counts'], metrics=motion_experiment['metrics']),
                weak_caption_pose=caption_experiment,
                belgian_lsfb=dict(seed_count=len(seeds['runs']),development_examples=seeds['retrieval']['examples'],
                                  mean_operation_wer=seeds['aggregate']['operation_wer']['mean'],independent_test=False),
                speech=dict(asr_model=asr_config['model_id'],asr_revision=asr_config['revision'],
                            asr_license=asr_config['license'],local_model_files_verified=all(asr_files.values()),
                            model_file_checks=asr_files,
                            smoke=dict(status_code=asr_smoke['status_code'],
                                       transcript=asr_smoke['transcription']['text'],
                                       device=asr_smoke['transcription']['device'],
                                       evaluation_scope=asr_smoke['evaluation_scope'],
                                       accuracy_benchmark=False)),
                catalog_review_records=reviews,library_explicit_fsl_records=len(library),
                migration_snapshot_matches_current_library=migration_snapshot_matches_current,
                unusable_mt5_base_partial_bytes=partial.stat().st_size if partial.exists() else 0,
                evidence=evidence,
                interpretation='Artifact integrity and measured failures are verified; absence of human validation cannot be supplied by software tests.',
                next_required_external_input='Qualified France-LSF/Belgian LSFB review of source alignment and representation, or a newly accessible appropriately licensed verified parallel corpus.')
    (ROOT/'research/CURRENT_RESEARCH_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))


if __name__=='__main__':main()
