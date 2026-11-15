"""Protocol checks for tokenizer diagnosis and the single longer research run."""
import json
from pathlib import Path
from catalog import ROOT,sha

RUN=ROOT/'research/runs/mt5-small-lora-long-validation-selected'
def load(path):return json.loads(Path(path).read_text())

def test_train_only_audit_is_lossless_and_authentic():
    a=load(ROOT/'research/runs/mt5-train-only-diagnostic/tokenizer_audit.json')
    train=[json.loads(x) for x in (ROOT/'data/text_to_gloss/v002/train.jsonl').read_text().splitlines()]
    assert a['train_sha256']==sha(ROOT/'data/text_to_gloss/v002/train.jsonl')
    assert a['records']==len(train)==61
    assert a['round_trip_mismatches']==[] and a['decoder_shift_verified']
    expected={r['pair_id']:r['gloss_sequence'] for r in train}
    assert {r['pair_id']:r['raw_target'] for r in a['all_train_targets']}==expected
    assert all(r['unk_count']==r['sentinel_target_count']==0 and r['eos_at_end'] for r in a['all_train_targets'])
    assert a['selected_pair_ids']==sorted(expected)[:4]
    assert sum(p['count'] for p in a['trainable_tensors'])==a['trainable_parameters_before_freeze']==2555904
    assert all('lora_A' in p['name'] or 'lora_B' in p['name'] for p in a['trainable_tensors'])

def test_frozen_data_and_prior_runs_remain_unchanged():
    p=load(ROOT/'research/MT5_LONG_RUN_PLAN.json')
    for group in ['frozen_data','preserved_historical_artifacts']:
        for filename,expected in p[group].items():assert sha(ROOT/filename)==expected

def test_long_run_follows_locked_configuration():
    plan=load(ROOT/'research/MT5_LONG_RUN_PLAN.json');p=load(RUN/'training_plan.json')
    a=p['arguments'];s=plan['settings']
    assert a['epochs']==s['max_epochs']==100
    assert a['patience']==s['patience']==10
    for key in ['rank','alpha','lr','accumulation','min_delta','warmup_ratio','weight_decay']:
        assert a[key]==s[key]
    assert a['targets'].split(',')==s['targets']
    assert a['validation_loss']=='token'
    assert p['trainable_parameters_before_freeze']==2555904
    assert sum(t['parameters'] for t in p['trainable_tensors'])==2555904

def test_validation_selection_precedes_single_final_test_result():
    m=load(RUN/'metrics.json');s=load(RUN/'selection_complete.json');h=m['loss_history']
    assert s['test_read'] is False
    assert 1<=m['epochs']<=m['max_epochs']==100
    assert m['patience']==10
    best=min(h,key=lambda r:r['val_loss'])
    assert m['best_epoch']==s['best_epoch']==best['epoch']
    assert s['best_validation_loss']==best['val_loss']
    if m['epochs']<100:assert m['epochs']-m['best_epoch']==10
    assert m['trainable_parameters']==2555904
    assert m['trainable_parameter_count_stage']=='before selected adapter load/freeze'
    assert m['counts']=={'train':61,'val':28,'test':39}
    predictions=[json.loads(x) for x in (RUN/'test_predictions.jsonl').read_text().splitlines()]
    assert len(predictions)==len({p['pair_id'] for p in predictions})==39
    assert all(set(p['predictions'])=={'retrieval','lexical','model'} for p in predictions)
    assert m['metrics']['test']['model']['examples']==39
