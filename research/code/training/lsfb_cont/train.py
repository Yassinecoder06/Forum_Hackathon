"""Train an exploratory context-conditioned T5-base LSFB lexical planner.

The source rows are technically prepared but unreviewed. Metrics from this run
are exploratory and must not be described as validated translation quality.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from torch.utils.data import Subset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          TrainerCallback, set_seed)

try:
    from .prepare import parse_plan
except ImportError:
    from prepare import parse_plan

BASE = 'google-t5/t5-base'
REVISION = 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
MAX_TOKENS = 512
GENERATION_CAP = 320


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def rows_from(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def operation_sequence(text):
    return [f"SIGN:{op['lemma']}" if op['op'] == 'sign'
            else f"FINGERSPELL:{op['label']}:{op['annotation']}"
            for op in parse_plan(text.strip())]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def legacy_score(predictions, references):
    pred_seq, ref_seq, invalid = [], [], 0
    for pred, ref in zip(predictions, references):
        try:
            pred_seq.append(operation_sequence(pred))
        except (ValueError, KeyError, TypeError):
            invalid += 1
            pred_seq.append([])
        ref_seq.append(operation_sequence(ref))
    total_ref = sum(map(len, ref_seq))
    return {
        'operation_wer': sum(edit_distance(r, p) for p, r in zip(pred_seq, ref_seq)) / max(1, total_ref),
        'exact_match': sum(p == r for p, r in zip(pred_seq, ref_seq)) / max(1, len(ref_seq)),
        'invalid_plan_rate': invalid / max(1, len(predictions)),
        'mean_predicted_operations': float(np.mean([len(x) for x in pred_seq])) if pred_seq else 0.0,
        'mean_reference_operations': float(np.mean([len(x) for x in ref_seq])) if ref_seq else 0.0,
        'examples': len(ref_seq),
        'reference_operations': total_ref,
    }


try:
    from .scoring import score
except ImportError:
    from scoring import score


class EncodedDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.items = []
        for row in rows:
            source = tokenizer(row['input_text'], truncation=False)
            target = tokenizer(row['target_text'], truncation=False)
            if len(source['input_ids']) > MAX_TOKENS or len(target['input_ids']) > MAX_TOKENS:
                raise ValueError('Frozen row exceeds model limit; truncation is forbidden: ' + row['id'])
            if not target['input_ids'] or tokenizer.unk_token_id in target['input_ids']:
                raise ValueError('Target is empty or has unknown tokenizer units: ' + row['id'])
            if parse_plan(row['target_text']) != parse_plan(tokenizer.decode(target['input_ids'], skip_special_tokens=True)):
                raise ValueError('Target does not roundtrip through the pinned tokenizer: ' + row['id'])
            self.items.append({'input_ids': source['input_ids'], 'attention_mask': source['attention_mask'],
                               'labels': target['input_ids']})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class ProgressCallback(TrainerCallback):
    def __init__(self, path, planned_steps):
        self.path = path
        self.planned_steps = planned_steps
        self.started = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        dump(self.path, {'status': 'training', 'completed_steps': state.global_step,
                         'planned_steps': self.planned_steps, 'started_at_epoch': state.epoch,
                         'elapsed_seconds': 0})

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = logs or {}
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in values.values()):
            raise ValueError('Non-finite training or validation log; stopping this run')
        dump(self.path, {'status': 'training', 'completed_steps': state.global_step,
                         'planned_steps': self.planned_steps, 'epoch': state.epoch,
                         'elapsed_seconds': time.monotonic() - self.started,
                         'latest_log': values})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        dump(self.path, {'status': 'evaluated', 'completed_steps': state.global_step,
                         'planned_steps': self.planned_steps, 'epoch': state.epoch,
                         'elapsed_seconds': time.monotonic() - self.started,
                         'validation': metrics or {}})

    def on_train_end(self, args, state, control, **kwargs):
        dump(self.path, {'status': 'training_complete', 'completed_steps': state.global_step,
                         'planned_steps': self.planned_steps, 'epoch': state.epoch,
                         'elapsed_seconds': time.monotonic() - self.started})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('.runtime/lsfb-cont-task-v3'))
    ap.add_argument('--output', type=Path, default=Path('.runtime/lsfb-cont-local-v1'))
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--gradient-accumulation', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--max-steps', type=int, default=-1, help='Optional cap for a short local run')
    ap.add_argument('--selection-validation-size', type=int, default=128)
    ap.add_argument('--eval-batch-size', type=int, default=4)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('Output directory already exists; choose a new run directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing to silently train on CPU')
    torch.set_num_threads(2)
    set_seed(42)
    random.seed(42)

    manifest = json.loads((args.data / 'manifest.json').read_text())
    readiness = json.loads((args.data / 'readiness.json').read_text())
    if manifest['policy']['task'] != 'lsfb-cont-context-lexical-plan-v1':
        raise ValueError('Unexpected dataset task')
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((args.data / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen dataset checksum mismatch: ' + name)
    train_rows = rows_from(args.data / 'exploratory/train.jsonl')
    val_rows = rows_from(args.data / 'exploratory/val.jsonl')
    if not train_rows or not val_rows:
        raise ValueError('Both train and validation must contain rows')
    if {r['conversation_id'] for r in train_rows} & {r['conversation_id'] for r in val_rows}:
        raise ValueError('Conversation overlap detected')

    args.output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained('.runtime/lsfb-cont-tokenizer',
                                              use_fast=True, local_files_only=True)
    train_data, val_data = EncodedDataset(train_rows, tokenizer), EncodedDataset(val_rows, tokenizer)
    selection_rng = random.Random(42)
    selection_indices = sorted(selection_rng.sample(range(len(val_data)),
                                  min(args.selection_validation_size, len(val_data))))
    selection_val_data = Subset(val_data, selection_indices)
    steps_per_epoch = math.ceil(len(train_data) / (args.batch_size * args.gradient_accumulation))
    planned_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    run_info = {
        'status': 'preflight_passed', 'kind': 'exploratory training; weakly aligned and linguistically unreviewed',
        'linguistically_validated': bool(readiness.get('linguistically_validated', False)),
        'accepted_train_rows': manifest['counts'].get('approved_train', 0),
        'accepted_validation_rows': manifest['counts'].get('approved_val', 0),
        'exploratory_train_rows': len(train_data), 'exploratory_validation_rows': len(val_data),
        'train_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        'model': BASE, 'model_revision': REVISION, 'architecture': 'T5-base with LoRA q/v rank 8',
        'tokenizer_added_tokens': 0, 'seed': 42,
        'max_source_tokens': MAX_TOKENS, 'max_target_tokens': MAX_TOKENS,
        'target_truncation': False, 'generation_cap': GENERATION_CAP,
        'epochs_requested': args.epochs, 'max_steps': args.max_steps,
        'batch_size': args.batch_size, 'gradient_accumulation': args.gradient_accumulation,
        'selection_validation_size': len(selection_val_data),
        'selection_validation_seed': 42, 'final_validation_size': len(val_data),
        'eval_batch_size': args.eval_batch_size,
        'learning_rate': args.learning_rate, 'official_test_opened': False,
        'gpu': torch.cuda.get_device_name(0), 'gpu_vram_bytes': torch.cuda.get_device_properties(0).total_memory,
        'quality_claim': 'none; source examples have not been linguistically reviewed',
    }
    dump(args.output / 'RUN_PLAN.json', run_info)
    dump(args.output / 'progress.json', {'status': 'downloading_or_loading_model', 'planned_steps': planned_steps})

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE, revision=REVISION, torch_dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8,
                                              lora_alpha=16, lora_dropout=0.05,
                                              target_modules=['q', 'v']))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100,
                                      pad_to_multiple_of=8)

    def compute_metrics(result):
        predictions = result.predictions[0] if isinstance(result.predictions, tuple) else result.predictions
        predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
        labels = np.where(result.label_ids < 0, tokenizer.pad_token_id, result.label_ids)
        pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ref_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return score(pred_text, ref_text)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output / 'checkpoints'), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, weight_decay=0.01, optim='adamw_torch',
        lr_scheduler_type='cosine', warmup_ratio=0.03, gradient_checkpointing=True,
        fp16=True, bf16=False, predict_with_generate=True,
        generation_max_length=GENERATION_CAP, generation_num_beams=1,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='operation_wer',
        greater_is_better=False, logging_steps=20, logging_nan_inf_filter=False,
        report_to='none', seed=42, dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=selection_val_data,
        data_collator=collator, processing_class=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience),
                   ProgressCallback(args.output / 'progress.json', planned_steps)],
    )
    dump(args.output / 'progress.json', {
        'status': 'training', 'completed_steps': 0, 'planned_steps': planned_steps,
        'train_rows': len(train_data), 'validation_rows': len(val_data),
        'steps_per_epoch': steps_per_epoch,
    })
    print(json.dumps({'event': 'training_started', 'train_rows': len(train_data),
                      'validation_rows': len(val_data), 'steps_per_epoch': steps_per_epoch,
                      'planned_steps': planned_steps, 'gpu': torch.cuda.get_device_name(0),
                      'linguistically_validated': run_info['linguistically_validated']}), flush=True)
    started = time.monotonic()
    outcome = trainer.train()
    trainer.model.save_pretrained(args.output / 'adapter')
    tokenizer.save_pretrained(args.output / 'tokenizer')
    validation = trainer.predict(val_data, metric_key_prefix='final_validation')
    pred_ids = validation.predictions[0] if isinstance(validation.predictions, tuple) else validation.predictions
    pred_ids = np.where(pred_ids < 0, tokenizer.pad_token_id, pred_ids)
    ref_ids = np.where(validation.label_ids < 0, tokenizer.pad_token_id, validation.label_ids)
    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    ref_text = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
    (args.output / 'validation_predictions.jsonl').write_text(''.join(
        json.dumps({'id': row['id'], 'prediction': pred.strip(), 'reference': ref.strip(),
                    'conversation_id': row['conversation_id']}, ensure_ascii=False) + '\n'
        for row, pred, ref in zip(val_rows, pred_text, ref_text)))
    final = {
        **run_info, 'status': 'complete', 'best_checkpoint': trainer.state.best_model_checkpoint,
        'best_validation_operation_wer': trainer.state.best_metric,
        'steps_completed': trainer.state.global_step,
        'train_loss': float(outcome.training_loss),
        'elapsed_seconds': time.monotonic() - started,
        'peak_cuda_memory_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'validation': score(pred_text, ref_text),
        'training_history': trainer.state.log_history,
    }
    dump(args.output / 'RESULT.json', final)
    dump(args.output / 'progress.json', {'status': 'complete', 'completed_steps': trainer.state.global_step,
                                          'planned_steps': planned_steps, 'validation': final['validation'],
                                          'elapsed_seconds': final['elapsed_seconds']})
    print(json.dumps({'event': 'training_complete', 'steps': trainer.state.global_step,
                      'best_validation_operation_wer': trainer.state.best_metric,
                      'validation': final['validation'], 'elapsed_seconds': final['elapsed_seconds'],
                      'peak_cuda_memory_mib': final['peak_cuda_memory_mib']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
