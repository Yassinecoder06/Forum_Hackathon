"""Resumable HF/PEFT trainer with explicit supervision and language gates."""
import argparse,json,hashlib
from pathlib import Path
import yaml

def load_splits(data, feasibility):
 manifest = json.loads((data / 'manifest.json').read_text())
 if manifest.get('language') != 'fsl':
  raise ValueError('Expected France-LSF dataset manifest')
 splits = {}
 for split in ('train', 'val'):
  path = data / f'fsl_ud_feasibility_{split}.jsonl'
  content = path.read_bytes()
  if hashlib.sha256(content).hexdigest() != manifest['files'][split]:
   raise ValueError(f'{split} checksum mismatch')
  rows = [json.loads(line) for line in content.splitlines()]
  if not rows or len(rows) != manifest['counts'][split]:
   raise ValueError(f'{split} row count differs from manifest')
  for row in rows:
   if (row.get('sign_language'), row.get('split'), row.get('annotation_convention')) != ('fsl', split, 'UD-FSL-STK-FORM'):
    raise ValueError(f'Unexpected language, split or annotation convention in {split}')
   if not feasibility and not (
    row.get('gold_translation_pair') is True and row.get('review_status') == 'approved'
    and row.get('alignment_status') == 'reviewed_aligned'
   ):
    raise ValueError(f'{split} contains an unreviewed translation pair. Use --feasibility for published source annotations.')
  splits[split] = rows
 for key in ('id', 'source_recording_id', 'split_group'):
  values = [{r[key] for r in splits[split]} for split in ('train', 'val')]
  if key == 'id' and any(len(ids) != len(rows) for ids, rows in zip(values, splits.values(), strict=True)):
   raise ValueError('Duplicate example IDs')
  if any(not v or None in v or '' in v for v in values) or values[0] & values[1]:
   raise ValueError(f'Missing or overlapping {key}')
 return splits

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('data/parallel'));p.add_argument('--output');p.add_argument('--feasibility',action='store_true');p.add_argument('--resume');p.add_argument('--preflight-only',action='store_true');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text())
 if cfg['sign_language']!='fsl':raise ValueError('Wrong sign language')
 splits = load_splits(a.data, a.feasibility)
 if a.preflight_only:
  print(json.dumps({'examples': {split: len(rows) for split, rows in splits.items()}, 'feasibility': a.feasibility}))
  return
 if not a.output:
  p.error('--output is required for training')
 from datasets import Dataset
 from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,Seq2SeqTrainer,Seq2SeqTrainingArguments,DataCollatorForSeq2Seq,EarlyStoppingCallback,set_seed
 from peft import LoraConfig,TaskType,get_peft_model
 import torch
 set_seed(cfg['seed'])
 model_path=cfg['local_model'];tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=False)
 def encode(r):
  x=tok(cfg['prefix']+r['french_text']);labels=tok(' '.join(r['gloss_sequence']))['input_ids']
  if len(x['input_ids'])>cfg['source_limit'] or len(labels)>cfg['target_limit']:raise ValueError('Sequence exceeds explicit budget')
  if tok.unk_token_id in labels:raise ValueError('Unknown target subtokens')
  return {**x,'labels':labels}
 data={s:Dataset.from_list([encode(r) for r in rows]) for s,rows in splits.items()}
 model=AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16)
 model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=cfg['lora_rank'],lora_alpha=2*cfg['lora_rank'],target_modules=['q','v'],lora_dropout=.05));model.enable_input_require_grads();model.config.use_cache=False
 args=Seq2SeqTrainingArguments(output_dir=a.output,num_train_epochs=20,per_device_train_batch_size=cfg['batch_size'],per_device_eval_batch_size=1,gradient_accumulation_steps=cfg['gradient_accumulation'],learning_rate=cfg['learning_rate'],bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='eval_loss',save_total_limit=2,seed=cfg['seed'],report_to=[],logging_steps=5)
 trainer=Seq2SeqTrainer(model=model,args=args,train_dataset=data['train'],eval_dataset=data['val'],data_collator=DataCollatorForSeq2Seq(tok,model=model),callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])
 trainer.train(resume_from_checkpoint=a.resume);out=Path(a.output);model.save_pretrained(out/'adapter');tok.save_pretrained(out/'tokenizer');(out/'config.json').write_text(json.dumps({**cfg,'linguistically_validated':False,'feasibility':a.feasibility},indent=2))
if __name__=='__main__':main()
