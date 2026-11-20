"""Conversation-held-out LSFB frequent-label development experiment.

Uses only the previously prepared LSFB-CONT exploratory training conversations.
The source annotations remain unreviewed; results are a modeling diagnostic,
not evidence of valid LSFB translation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from .prepare import parse_plan
from .scoring import distance

SOURCE = Path(".runtime/lsfb-cont-task-v3/candidates.jsonl")
BASE = "google-t5/t5-base"
REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
TOKENIZER = Path(".runtime/lsfb-cont-local-v3/tokenizer")
SEED = 111
MIN_LABEL_COUNT = 20
MIN_LABEL_CONVERSATIONS = 3
HOLDOUT_FRACTION = 0.20
MIN_OPS, MAX_OPS = 2, 8


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_candidates():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    # Restrict the split to official-source training conversations. The local
    # candidate file also contains validation rows, excluded before splitting.
    rows = [r for r in rows if r.get("split") == "train" and not r.get("technical_exclusions")]
    if not rows or any(r.get("review_status") != "pending" for r in rows):
        raise ValueError("Expected nonempty, technically eligible, pending LSFB rows")
    return rows


def build_split(rows):
    conversations = sorted({r["conversation_id"] for r in rows})
    order = sorted(conversations, key=lambda c: hashlib.sha256(("lsfb-frequent-context-v1:" + c).encode()).hexdigest())
    n_holdout = max(1, round(len(order) * HOLDOUT_FRACTION))
    holdout = set(order[:n_holdout])
    train = [r for r in rows if r["conversation_id"] not in holdout]
    dev = [r for r in rows if r["conversation_id"] in holdout]

    # Select labels only from the model-training conversations.
    count = collections.Counter()
    label_conversations = collections.defaultdict(set)
    for row in train:
        for op in row["target_operations"]:
            if op.get("op") == "sign":
                count[op["lemma"]] += 1
                label_conversations[op["lemma"]].add(row["conversation_id"])
    vocabulary = {label for label, n in count.items()
                  if n >= MIN_LABEL_COUNT and len(label_conversations[label]) >= MIN_LABEL_CONVERSATIONS}

    def admit(row):
        ops = row["target_operations"]
        # Keep a narrowly defined, consistent lexical operation task.
        return (MIN_OPS <= len(ops) <= MAX_OPS
                and all(op.get("op") == "sign" and op.get("lemma") in vocabulary for op in ops)
                and bool(row.get("context")))

    train, dev = [r for r in train if admit(r)], [r for r in dev if admit(r)]
    if not train or not dev:
        raise ValueError("Frequent-label/context filter left an empty split")
    if {r["conversation_id"] for r in train} & {r["conversation_id"] for r in dev}:
        raise AssertionError("Conversation leakage")
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise AssertionError("Row leakage")
    train_inputs = {" ".join(re.findall(r"\w+", r["input_text"].casefold())) for r in train}
    dev_inputs = {" ".join(re.findall(r"\w+", r["input_text"].casefold())) for r in dev}
    if train_inputs & dev_inputs:
        raise AssertionError("Normalized model input leakage")
    actual_train_counts = collections.Counter(
        op["lemma"] for row in train for op in row["target_operations"]
    )
    return train, dev, vocabulary, {
        "source": str(SOURCE), "source_rows": len(rows), "source_conversations": len(conversations),
        "split_method": "deterministic SHA-256 conversation holdout; 20% of official training conversations reserved before label filtering",
        "train_conversations": len(conversations) - n_holdout, "heldout_conversations": n_holdout,
        "train_rows": len(train), "heldout_rows": len(dev), "label_count": len(vocabulary),
        "retained_train_conversations": len({r["conversation_id"] for r in train}),
        "retained_heldout_conversations": len({r["conversation_id"] for r in dev}),
        "shared_signers": len({r["signer_id"] for r in train} & {r["signer_id"] for r in dev}),
        "heldout_operations_missing_from_retained_training": sum(
            op["lemma"] not in actual_train_counts for row in dev for op in row["target_operations"]
        ),
        "minimum_train_label_frequency": MIN_LABEL_COUNT,
        "minimum_train_conversations_per_label": MIN_LABEL_CONVERSATIONS,
        "target_policy": "2-8 SIGN operations; support measured before filtering the training pool; no fingerspelling/nonlexical operations",
        "context_policy": "nonempty completed French dialogue context required; use exact prepared input_text",
        "linguistically_validated": False, "official_validation_or_test_used": False,
        "heldout_conversation_ids": sorted(holdout),
        "train_rows_sha256": hashlib.sha256("\n".join(r["content_sha256"] for r in train).encode()).hexdigest(),
        "heldout_rows_sha256": hashlib.sha256("\n".join(r["content_sha256"] for r in dev).encode()).hexdigest(),
    }


def seq(row):
    return ["SIGN:" + op["lemma"] for op in parse_plan(row["target_text"])]


def score(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Prediction and reference counts differ")
    references = [seq(r) for r in rows]
    edit = sum(distance(ref, pred) for ref, pred in zip(references, predictions))
    count = max(1, sum(map(len, references)))
    return {
        "examples": len(rows),
        "exact_match": sum(a == b for a, b in zip(references, predictions)) / max(1, len(rows)),
        "operation_wer": edit / count,
        "invalid_or_empty_rate": sum(not p for p in predictions) / max(1, len(rows)),
        "mean_reference_operations": float(np.mean([len(x) for x in references])) if references else 0,
        "mean_predicted_operations": float(np.mean([len(x) for x in predictions])) if predictions else 0,
    }


def features(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())
    feats = collections.Counter("w:" + x for x in tokens)
    padded = " " + " ".join(tokens) + " "
    feats.update("c" + str(n) + ":" + padded[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(padded)-n+1)))
    return feats


def retrieval(train, dev):
    train_features = [features(r["input_text"]) for r in train]
    vocab = sorted({key for row in train_features for key in row})
    index = {key: i for i, key in enumerate(vocab)}
    df = np.zeros(len(vocab), dtype=np.float32)
    for row in train_features:
        for key in row:
            df[index[key]] += 1
    idf = np.log((1 + len(train)) / (1 + df)) + 1

    def vectorize(rows):
        vectors = np.zeros((len(rows), len(vocab)), dtype=np.float32)
        for i, row in enumerate(rows):
            for key, n in row.items():
                if key in index:
                    vectors[i, index[key]] = (1 + math.log(n)) * idf[index[key]]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return vectors

    matrix = vectorize(train_features)
    sims = vectorize([features(r["input_text"]) for r in dev]) @ matrix.T
    ids = sims.argmax(axis=1)
    predictions = [seq(train[i]) for i in ids]
    return score(dev, predictions), ids, sims.max(axis=1).tolist(), predictions


def train_model(train, dev, out: Path, epochs: int):
    set_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE, revision=REVISION,
                                                 local_files_only=True, torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16,
                                            lora_alpha=32, lora_dropout=0.05,
                                            target_modules=["q", "k", "v", "o", "wi", "wo"])).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    encoded = []
    for row in train:
        source = tokenizer(row["input_text"], truncation=False)
        target = tokenizer(row["target_text"], truncation=False)["input_ids"]
        if len(source["input_ids"]) > 512 or len(target) > 256:
            raise ValueError("Prepared row exceeds configured length budget: " + row["id"])
        encoded.append((source, target))

    def batch(items):
        x = tokenizer.pad([a for a, _ in items], padding=True, return_tensors="pt").to("cuda")
        labels = torch.full((len(items), max(len(y) for _, y in items)), -100, device="cuda", dtype=torch.long)
        for j, (_, target) in enumerate(items):
            labels[j, :len(target)] = torch.tensor(target, device="cuda")
        return x, labels

    def generate(rows):
        model.eval(); texts=[]; caps=0
        for i in range(0, len(rows), 4):
            inputs = tokenizer([r["input_text"] for r in rows[i:i+4]], padding=True,
                               truncation=False, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                pred = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1, use_cache=True)
            texts.extend(tokenizer.batch_decode(pred, skip_special_tokens=True))
            caps += sum(tokenizer.eos_token_id not in y.tolist() for y in pred)
        parsed=[]; invalid=0
        for text in texts:
            try:
                ops = parse_plan(text)
                values = ["SIGN:" + op["lemma"] for op in ops]
                if not values or any(op["op"] != "sign" for op in ops):
                    raise ValueError("invalid operation")
            except (ValueError, KeyError, TypeError):
                values=[]; invalid += 1
            parsed.append(values)
        return texts, parsed, caps, invalid

    planned = math.ceil(len(train) / 8) * epochs
    history=[]; best=None; step=0; started=time.monotonic(); patience=5; stale=0
    dump(out / "TRAINING_MANIFEST.json", {
        "model": BASE, "revision": REVISION, "seed": SEED, "epochs_max": epochs,
        "train_rows": len(train), "heldout_rows": len(dev), "batch_size": 2,
        "gradient_accumulation": 4, "precision": "bfloat16", "target_modules": ["q", "k", "v", "o", "wi", "wo"],
        "rank": 16, "learning_rate": 2e-4, "planned_updates": planned,
        "selection": "lowest held-out operation WER; report this as development performance only",
        "official_validation_or_test_used": False, "linguistically_validated": False,
    })
    for epoch in range(epochs):
        order=list(range(len(train)));random.shuffle(order);model.train()
        losses = []
        for start in range(0, len(order), 8):
            group=order[start:start+8]; optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(group), 2):
                x, y = batch([encoded[i] for i in group[j:j+2]])
                with torch.autocast("cuda", dtype=torch.bfloat16): loss=model(**x, labels=y, use_cache=False).loss
                if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
                (loss * len(group[j:j+2]) / len(group)).backward(); losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
            lr=2e-4*min(1, (step+1)/max(1, int(planned*.05)))*max(.05, 1-step/planned)
            for param in optimizer.param_groups: param["lr"]=lr
            optimizer.step(); step+=1
        texts, preds, caps, invalid=generate(dev)
        metrics=score(dev, preds); record={"epoch":epoch+1,"step":step,"mean_train_loss":float(np.mean(losses)),
                                        "heldout":metrics,"invalid_plans":invalid,"generation_cap_hits":caps,
                                        "seconds":time.monotonic()-started}
        history.append(record);dump(out/"history.json",history);dump(out/"progress.json",record)
        print(json.dumps(record),flush=True)
        if best is None or metrics["operation_wer"] < best["heldout"]["operation_wer"]:
            best=record;stale=0
            model.save_pretrained(out/"best_adapter");tokenizer.save_pretrained(out/"tokenizer")
            dump(out/"best_predictions.json",[{"id":r["id"],"prediction":p,"reference":seq(r)} for r,p in zip(dev,preds)])
        else:
            stale+=1
        if stale >= patience: break
    dump(out/"RESULT.json",{"status":"complete","best":best,"history":history,
                             "elapsed_seconds":time.monotonic()-started,
                             "claim":"exploratory lexical projection; annotations unreviewed; no signing intelligibility claim"})
    return best, history


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".runtime/lsfb-frequent-context-v2"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--baselines-only", action="store_true")
    args=parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.output.exists(): raise ValueError("Output exists; select a fresh output directory")
    args.output.mkdir(parents=True)
    rows=load_candidates();train,dev,vocab,manifest=build_split(rows)
    manifest["dataset_sha256"]=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    dump(args.output/"DATA_MANIFEST.json",manifest)
    (args.output/"train.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in train))
    (args.output/"heldout.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in dev))
    baseline, ids, similarities, baseline_preds = retrieval(train,dev)
    dump(args.output/"RETRIEVAL_RESULT.json", {"method":"TF-IDF lexical/character n-gram nearest training example using full current+context input",
                                                 "metrics":baseline,"maximum_similarity_mean":float(np.mean(similarities)),
                                                 "predictions":[{"id":r["id"],"retrieved_train_id":train[i]["id"],
                                                                 "similarity":s,"prediction":p,"reference":seq(r)}
                                                                for r,i,s,p in zip(dev,ids,similarities,baseline_preds)]})
    current_train = [{**row, "input_text": row["french_text"]} for row in train]
    current_dev = [{**row, "input_text": row["french_text"]} for row in dev]
    current_baseline, _, _, _ = retrieval(current_train, current_dev)
    dump(args.output / "CURRENT_RETRIEVAL_RESULT.json", current_baseline)
    most_common_sequence, _ = collections.Counter(tuple(seq(row)) for row in train).most_common(1)[0]
    controls = {
        "empty": score(dev, [[] for _ in dev]),
        "most_common_training_sequence": score(dev, [list(most_common_sequence) for _ in dev]),
    }
    dump(args.output / "CONTROL_RESULTS.json", controls)
    print(json.dumps({"stage":"preflight","manifest":manifest,"retrieval":baseline},indent=2),flush=True)
    if args.baselines_only:
        return
    best,history=train_model(train,dev,args.output,args.epochs)
    summary={"task":"frequent short LSFB lexical operation sequences; explicit dialogue context",
             "manifest":manifest,"retrieval":baseline,"current_turn_retrieval":current_baseline,
             "constant_controls":controls,"model_best":best,
             "model_beats_retrieval_wer":best["heldout"]["operation_wer"] < baseline["operation_wer"],
             "model_beats_retrieval_exact":best["heldout"]["exact_match"] > baseline["exact_match"],
             "heldout_is_conversation_disjoint_development":True,"linguistically_validated":False,
             "human_review_required_before_translation_or_avatar_claim":True}
    dump(args.output/"SUMMARY.json",summary)
    dump(args.output/"progress.json",{"status":"complete",**summary})
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__": main()
