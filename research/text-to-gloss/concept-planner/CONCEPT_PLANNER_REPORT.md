# Pretrained French concept-planning experiment

This is **lexical concept assistance for a human editor, not French-to-LSF translation**. It does not generate approved gloss sentences or validate captured motion. Default literal lookup stays unchanged. The model is an optional editable draft only; every selection remains pending linguistic review.

## Frozen experiment

Before any generative inference, we wrote 24 French software test cases and expected lexical concepts, plus a 440-entry vocabulary snapshot. These are authored engineering expectations, not LSF ground truth, STK test examples or model-training records. Cases cover exact words, inflection, synonyms, multiple concepts, negation, questions, numbers, names and unknown meanings. No training or prompt examples were created from the failed 61-pair corpus.

The vocabulary comes from the lead's real runtime library: 439 unique, non-synthetic, rights-cleared entries plus the curated Bonjour entry. IDs are the actual lead IDs. Synthetic fixtures were excluded. Source metadata and SHA256 values are retained in concept_benchmark/; the lead files were read without repository rebasing or editing. The existing lead lookup implementation was snapshotted, then executed with real GlossEntry objects and fr_core_news_sm 3.8.0 through `/home/jmal/Projects/Active/Forum_Hackathon/.venv/bin/python`.

The requested Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` downloaded only about 79 MB of partial weights after more than four minutes. Its owned download was stopped. Before benchmark inference, the user authorized switching to the already cached [Unsloth Qwen3.5-9B GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), revision `3885219b6810b007914f3a7950a8d1b469d598a5`, `Qwen3.5-9B-UD-Q6_K_XL.gguf`, Apache 2.0. The [upstream Qwen model](https://huggingface.co/Qwen/Qwen3.5-9B) is also Apache 2.0. Cached bytes: 8,756,929,760; verified SHA256 `33b0050fb9c19abcf815647a78464dad959a06dadaecb0b96af798669f9074d4`. No new GGUF weights were downloaded. Ollama tags were inspected read-only; no Ollama inference, cloud endpoint, model mutation or existing-session interruption occurred.

Dedicated llama-server: localhost port 18195, context 8192, one slot, 20 GPU layers, six CPU threads, Flash Attention, thinking off, max 384 generated tokens, temperature 0, seed 42. All 440 concepts are included with compact integer indexes, mapped strictly back to actual runtime IDs. Preflight tokenizer input length including a 1000-character allowance was 3566; 256 chat-framing and 384 output tokens fit the context. Model load took about 16.15 seconds and used about 5207 MiB GPU memory before inference. No context/offload adjustment or prompt change occurred after evaluation began.

Exactly one model call per test case was made. Original policy/code/settings/results remain in runs/pretrained-concept-planner/: evaluation_lock.json, concept_planner_locked.py, predictions.jsonl and metrics.json. The literal baseline and a baseline with the same deterministic guards are separately recorded. There was no retry or prompt tuning on failed cases.

## Measured result — original locked code

| Method | Ordered concept-ID exact match | Micro concept F1 | Median latency |
|---|---|---|---|
| Pretrained model plus original guards | 13/24 (54.17%) | 0.7600 | 6.770 s |
| Existing literal lemmatized lookup | 12/24 (50.00%) | 0.7083 | 0.0031 s |
| Literal lookup plus the same original guards | 10/24 (41.67%) | 0.6364 | See note below |

Model mean latency was 7.795 s; first request 23.744 s. The literal first request was 0.934 s and its mean 0.0419 s, including its initial phrase-cache work. Baseline guard latency was not separately instrumented, so its recorded time is the underlying lookup time, not a full guarded end-to-end latency claim. Model evaluation took 187.114 s in total. Observed peak device memory was 5251 MiB; sampled dedicated server compute-process memory peaked at 5228 MiB. These are nvidia-smi observations, not PyTorch allocator statistics. Zero invalid external IDs and zero model-output rejections occurred.

### Concept-only component, separated from guards

Descriptive analysis of the **already-recorded raw model choices**, before advisory NER merging, on the 16 fully expressible cases with no required critical/unknown meanings: 11/16 (68.75%) versus literal lookup 6/16 (37.50%). This is an internal component diagnostic of the locked responses, not a benchmark score for the repaired runtime. The six synonym cases had only three correct concept selections in the original full pipeline. The model sometimes still missed synonyms, selected incorrect concepts, selected an interrogative concept despite the policy, or claimed a selected meaning was also unsupported. All failures are retained in component_analysis.json and predictions.jsonl.

### Guard checks, not semantic gains

The original shared guards flagged all six required critical cases and both required unknown terms were retained, but there were **11 false-positive critical warnings**. Warning-code precision was only 6/17 (35.29%); recall 6/6 on this tiny fixture does not establish reliable name detection. The independent desired Alice regression failed: the French model parsed Alice as NOUN/nsubj and emitted neither a PER entity nor a PROPN tag. The original assertion and its failure are preserved in guard_unit_pre_fix.txt.

A more immediate runtime bug was found: original guard postprocessing deleted valid model-selected Bonjour, Merci and Chien IDs after false-positive name detection. This also reduced the guarded literal baseline. Consequently, the narrow numeric usefulness flag in the original metrics is not a deployment approval. The result is modest engineering evidence, not a convincing translator, and does not justify replacing default lookup.

## Separately repaired runtime — no benchmark score

After the 24-case run finished, only runtime merging was repaired: advisory NER can no longer erase a strictly validated model-selected vocabulary ID. Detected names remain explicit unsupported meanings and review warnings alongside editable lexical anchors. Every output now additionally states `NAME_DETECTION_IS_ADVISORY_AND_INCOMPLETE`. Membership validation, the model, prompt, vocabulary and generation settings are unchanged.

The original desired Alice unit was not replaced with an easier name. It is retained verbatim as a **strict documented xfail**; if it unexpectedly passes, the suite requires review of that capability claim. Independent non-generative regression tests prove a false-positive name warning cannot erase a valid Bonjour ID and an entirely empty Alice result must retain the input as unsupported. This does not guarantee detection of names inside arbitrary sentences. The repaired runtime has **no assigned benchmark score**, and the 24 cases were not rerun. One separate `Le soleil.` CLI call is an operational JSON/interface smoke check, not a new accuracy test.

The local runtime manifest `research/concept_planner_runtime.json` records both original locked and current repaired code hashes, pinned model path/revision/hash, localhost URL, actual startup args and the explicit unscored runtime status. Its paths describe this machine, not a portability guarantee. The dedicated server remains alive for the lead's API/browser checks and is reused across requests; the CLI never starts a new server per sentence.

Actual CLI:

```bash
research/code/training/.venv-train/bin/python research/code/training/concept_planner.py \
  --text 'Le soleil.' \
  --vocabulary research/text-to-gloss/concept-planner/concept_benchmark/vocabulary.json \
  --model /home/jmal/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-UD-Q6_K_XL.gguf \
  --device cuda
```

The supplied vocabulary must be a JSON list of `{id,concept}` objects with unique IDs. JSON is the last and only stdout line. It includes concept_ids, unsupported, warnings, candidate=true, translation_validated=false, pending review, exact model revision/backend, vocabulary/prompt hashes, latency and retained raw output. Errors abstain with no IDs and explicit unsupported input. Only localhost HTTP and the recorded local model are accepted. French spaCy/model dependencies are available in the recorded own Python environment. Unknown/negated/question/name/number meanings require manual review; concept ordering is French source/semantic order, never an LSF grammatical claim. Default lookup and lead UI/motion code are untouched. Weights, environments and runtime server logs are not committed.
