# SignaCulture AI — Paper B, LSF dataset discovery, centralization & RTX 4060 fine-tuning

> **Purpose:** implementation and research instructions for a coding agent / development team.  
> **Scope:** French Sign Language used in **France** (`LSF`, not Swiss `LSF-CH`); French → candidate LSF gloss sequence → approved gloss/motion lookup → 3D avatar.  
> **Target machine:** NVIDIA GeForce RTX 4060 **Laptop, 8 GB VRAM** (verify with `nvidia-smi`; if yours differs, change the settings).  
> **Project:** SignaCulture AI — IA Éthique 360 hackathon, November 2026.  
> **State:** a **research and implementation plan**; no model has been trained and no dataset has been downloaded in this document.  
> **Companion document:** `SignaCulture_AI_Plan_Developpement.md` (four-team development plan).

---

## 0 — Instructions for the coding/research AI agent (READ FIRST)

**Your task is not to invent training data or claim that a sign dictionary is a parallel translation corpus.** Execute the work in this order:

1. **Inspect the supplied Paper B** (the user's `1-s2.0-S1877050926030693-main.pdf`, or the local copy specified by the user), the source links below, and the project's existing files before writing code. If the PDF is not in your workspace, read the reference and ask for it rather than claiming to have inspected it.
2. Run a **dataset discovery/audit** targeted at real French LSF (`LSF`, `fsl`). Find what each source *actually provides* (isolated glosses, French parallel sentence pairs, signer videos, HamNoSys/SiGML, pose sequences, usage rights). Verify current access and licenses from primary documentation.
3. Build a **central catalog** storing metadata, source citations, consent/license decisions, canonical IDs, splits and asset paths. **Do not bulk scrape websites or download videos without explicit permission.**
4. Only prepare training records when a trustworthy **French sentence ↔ genuine LSF gloss sequence** pair is present. Keep lexical signs and video landmarks in other tables.
5. Build the **deterministic baseline first:** phrase matcher/spaCy lemmatization → gloss candidate retrieval → coverage/unknown report.
6. If enough authorized, validated sentence pairs are available, fine-tune **`google/mt5-small` + LoRA** on the 8 GB 4060. Benchmark **`t5-base`** as the Paper B reproduction baseline, if feasible. Do not pretend the original paper's training data or weights are publicly available.
7. Evaluate on a deduplicated **held-out split grouped by source/recording/signer** where possible. Record unsupported glosses, semantic errors, exact match, gloss WER, BLEU and lexicon coverage. Request qualified LSF feedback before calling translations intelligible.
8. Hand generated canonical gloss IDs to the **Animics / 3D team**. That team—not the text-to-gloss model—extracts and retargets authorized sign motion.
9. Only then explore **audio prosody and emotion2vec+** as a separate, constrained expressive layer. It cannot overwrite LSF grammatical face/head/body markers.
10. Maintain a `RESEARCH_LOG.md`, `DATASET_CARD.md`, `MODEL_CARD.md`, reproducible configs, test results and a **clear list of what is not yet implemented**.

**Stop conditions:** if dataset rights are unclear, record metadata only and request access. If no aligned French–LSF gloss sentences exist locally, **do not fine-tune on fabricated examples**; continue with the lexicon lookup/animation MVP and an approved data-collection plan.

---

## 1 — Paper B: exact scientific reference and what it actually did

**Citation:** Manuel Muanji et al., *Signing Avatar Synthesis via a Hybrid T5-GAN Approach: Automatic Speech-to-French Sign Language Translation*, *Procedia Computer Science*, volume 290 (2026), pp. 578–584. DOI: https://doi.org/10.1016/j.procs.2026.08.433. The supplied PDF is `1-s2.0-S1877050926030693-main.pdf`. The publication's PDF is marked **CC BY-NC-ND 4.0**; its license does not automatically grant access to underlying datasets, code, signer videos or model weights.

### 1.1 Architecture reported by the authors (paper pp. 2–4)

```text
Speech ──(Google ASR)──> French text
French text ──(fine-tuned T5-base)──> candidate LSF gloss sequence
Gloss candidates ──(47 expert rules + RapidFuzz ≥ 85% + SOV-style rules)──> corrected gloss labels
Canonical gloss labels ──(CSV lookup)──> stored MediaPipe movement sequences
Adjacent sign movements ──(conditional GAN)──> 10-frame transition motion
Stored signs + transitions ──(Pygame)──> stylized animated output
```

**Reported corpus (paper p. 3):** 319 isolated signs; 2,500 French–gloss sentence pairs; 379 full sentence sequences; 379 real transition pairs; 1,000 synthetic transition pairs. The paper says videos came from YouTube and an Edouard Lopez GitHub resource, and landmarks were extracted with MediaPipe Holistic. **These counts are from the authors, not a public dataset you can simply fetch.** The article says code and data are available **upon request** (p. 6).

**Reported T5 settings (paper p. 3):** `t5-base` (~220M parameters), AdamW, LR `2e-5`, batch size `4`, `20` epochs, tokenizer extended by `319` gloss labels.

**Reported post-correction (paper p. 4):** 47 expert mapping rules; RapidFuzz threshold `85%`; rule-driven SOV reordering. **"100% animability" means every accepted gloss can be mapped to a known motion entry, not 100% linguistic accuracy.** Automatic fuzzy replacement may change meaning.

**Reported conditional GAN (paper p. 4):** original poses contain `75 × 3 = 225` coordinate values per frame, but transition generation uses **5 selected landmarks** (nose, wrists, middle fingertips) reduced to 10 features, generating **10 frames at 30 fps (~0.33 s)**. Adversarial + L1 objective uses `λ = 1000`. Do not describe this as full 3D hand/face synthesis.

**Reported results (paper pp. 5–6):** BLEU `0.3532 → 0.4347`, WER `0.2907 → 0.2364`, exact gloss sequence match `21.9% → 30%` after post-correction. Motion RMSE `0.146354` (normalized coordinates). Two LSF interpreters rated 20 generated sentences (reported intelligibility `4.3/5`). The assessment is small and does **not** certify general-purpose translation quality.

**Authors' stated limitations (p. 6):** lexicon of 319 signs; missing grammatical non-manual features; 2D stylized avatar; simplified hand landmark representation; single-signer data; occasional occlusion tracking errors. The latency text and table also report inconsistent internal-timing values; remeasure locally rather than quote an unverified real-time claim.

### 1.2 What we will copy conceptually vs. what we will change

| Part | Paper B | SignaCulture implementation |
|---|---|---|
| French → gloss | `t5-base` | **spaCy baseline** first; **mT5-small LoRA** as French-aware experimental model; T5-base benchmark |
| Training source | Authors' 2,500 pairs, **not yet obtained** | Independently sourced and licensed **genuine aligned pairs**, verified by LSF reviewers |
| Canonicalization | 47 rules + 85% fuzzy auto-correction | Deterministic canonical IDs; fuzzy **suggestions**, not silent semantic replacements |
| Motion data | CSV MediaPipe landmarks | Source/consent-linked pose assets, Animics-compatible motion manifests, common rig |
| Transitions | 5-landmark GAN | Direct concatenation & easing baseline first; optional learned transitions later |
| Avatar | Pygame 2D-like result | Animics authoring + Performs/compatible rigged 3D playback |
| Expression | Missing | Linguistic facial controls + experimentally constrained audio-driven expression |
| Evaluation | BLEU/WER + small human panel | Held-out test, signer-level splits, motion QA, LSF reviewer + Deaf signer comprehension tests |

**Critical principle:** there are **two different models**: (A) text-to-gloss translation (trained on sentence pairs), and optionally later (B) motion transition generation (trained on temporal motion). The gloss lexicon and pose database are **retrieval resources**, not automatically model-training examples.

---

## 2 — Dataset discovery: exact source list, search procedure and license gate

### 2.1 Priority sources and the questions to answer

| Resource | URL | What is currently documented | Role in the system | Licensing/access notes |
|---|---|---|---|---|
| **Dicta-Sign French Concepts** | https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_fre.html | Browsable French concepts linked to sign entries, gloss/form information and video | **Lexicon and candidate motion-source catalog**; **not** a parallel sentence training set | Corpus portal has an **individual research licensing framework**; inspect each asset's permissions before use |
| **Dicta-Sign-LSF-v2** | https://hdl.handle.net/11403/dicta-sign-lsf-v2/v1 | ~11 hours continuous LSF, annotations and French translations | Linguistic research, possibly paired alignments *after inspection* | **CC BY-NC-SA 4.0**, some files require registration; noncommercial/reuse conditions apply |
| **Rosetta-LSF** | https://aclanthology.org/2022.lrec-1.529/ ; https://hdl.handle.net/11403/rosetta-lsf/v3 | French headlines, avatar LSF translations, gloss annotations, AZee representations | Potential **genuine aligned sentence examples**; linguistic structure beyond glosses | Confirm each release's license and accessibility; don't assume all videos are downloadable |
| **STKAutogramm (LSF)** | https://github.com/UniversalDependencies/UD_French_Sign_Language-STKAutogramm | **108 sentences / 615 tokens**; `text_fr`, tokenized LSF transcription, `sent_id`, `source_file` | Small **paired-text linguistic dataset** and evaluation fixture | **CC BY-SA 4.0** for the treebank; linked videos may have separate terms |
| **Animics** | https://github.com/upf-gti/animics | Script/Keyframe tools, video MediaPipe extraction, BVH/BVHE/GLB exports | **Motion extraction/retargeting tool**, not a French → gloss dataset | MIT software; bundled/referenced sign data may have different terms; 6,484 examples are **LSF-CH** (Swiss), not France LSF |
| **Performs** | https://github.com/upf-gti/performs | Compatible avatar animation playback | 3D visualizer | Verify tool and asset licenses independently |
| **Hybrid T5-GAN paper** | https://doi.org/10.1016/j.procs.2026.08.433 | Research pipeline and reported results | Design/reference baseline | **Code and data upon request**; not a verified public model checkpoint |

**Additional research:** Inspect the 2020 Dicta-Sign-LSF-v2 paper at https://aclanthology.org/2020.lrec-1.740/ and compare its annotation format with the dataset release. Do not conflate 2020 paper, original Dicta-Sign portal, and LSF-v2 release.

### 2.2 Search queries for the research agent

Use web search, institutional sites (Ortolang/Universität Hamburg), GitHub and Hugging Face search. Preserve the returned **original page URLs** in `research/sources.csv`.

```text
"French Sign Language" "French" gloss parallel corpus license
"Langue des signes française" gloss traduction corpus aligné Ortolang
"Rosetta-LSF" gloss text_fr AZee download license
site:github.com "French Sign Language" "text_fr" gloss
site:universaldependencies.org/treebanks/fsl_stkautogramm "text_fr"
site:sign-lang.uni-hamburg.de/dicta-sign/portal/concepts "bonjour"
site:ortolang.fr "Dicta-Sign-LSF-v2" terms access
LSF 3D avatar motion dataset HamNoSys SiGML France license
LSF France sentence gloss text pair training text-to-gloss
LSF video gloss timestamps ELAN annotation licensing
```

For **each** discovery, record and verify:

- Identity, authors, stable URL, date/version, primary paper, publication/repository commit.
- Language **and regional variety**: `LSF_FR`, `LSF_CH`, `LSFB`, `LST` are **not interchangeable**.
- Available modality: `(french_text, sign_gloss_sequence)`, sign lexicon, video, timestamps, landmarks, HamNoSys, AZee, SiGML.
- Alignment unit: concept, word/sign, phrase, continuous discourse; exact sentence pairing or merely related subtitles.
- Gloss annotation conventions: pointers, non-lexical signs, fingerspelling, spatial roles, omissions.
- Count of **valid trainable examples**, not count of videos or words; check data schema and manually sample 5 examples.
- Consent, copyright, biometrics, privacy, derivatives, training, redistribution, commercial rights, approval required.
- Availability: open, account required, application required, link dead, paper-only, data-on-request.
- Reliability issues: duplicates, near-duplicates, different signers, uncertain annotations, videos with non-LSF content.

### 2.3 Source approval state machine

```text
DISCOVERED → METADATA_ONLY → RIGHTS_VERIFIED → AUTHORIZED_FOR_INGEST
                                 │                         │
                                 ▼                         ▼
                            BLOCKED / PENDING      IMPORTED → VALIDATED
                                                         │
                                                         ▼
                                               TRAIN_APPROVED or MOTION_APPROVED
```

`RIGHTS_VERIFIED` **does not imply** `TRAIN_APPROVED`: check permission for the **specific use**, including derived keypoints, model weights, publication and commercial application. Annotate individual records when a corpus has mixed restrictions.

Prohibit automatic scraping, video downloads, stream ripping, bypassing authentication and redistribution before rights verification. Metadata may be indexed for research when permitted, with source URL and restrictions.

### 2.4 A critical data distinction

```text
(1) French concept ↔ LSF sign gloss ↔ sign video
    → great for lexical lookup and (if permitted) motion extraction
    → NOT enough to train sentence-level text-to-gloss

(2) French *sentence* ↔ LSF gloss *sequence* (aligned & approved)
    → appropriate supervised text-to-gloss training record

(3) Signed recording ↔ time-aligned landmarks / 3D rotations
    → appropriate motion / transition research record

(4) Audio ↔ transcript / emotion/prosodic features
    → separate expressive-control research record
```

Do **not** convert isolated glosses into made-up French–LSF sentences. Do not train on LSF-CH without explicitly labeling it as a cross-variety experiment and evaluating against genuine France-LSF data.

---

## 3 — Centralized architecture: one source of truth

### 3.1 Design: SQLite + immutable data store + versioned exports

For the hackathon, prefer **SQLite** (metadata) and a well-structured directory tree (assets). Export training pairs as **JSONL**; optionally Parquet for large datasets. There is no need for an expensive remote database service.

```text
signaculture-ai/
├── docs/
│   ├── SignaCulture_AI_Plan_Developpement.md
│   ├── PAPER_B_REPRODUCTION_NOTES.md
│   ├── DATASET_CARD.md
│   ├── MODEL_CARD.md
│   └── DATA_LICENSE_MATRIX.md
├── references/
│   ├── README.md                # DOI / citation / original file locations
│   └── paper_B.pdf              # optional local copy with compatible usage rights
├── research/
│   ├── RESEARCH_LOG.md
│   └── sources.csv              # source registry, research provenance
├── data/
│   ├── catalog.sqlite           # single source of truth, not public if sensitive
│   ├── raw_private/             # excluded from git: approved videos, consent records
│   ├── processed_private/       # pose sequences and derived motion, if restricted
│   ├── public_examples/         # only explicitly redistributable samples
│   ├── manifests/               # sign and motion registries with provenance
│   └── text_to_gloss/
│       └── v001/
│           ├── train.jsonl
│           ├── val.jsonl
│           ├── test.jsonl
│           └── dataset_manifest.json
├── scripts/
│   ├── discover_sources.py
│   ├── ingest_lexicon.py
│   ├── import_aligned_corpus.py
│   ├── validate_catalog.py
│   ├── build_training_splits.py
│   ├── train_text_to_gloss.py
│   ├── evaluate_text_to_gloss.py
│   └── export_motion_registry.py
├── models/
│   └── text_to_gloss_lora/      # adapters and tokenizer (restricted as needed)
├── experiments/
│   ├── motion_extraction/
│   └── audio_expression/
├── backend/
├── frontend/
└── .gitignore
```

### 3.2 Minimal normalized tables

**`sources`**: `source_id (PK)`, `name`, `url`, `version`, `authors`, `language_variant`, `license_id`, `access_level`, `terms_url`, `derivative_policy`, `train_permission`, `redistribute_permission`, `commercial_permission`, `verification_date`, `reviewer`, `notes`.

**`concepts`**: `concept_id`, `label_fr`, `definition_fr`, `source_id`, `source_concept_id`.

**`glosses`**: `gloss_id`, `canonical_label`, `language_variant`, `meaning`, `annotation_scheme`, `status`, `linguist_review_id`. **A canonical gloss is a database label, not automatically a unique sign/meaning.**

**`lexical_variants`**: `variant_id`, `gloss_id`, `signer_pseudonym`, `regional_variant`, `context`, `source_id`, `source_asset_id`, `status`.

**`assets`**: `asset_id`, `variant_id`, `source_id`, `kind` (`video`, `mediapipe`, `bvh`, `bvhe`, `glb`, `sigml`), `local_path`, `sha256`, `fps`, `frame_count`, `rig_id`, `rights_status`, `review_status`, `derivation_id`.

**`sentence_pairs`**: `pair_id`, `french_text`, `gloss_sequence`, `language_variant`, `source_id`, `source_segment_id`, `speaker_group`, `annotation_scheme`, `linguistic_review`, `rights_status`, `quality_flags`, `split_group`, `created_at`.

**`reviews`**: `review_id`, `asset_or_pair_id`, `reviewer_role`, `decision`, `timestamp`, `notes`, `version`.

**`derivations`**: `derived_id`, `input_asset_ids`, `tool`, `tool_commit`, `parameters`, `output_asset_id`, `timestamp`.

**`model_runs`**: `run_id`, `model_base`, `adapter_config`, `dataset_version`, `dataset_sha256`, `split_seed`, `hyperparameters_json`, `cuda_device`, `max_vram_mb`, `metrics_json`, `checkpoint_uri`.

Never store full signers' names or private signed consent forms in public tables. Keep sensitive material under separate access controls.

### 3.3 Minimal training record — for **sentence pairs only**

```json
{
  "pair_id": "EXAMPLE_001",
  "source_id": "APPROVED_SENTENCE_SOURCE",
  "language_variant": "LSF_FR",
  "french_text": "Phrase française réellement alignée et vérifiée",
  "gloss_sequence": "GLOSS_1 GLOSS_2",
  "annotation_scheme": "SOURCE_SCHEME_V1",
  "source_segment_id": "SRC_SEG_001",
  "split_group": "DOC_001_SIGNER_P001",
  "rights_status": "TRAIN_APPROVED",
  "linguistic_review": "APPROVED"
}
```

**All values above are PLACEHOLDERS, NOT real LSF training examples.** Do not train on this record. `GLOSS_1 GLOSS_2` is dummy text.

### 3.4 Minimal lexical/motion record — **not a sentence pair**

```json
{
  "gloss_id": "DICTASIGN_FR_SOURCEENTRY_001",
  "canonical_label": "SOURCE_GLOSS_EXACTLY_AS_ANNOTATED",
  "concept_fr": "SOURCE_CONCEPT",
  "language_variant": "LSF_FR",
  "source_url": "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_fre.html",
  "source_version": "verify",
  "license_status": "PERMISSION_PENDING",
  "motion_asset_id": null,
  "motion_status": "NOT_EXTRACTED",
  "review_status": "NOT_REVIEWED"
}
```

### 3.5 Example source registry CSV header

```csv
source_id,name,url,version,language_variant,asset_type,access,license,train_permission,motion_permission,redistribution_permission,verified_on,notes
```

**Rule:** `train_permission=TRUE` only with documented permission for the intended training use. Unknown is `PENDING`, not `TRUE`.

### 3.6 Data versioning and audit

- Compute SHA-256 per input file and per train/val/test JSONL.
- Keep a `dataset_manifest.json` recording IDs, counts, exclusions, licenses, source revisions, checksums, split algorithm and random seed.
- Version any annotation edits; preserve source glosses verbatim, and store normalized glosses in a separate field.
- Split before any augmentation. Treat near-duplicates and related clips as belonging to one split group.
- Maintain data-lineage from gloss prediction to signer-derived motion. Unknown entries remain unknown.

---

## 4 — Dataset ETL workflow: concrete tasks for Team A

**Discovery (metadata only):** inspect the official portal pages and institutional archives; identify stable concept URLs/IDs and the exact gloss annotation field. Produce a source table and 5-entry sample inventory. Use an existing official API or download facility **only if documented and authorized**. Do not assume HTML layout fields are stable.

**Import:** for each explicitly permitted dataset:

1. Save the exact original license/usage documentation and URL/date.
2. Inspect release format (ELAN `.eaf`, `.conllu`, CSV, JSON, video, notation).
3. Build **one importer per source**, not a universal parser that drops language-specific annotations.
4. Preserve original source IDs and representation scheme; then map to local IDs.
5. Extract candidate pairs only where a **genuine aligned French translation and sign annotation** exist. For UD/STKAutogramm, check `# text_fr =` and the tokenized signed utterance; retain pointers and specialized gloss symbols rather than stripping them indiscriminately.
6. Run deduplication, quality checks, manual sampling and qualified linguistic review.
7. Mark approved pairs as eligible for training; leave research-only or unclear records in the catalog but **excluded** from exports.
8. For video-to-motion use, hand **authorized video/clip references** to Team B and attach the extracted motion asset to the same source lineage.

### 4.1 Recommended validation assertions

```python
REQUIRED = {
    "pair_id", "source_id", "language_variant",
    "french_text", "gloss_sequence", "split_group",
    "rights_status", "linguistic_review"
}

def eligible_for_training(row):
    assert REQUIRED.issubset(row.keys())
    return (
        row["language_variant"] == "LSF_FR"
        and row["rights_status"] == "TRAIN_APPROVED"
        and row["linguistic_review"] == "APPROVED"
        and bool(row["french_text"].strip())
        and bool(row["gloss_sequence"].strip())
    )
```

Also assert no overlap between split groups across train/val/test, no duplicate French–gloss pairs across splits, and no unknown source IDs. For small datasets use explicit, documented grouping rather than random sample splitting.

### 4.2 Test corpus selection strategy

- **Train**: approved sentences from a subset of sources and/or signer groups.
- **Validation**: held-out groups for hyperparameter choice.
- **Test**: untouched groups for final reporting. Prefer holding out source videos and signers when realistic.
- If so few examples exist that a reliable test set is impossible, report an **engineering smoke test**, not a meaningful translation benchmark.
- Keep a separate challenge set for metaphor, pronouns, negation, questions, cultural terminology and unseen glossary entries, all reviewed by an LSF specialist.

---

## 5 — RTX 4060 (8 GB): realistic training strategy for **French → gloss**

### 5.1 Which model to train?

| Candidate | Purpose | Recommendation |
|---|---|---|
| **spaCy + lexical/phrase lookup** | No-training baseline | **Build first**, always keep as fallback |
| **`google/mt5-small`** | Multilingual seq2seq, pretrained on French-containing multilingual text | **Recommended first LoRA experiment** |
| **`t5-base`** | Direct architecture match to Paper B | **Scientific reproduction/ablation**; English-centric pretraining may affect French results |
| **Large instruction LLM** | Optional semantic candidate generation | **Not necessary** for the first dataset/motion proof |
| **Conditional GAN** | Movement transitions, not text-to-gloss | **Separate later experiment**, requiring temporal motion pairs |

**Why LoRA?** It updates a small number of trainable matrices while freezing base model weights; this lowers optimizer/gradient memory compared with full fine-tuning. With 8 GB VRAM, prefer LoRA to full `t5-base` training. **Do not assume QLoRA/4-bit is automatically faster or compatible with all seq2seq models**; first test normal 16-bit LoRA.

**Important:** if all you have is Dicta-Sign's isolated sign dictionary, **STOP**. You do not yet have the paired training data needed for this module. Build the glossary and motion prototype instead. On a tiny corpus, T5 may memorize training pairs; do not report general translation ability from train accuracy.

### 5.2 Environment and hardware checks (Linux / NVIDIA)

```bash
nvidia-smi
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install CUDA-compatible PyTorch using the CURRENT selector at:
# https://pytorch.org/get-started/locally/
# THEN install the Python dependencies:
pip install 'transformers>=4.51,<5' 'peft>=0.17,<0.22' \
  'accelerate>=1.0' 'datasets>=3.0' sentencepiece \
  rapidfuzz spacy pandas scikit-learn

python - <<'PY'
import torch
print('torch', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    print('bf16 supported:', torch.cuda.is_bf16_supported())
PY

pip freeze > requirements.lock.txt
```

**Windows:** native CUDA PyTorch or WSL2 may work; adapt `venv` activation and filesystem paths accordingly. Do not run this training in the browser/Animics runtime. It is a standalone Python job.

### 5.3 Initial safe hyperparameters for 8 GB

| Parameter | Start | Reason |
|---|---:|---|
| Base | `google/mt5-small` | Multilingual French support |
| LoRA rank `r` | `8` | Low trainable parameter count |
| LoRA alpha | `16` | Common 2× rank baseline |
| LoRA dropout | `0.05` | Regularization |
| Target modules | `q`, `v` | Standard T5/mT5 attention projection names; verify actual module names |
| Per-device batch | `1` | 8 GB-friendly starting point |
| Gradient accumulation | `16` | Effective batch size 16 before any distributed scaling |
| Max source tokens | `128` | Short controlled French sentences |
| Max target tokens | `64` | Short gloss outputs |
| Epochs | `3–8` with validation/early stopping | Paper B uses 20, but do not copy blindly |
| Learning rate | `2e-4` LoRA starting value | Tune on validation; Paper B full fine-tune used `2e-5` |
| Precision | `bf16` if supported, otherwise `fp16` | Reduced VRAM |
| Gradient checkpointing | `True` | Reduce activation memory at compute cost |
| Decoding | deterministic greedy/beam in evaluation | Reproducible comparisons |

Actual VRAM and runtime depend on driver, framework version, tokenizer lengths, model choice and other GPU workloads. **No measured throughput or success is claimed here.**

### 5.4 Minimum runnable training script

Save the following as `scripts/train_text_to_gloss.py`. It expects **real, rights-cleared, linguistically approved** JSONL files created by Team A. It **will not generate a training corpus**.

```python
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("pair_id", "french_text", "gloss_sequence",
                        "language_variant", "rights_status", "linguistic_review"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing {key}")
            if row["language_variant"] != "LSF_FR":
                raise ValueError(f"{path}:{line_no}: non-France variety")
            if row["rights_status"] != "TRAIN_APPROVED":
                raise ValueError(f"{path}:{line_no}: training not authorized")
            if row["linguistic_review"] != "APPROVED":
                raise ValueError(f"{path}:{line_no}: linguistic review missing")
            if not row["french_text"].strip() or not row["gloss_sequence"].strip():
                raise ValueError(f"{path}:{line_no}: empty input or output")
            rows.append(row)
    if not rows:
        raise ValueError(f"No approved training examples in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/text_to_gloss/v001")
    parser.add_argument("--base", default="google/mt5-small")
    parser.add_argument("--output", default="models/text_to_gloss_lora")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; check NVIDIA driver/PyTorch setup")

    set_seed(42)
    train_rows = read_jsonl(Path(args.data) / "train.jsonl")
    val_rows = read_jsonl(Path(args.data) / "val.jsonl")
    train_ids = {r["pair_id"] for r in train_rows}
    val_ids = {r["pair_id"] for r in val_rows}
    if train_ids & val_ids:
        raise ValueError("Train/validation IDs overlap")
    train_groups = {r.get("split_group", r["pair_id"]) for r in train_rows}
    val_groups = {r.get("split_group", r["pair_id"]) for r in val_rows}
    if train_groups & val_groups:
        raise ValueError("Train/validation split groups overlap")
    print(f"Train={len(train_rows)} Val={len(val_rows)}; verified rights + variety")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    base_model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q", "v"],
        bias="none",
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    def tokenize(batch):
        prompts = ["Traduire du français vers des glosses LSF : " + s
                   for s in batch["french_text"]]
        src = tokenizer(prompts, max_length=128, truncation=True)
        tgt = tokenizer(text_target=batch["gloss_sequence"],
                        max_length=64, truncation=True)
        src["labels"] = tgt["input_ids"]
        return src

    train_data = Dataset.from_list(train_rows).map(
        tokenize, batched=True, remove_columns=list(train_rows[0].keys()))
    val_data = Dataset.from_list(val_rows).map(
        tokenize, batched=True, remove_columns=list(val_rows[0].keys()))

    bf16 = bool(torch.cuda.is_bf16_supported())
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        bf16=bf16,
        fp16=not bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        predict_with_generate=False,
        optim="adamw_torch",
        seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Adapter + tokenizer saved. Evaluate on untouched test split separately.")


if __name__ == "__main__":
    main()
```

**Run:**

```bash
# Inspect generated training JSONL before proceeding; do NOT use synthetic examples.
python scripts/train_text_to_gloss.py \
  --data data/text_to_gloss/v001 \
  --base google/mt5-small \
  --output models/text_to_gloss_lora

# For a direct architectural comparison with Paper B (only if GPU memory allows):
python scripts/train_text_to_gloss.py \
  --data data/text_to_gloss/v001 \
  --base t5-base \
  --output models/text_to_gloss_t5base_lora
```

**Important reproducibility checks:**

- Verify package versions, the actual model module names and GPU memory before a long run. This is a script template, not an already executed or benchmarked training run.
- The original Paper B **extended the T5 tokenizer** for 319 labels. The starter script deliberately **does not**, so subword label tokenization remains enabled. If extending the vocabulary, carefully initialize/train/store new embeddings and preserve the identical tokenizer in deployment.
- `t5-base` is used to compare architecture, not to replicate the authors' unshared corpus, train/validation split or exact results.
- If low-memory training fails: reduce `max_source_length`/`max_target_length`, stop other GPU jobs, verify dynamic padding, lower LoRA rank, run `google/mt5-small` only or use `t5-small` **as a reduced smoke test**. Don't silently claim a completed training run.

### 5.5 OOM troubleshooting sequence

1. `nvidia-smi` to check current use; close browser GPU-heavy tabs, games, Animics rendering and other models while training.
2. Confirm batch size = `1`, dynamic padding and gradient checkpointing enabled.
3. Reduce input length to 96, target length to 48 after checking that examples aren't truncated.
4. Reduce LoRA rank `8 → 4`; reduce accumulation only if training time is a concern (it does not directly lower per-step activation memory).
5. Try smaller model for a proof of running training, then return to the main experiment.
6. Record OOM configuration, VRAM and traceback in `experiments/training_log.md`.
7. Do not start model training until approved parallel pairs exist. Finishing the supervised-data audit may matter more than GPU tuning.

---

## 6 — Evaluation: evidence of real improvement

### 6.1 Baselines to compare

- **B0 — lexical baseline:** spaCy normalization / manually approved phrase matching / unknown handling.
- **B1 — Paper B-inspired canonicalization:** add 47-like expert normalization rules **only if verified for our lexicon**; return fuzzy match **suggestions**.
- **M1 — mT5-small LoRA:** fine-tune on authentic pairs; compare against B0 on the same test set.
- **M2 — T5-base LoRA:** architecture-controlled comparison, if feasible.
- **M3 — optional full fine-tune:** only if memory and sufficient data permit; not part of hackathon critical path.

### 6.2 Metrics

| Metric | What it tells us | Warning |
|---|---|---|
| Gloss exact match | Whole predicted sequence equals one reference | Harsh if valid alternative glossing exists |
| Gloss WER | Token insertions/deletions/substitutions | Depends on consistent annotation scheme |
| BLEU | Surface overlap with reference gloss sequence | Not equivalent to LSF grammar/intelligibility |
| OOV gloss rate | Portion of predicted labels outside approved motion registry | Low OOV can be achieved by wrong substitutions |
| Candidate coverage | Fraction of requested items resolvable to permitted assets | Coverage ≠ semantic correctness |
| Error taxonomy | Ambiguity, sign choice, word order, negation, reference, non-manual marker | Requires expert interpretation |
| Hardware | Peak CUDA VRAM, runtime, epochs, errors | Must be **measured on RTX 4060**, never invented |

Keep **test data frozen**. Log per-example prediction and expected gloss text, source, model version, normalization step, whether a proposed fuzzy match was used, and review outcome.

### 6.3 Minimum evaluation gates

- All training records have positive train permission and `LSF_FR` labels; no leaked private files.
- No sentence/video/signer group overlap between train and test where grouping data exists.
- Evaluation beats the deterministic baseline on **real held-out data** or clearly states that it does not.
- No model-generated sequence is treated as linguistically validated unless approved.
- For final output, use **Deaf fluent LSF user comprehension** and qualified LSF linguistic assessment; not BLEU alone.

---

## 7 — Connection with Animics and the motion database

**Animics does not convert French into glosses.** It is the motion-authoring and 3D animation tool. Team A/C produce a canonical gloss/motion request; Team B uses **Animics Keyframe Mode** for authorized video-to-motion extraction, and/or investigates its **Script Mode** for phonetic/SiGML/BML animation. The README says the modes are distinct; do not claim to seamlessly mix their internal modes without engineering an export/merge interface.

```text
French text
   │
   ├── baseline or fine-tuned text-to-gloss → gloss candidate IDs
   │                                             │
   │                                      human/grammar review
   │                                             │
   │                                     rights-checked lexicon
   │                                             │
   │                                      canonical motion IDs
   │                                             │
   └── optional speech audio                       │
             │                                     │
        emotion2vec + prosody                     │
             │                                     │
      constrained expressive parameters            │
             └──────────────┬──────────────────────┘
                            ▼
               motion composer + common 3D rig
                            ▼
                    Performs/3D preview
                            ▼
                  reviewer / publication gate
```

The Dicta-Sign portal's isolated concepts can support gloss/motion **lookup**, subject to permissions. They cannot independently train the sentence-level neural translator, and a generated gloss sequence cannot guarantee grammatical LSF. Disallow unapproved publication even if an animation is technically playable.

### Motion handoff record

```json
{
  "gloss_id": "LSF_FR_CANONICAL_GLOSS_ID",
  "motion_id": "ANIM_APPROVED_ASSET_001",
  "language_variant": "LSF_FR",
  "source_id": "AUTHORIZED_SOURCE_001",
  "rig_id": "SIGNACULTURE_RIG_V1",
  "format": "GLB",
  "file": "data/processed_private/motions/asset_001.glb",
  "status": "PREVIEW_ONLY",
  "quality_review": "PENDING",
  "linguistic_review": "PENDING",
  "rights_status": "AUTHORIZED_FOR_INTERNAL_RESEARCH"
}
```

All sample IDs and paths are illustrative; the record is not evidence of an existing animation.

---

## 8 — Four-team ownership, prioritized tickets and final deliverables

| Owner | Ticket | Priority | Output and acceptance |
|---|---|---|---|
| **Team A: Data** | `A-01` Source inventory + license audit | **P0** | Verified sources.csv with per-asset permissions |
| A | `A-02` Schema + SQLite catalog | **P0** | Tables, migrations, tests; reproducible export |
| A | `A-03` Import approved dictionary entries | **P0** | At least one verified lexical entry with real source ID; **no license violations** |
| A | `A-04` Inspect genuine parallel sentence resources | **P1** | Data counts and eligibility report; no invented pairs |
| A | `A-05` Build train/val/test JSONL | **P1, conditional** | Only when approved paired data exists; leakage tests pass |
| **Team B: 3D** | `B-01` Run Animics and play one permitted sample | **P0** | Recorded 3D preview |
| B | `B-02` Extract/retarget authorized sign video | **P1** | Rigged motion asset with provenance |
| B | `B-03` Connect gloss ID → GLB lookup | **P0** | API loads actual asset; missing errors explicit |
| B | `B-04` Transition benchmark | **P2** | Direct concat vs. eased transition and quality check |
| **Team C: NLP** | `C-01` French phrase/lemma baseline | **P0** | Lookup and unknown-gloss detection with tests |
| C | `C-02` Data validation + split builder | **P1** | Reproducible, verified JSONL only |
| C | `C-03` Fine-tune mT5-small LoRA on 4060 | **P1, conditional** | Adapter + loss logs + measured VRAM + test predictions |
| C | `C-04` T5-base comparison | **P2** | Same splits, metrics and reported limitations |
| **Team D: Audio** | `D-01` emotion2vec+ experiment | **P1** | Timestamped scores on authorized speech samples |
| D | `D-02` Pitch/energy/pause extraction | **P1** | Aligned acoustic feature JSON |
| D | `D-03` Constrained avatar expression A/B | **P2** | Reviewer-evaluated non-manual feature changes |
| **Lead** | `INT-01` API + versioned data contracts | **P0** | Integration test text → glossary → real avatar asset |
| Lead | `INT-02` GitHub docs/demo/reproducibility | **P0** | French README, first demo, ethical limitations |
| Lead | `INT-03` Linguistic evaluation & stakeholder feedback | **P1** | Signed-off findings, uncertainty reported |

### Immediate sequence (do not block on fine-tuning)

1. Run `nvidia-smi`, inspect local repo; confirm 8 GB VRAM and Python/CUDA setup.
2. Team A catalogs Dicta-Sign concepts and legal conditions (metadata only if permissions pending).
3. Team B gets **one rights-cleared** real or manually authored signing asset playing in Animics/Performs.
4. Team C builds `French input → approved gloss ID → playable motion ID` with unknown handling.
5. Team D runs a short audio feature prototype independently.
6. Integrate a real end-to-end vertical slice and record evidence.
7. Audit Rosetta-LSF/STKAutogramm for real approved paired sentences.
8. **Only if valid sentence pairs are available**, train mT5-small LoRA on 4060 and compare to the baseline.
9. Add more words and validated sentences, then evaluate expressive control and motion transition improvements.

---

## 9 — Coding-agent deliverable checklist

At completion, the agent must report in `IMPLEMENTATION_STATUS.md`:

- [ ] Paper B reviewed; precise reproduced claims separated from proposed improvements.
- [ ] Current dataset source links and rights verified; blocked sources excluded from ingestion.
- [ ] A catalog exists and each entry has a stable source ID and provenance.
- [ ] Real French-LSF lexical data can be searched by canonical gloss/concept.
- [ ] Source of each motion asset is documented and it plays on the intended 3D rig.
- [ ] Phrase-based baseline implemented and evaluated with unknown-sign handling.
- [ ] Genuine French↔LSF paired training data located, validated and split **or explicitly not available**.
- [ ] If trained: recorded exact base model, adapter settings, seeds, dataset hashes, loss, test results, peak VRAM.
- [ ] If **not** trained: report what data and permissions are missing; no false claims.
- [ ] Motion extraction and ASR/emotion features have separate tests and do not contaminate gloss labels.
- [ ] LSF linguistic review and rights gate required before publicly claiming accessible translation.
- [ ] No copyrighted dataset video or personally identifying signer material pushed to public GitHub without permission.

### Explicit non-goals for the first version

- Automatically translating arbitrary French sentences into grammatically correct France LSF.
- Assuming a gloss sequence alone captures pointing, spatial relations, iconic constructions or non-manual grammar.
- Treating all Swiss LSF-CH data as interchangeable with France LSF.
- Automatically downloading every Dicta-Sign video or training on unknown-permission content.
- Claiming a working CGAN or state-of-the-art performance without implementing and measuring it.
- Claiming speech emotion estimates reflect a speaker's internal emotional state.

---

## 10 — Verified sources and primary documentation

**Paper B:** Muanji et al. (2026), *Signing Avatar Synthesis via a Hybrid T5-GAN Approach: Automatic Speech-to-French Sign Language Translation*, *Procedia Computer Science* 290:578–584, https://doi.org/10.1016/j.procs.2026.08.433. **Source:** user-supplied 7-page PDF, pp. 2–6.

**Dicta-Sign Concepts (French):** https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_fre.html

**Dicta-Sign corpus summary and research-license context:** https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/dictasign.html

**Dicta-Sign-LSF-v2 data and license:** https://www.sign-lang.uni-hamburg.de/lrec/data/dictasignlsfv2.html ; https://hdl.handle.net/11403/dicta-sign-lsf-v2/v1

**Rosetta-LSF scientific paper:** https://aclanthology.org/2022.lrec-1.529/ ; archive handle https://hdl.handle.net/11403/rosetta-lsf/v3 (check access per file)

**STKAutogramm France LSF treebank:** https://universaldependencies.org/treebanks/fsl_stkautogramm/ ; https://github.com/UniversalDependencies/UD_French_Sign_Language-STKAutogramm

**Animics (Keyframe/Script modes and data caveat):** https://github.com/upf-gti/animics

**Performs:** https://github.com/upf-gti/performs

**Hugging Face mT5-small (French-inclusive multilingual pretrained model):** https://huggingface.co/google/mt5-small

**Hugging Face PEFT seq2seq LoRA reference:** https://huggingface.co/docs/peft/en/package_reference/peft_model

**Hugging Face Trainer/Seq2SeqTrainer:** https://huggingface.co/docs/transformers/main_classes/trainer

**PyTorch platform-specific installation:** https://pytorch.org/get-started/locally/

**Emotion2vec source:** https://github.com/ddlBoJack/emotion2vec

**Ortolang licensing and per-file access:** https://www.ortolang.fr/en/help/licenses/

---

## 11 — Copy/paste prompt for the coding agent

> You are the lead engineer and research data curator for SignaCulture AI. Read this entire Markdown file and our existing `SignaCulture_AI_Plan_Developpement.md`, inspect the supplied Paper B PDF if available, and inspect the current Git repository. We target **France LSF**, not Swiss LSF-CH, on a single **RTX 4060 8GB GPU**. Begin by verifying dataset permissions and centralizing lexical vs. parallel-sentence vs. motion records in SQLite and versioned JSONL. **Do not train a text-to-gloss model on a sign dictionary alone, do not scrape restricted videos, and do not fabricate French–LSF training pairs.** Build working deterministic gloss lookup and one playable, rights-cleared signing asset. Audit Rosetta-LSF, STKAutogramm and Dicta-Sign for truly aligned and authorized French↔LSF gloss sequences. Only if such approved examples exist, implement and run mT5-small LoRA training using the supplied script, measure actual RTX 4060 VRAM, make correct train/val/test splits, report reproducible metrics and compare with a T5-base Paper B-inspired baseline. Maintain logs, precise source attribution, validation gates and explicit unfinished work. Integrate with Animics/Performs for 3D animation; treat audio prosody/emotion as a separate constrained extension. Work autonomously within those bounds and report what is actually working.

---

**Remember:** `Dicta-Sign gloss dictionary ≠ French-LSF parallel sentences ≠ downloadable rights-cleared videos ≠ generated 3D motion`. Keeping these four objects separate is the central engineering and scientific safeguard.
