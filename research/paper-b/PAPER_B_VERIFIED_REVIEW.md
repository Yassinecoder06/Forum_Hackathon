> The paper interpretation remains applicable; the implementation table is a historical snapshot. See [current status](../STATUS.md) for later T5-base/GAN experiments, the expanded library and verified audio/export integration.

# Verified review of the supplied T5–GAN paper

> Subsequent Colab trials now include T5-base and FLAN-T5-base. The table below records the earlier implementation snapshot. The [new paper/data/scoring audit](../text-to-gloss/lsfb/COLAB_ZERO_EXACT_DIAGNOSIS.md) documents why those trials still do not reproduce the paper and why all exact-match scores were zero.

The lead read all seven pages of the actual local `research_paper.pdf` on 15 November 2026 and inspected the methodology and results figures. Its SHA-256 is `e2ecf423d5bc65c283f0c84bbe60f9b4bf799e309e791561344b191c4e498d36`; it is identical to `/home/jmal/Downloads/research_paper.pdf`. Earlier statements that the paper was unavailable are superseded by this review. The paper supplied by the user remains unchanged.

**Reference:** Manuel Muanji et al., *Signing Avatar Synthesis via a Hybrid T5-GAN Approach: Automatic Speech-to-French Sign Language Translation*, Procedia Computer Science 290 (2026), 578–584. Page references below are PDF page numbers, followed by printed pages where useful.

## The actual proposed pipeline

Voice → Google speech recognition → French sentence → fine-tuned T5-base → expert/fuzzy post-correction and reordering → retrieve sign landmark CSVs → conditional GAN between adjacent signs → Pygame animation. Typed French bypasses speech recognition. Algorithm 1 withholds animation when glosses are not validated (PDF pp. 2–3, §§3.1–3.2).

| Component | What the paper describes | Current SignaCulture implementation |
|---|---|---|
| Speech input | Google ASR; typed-text alternative | Typed French. Uploaded audio is analyzed for pitch, energy, pauses and emotion; it is not transcribed. |
| Text-to-gloss | T5-base, 220M parameters; 319 added gloss tokens; AdamW, LR 2e-5, batch 4, 20 epochs, weight decay .01 (PDF p. 3, §3.3) | T5-small and mT5-small LoRA research experiments; no T5-base reproduction or matching tokenizer extension. Default uses concept lookup; optional Qwen proposes editable French concepts. |
| Post-correction | 47 expert mappings, RapidFuzz threshold 85%, rule-based SOV reordering (PDF p. 4, §3.4) | Valid library IDs, unknowns and manual editing. No reproduction of the 47 rules, fuzzy correction or paper's reordering. |
| Sign assets | 319 isolated signs; MediaPipe Holistic landmarks stored in CSV (PDF p. 3, §3.2) | 440 real captured motions from 447 retrieved lexical clips; MediaPipe and attributed JSON assets. These are different assets, not the paper's training dataset. |
| Transitions | Conditional GAN, adversarial + L1 loss; λ=1000; 10 transition frames at 30 fps (PDF p. 4, §3.5) | Now includes a trained five-point intra-sign reconstruction CGAN experiment (TRANSITION_REPORT.md). Playback uses 333 ms quintic bridges; no validated inter-sign GAN co-articulation. |
| Rendering | Stylized 2D Pygame avatar; realistic 3D is future work (PDF p. 6, §5) | Procedural Three.js 3D avatar and browser WebM preview export. |
| Nonmanuals / expression | Missing facial/head grammatical modeling; separate nonmanual GANs are proposed future work (PDF p. 6, §5) | Some captured mouth/brow ratios and experimental emotion2vec modulation; not validated nonmanual grammar. This is our extension, not a reproduced paper component. |
| Evaluation | Two professional LSF interpreters evaluate 20 sentences (PDF p. 6, §4.5) | Software checks and measured research tests; no equivalent interpreter study. |

## What the GAN actually generates

The paper begins with 75 landmarks × 3 coordinates (225 values/frame), then selects nose, wrists and middle fingertips and reduces the representation to **10 features**. Its generator predicts **10 transition frames** between two signs, approximately **0.33 seconds at 30 fps**. It does not generate whole 3D signing performances, full 21-point hands or facial grammar (§3.5).

The generator is an MLP with three hidden layers, batch normalization and dropout .4; the discriminator uses LeakyReLU .2 and dropout .3. The generator objective combines adversarial loss and L1 reconstruction weighted by 1000. Hidden-layer widths and the exact 225-to-10 feature projection are not fully specified in the article; these must not be invented and called an exact reproduction.

## Data and reported results

The paper reports **2,500 French–gloss sentence pairs**, **319 isolated signs**, **379 complete sentence sequences / real transition pairs** and **1,000 synthetic transition pairs** (§3.2, Table 1). Its conclusion says code and data are available upon request; the supplied PDF does not provide those artifacts. Our 128 source-annotated research pairs (61 training) and lexical video library do not replace the missing sentence and transition datasets.

Reported corrected translation metrics are BLEU .4347, WER .2364 and exact match 30.0% (Table 3). Reported GAN metrics are MSE .021419, MAE .060860, RMSE .146354 and cosine similarity .985 (Table 4). These are authors' reported measurements, not our reproduced results, and their BLEU scale/protocol must not be silently equated to our sacreBLEU scores.

There is an internal latency inconsistency: the abstract and §4.4 report approximately 4,450 ms for internal modules, whereas §5 states 350 ms excluding ASR. Table 6 reports a total range of 2,475–8,450 ms. An exact latency reproduction needs clarification. The article also does not sufficiently detail the held-out split and all preprocessing choices to establish comparability from the PDF alone.

## Implication for this project

We have implemented a different, partial prototype inspired by this architecture. We have **not implemented the complete T5–GAN system**. Since this review, a paper-inspired CGAN has been trained for within-sign reconstruction only, and the renderer has been repaired with a rigged character and preserved clip transitions (see TRANSITION_REPORT.md and FAILURE_ANALYSIS_AND_REPAIR.md). The missing pieces include T5-base with a matched gloss vocabulary, the expert correction/reordering rules, paired transition supervision, conditional GAN research/code/training/evaluation, and professional perceptual evaluation. Local hardware is not evidence that a compact transition MLP cannot be attempted; actual paired data and a defensible evaluation are the immediate missing prerequisites. No GAN memory/quality result has been measured here.

The corrected `docs/presentation/SignaCulture_Full_Pipeline.pdf` separately depicts the current application and the supplied paper's architecture. Historical logs describing an earlier lack of PDF access remain historical, not the current evidence status.
