# Voice → avatar update: benchmarks

All numbers measured on a Windows laptop (CPU only), branch `feat/voice-emotion-face`. CaFE: Canadian French Emotional speech dataset (Gournay et al., 2018, CC BY-NC-SA 4.0), used locally and not redistributed.

## Emotion recognition (emotion2vec+, CaFE French emotional speech)
| Metric | Result |
|---|---|
| Clips | 936 (12 actors × 6 sentences × 7 emotions, 2 intensities) |
| Accuracy (clip-level) | **89.7 %** (chance 14.3 %) |
| Per emotion | surprised 99.3 · sad 94.4 · disgusted 92.4 · neutral 91.7 · fearful 85.4 · angry 84.0 · happy 81.9 % |
| Strong vs low intensity | 94.2 % vs 85.9 % |
| Per actor | 75.6 % – 97.4 % |
| Speed | 0.8 s per clip (CPU), all 185 checkpoint tensors verified |
| Caveat | 75 % of predictions have > 99 % confidence; the model may have seen CaFE in training, so real-world accuracy is likely lower |

## Stress detection (real speech, stress added digitally to one word)
| Test | Result |
|---|---|
| « Bonjour » / « fou » / « jardin » stressed (+8 dB, +3 semitones) | detected, and only that word ✓ |
| Same word, mild stress (+4 dB, +1.5 semitones) | not detected |
| Quiet sentence-final « lac » (+8 dB) | not detected |
| No stress (plain clip) | nothing flagged ✓ |
| 8 unedited neutral CaFE sentences | 7 with nothing flagged, 1 false alarm |

## Effect on the stressed sign (JARDIN)
| | Before | After |
|---|---|---|
| Sign duration | 0.60 s (squeezed) | 1.11 s incl. 0.15 s hold |
| Hand path (BONJOUR test) | 96 cm | 112 cm (+17 %) |
| Movement height (BONJOUR test) | 43.7 cm | 50.3 cm (+16 %) |
| Face contact point | – | unchanged (≤ 0.3 cm) |
| Head nod | 0° | up to 5.9° |

## Pace / timing
| | Before | After |
|---|---|---|
| Sign start vs its word | spread evenly over voiced audio (BONJOUR stretched to 2.75 s over a pause) | each sign starts at its word (DANS at 3.84 s = word onset) |
| Speeds handled | — | slow / normal / fast, playback rate kept 0.75–1.35× for readability |
| Known limit | — | LSF signs are longer than words: phrase ends ~1.9 s after the audio |

## Stability and quality
| Check | Result |
|---|---|
| Hand-tracking state changes over a 7 s phrase (176 frames) | 5, all at sign boundaries (glitch fix) |
| Face emotion during signing (surprise 51 %) | eye-widening 0.26 → 0.54–0.60 |
| Automated tests | backend 95 passed, weight audit 5 passed, frontend checks + build pass |
| Analysis time per upload (emotion + transcription, CPU) | ~20–35 s |
