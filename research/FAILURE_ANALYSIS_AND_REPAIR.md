# Why the first result fell short, and what was repaired

The supplied paper was read in full; see PAPER_B_VERIFIED_REVIEW.md. The earlier deliverable was a partial technical prototype, not a faithful T5–GAN implementation. Passing software checks did not establish signing quality.

| Failure | Concrete effect | Current repair / remaining gap |
|---|---|---|
| Independent primitive meshes instead of a skinned character | Unnatural body, disconnected-looking joints, poor hand appearance | Replaced with a local articulated Noa model. Body and fingers deform a continuous skinned mesh. Model provenance and pending redistribution rights are documented separately. |
| Directly positioning / stretching limb geometry | Inconsistent proportions; monocular depth could put hands inside the torso | Two-bone arm IK, fixed bind lengths, measured palm basis and finger rotations, plus a local torso penetration constraint. Source assets are unchanged; this is approximate retargeting. |
| Two clips share the same join timestamp | A map entry overwrote one source endpoint and the pose could snap | Dedicated 333 ms quintic transition interval; preserve both clips' endpoints. Original and audio-warped timelines are checked. This is interpolation, not learned linguistic co-articulation. |
| Synthetic fixtures remained in normal text lookup | Museum/culture prompts could play invented technical motions | Default glossary and text composition now exclude synthetic motions. Explicit include_synthetic is only for technical tests. Suggestions use actual available signs. Unknown real signs stay unknown. |
| T5-small / mT5 LoRA on tiny different supervision | Failed generation, unavailable target vocabulary | Still a research-only result. No fabricated replacement for the paper's 2,500 sentence pairs, 319 added tokens, T5-base setup or 47 expert rules. Qwen assistance is also not LSF translation. |
| Missing conditional GAN | No learned transition component at all | A real paper-inspired five-point CGAN has now been implemented, trained and tested, but only on intra-sign reconstruction. See TRANSITION_REPORT.md. Its small gain and worse boundary error do not justify deploying it as inter-sign co-articulation. |
| No comparable perceptual evaluation | Software success was mistaken for product readiness | Source-video comparison and actual rendering/export checks are retained. Qualified LSF intelligibility and grammatical/nonmanual evaluation remain undone. |

## Working path now

Real captured sign lookup → dedicated transition timeline → articulated body/hand retargeting → Three.js preview → actual browser video export. Missing fingers use neutral rendering posture rather than fabricated capture data. Typed text, prosody analysis and experimental emotion hypotheses remain separate. No full sentence translation or state-of-the-art result is claimed.

## Verifications

- API/schema/capture tests include synthetic exclusion, transition timing and publication gating.
- Frontend motion checks preserve both source endpoints, source data, monotonic time, and audio pause holds; IK checks preserve bone lengths and deterministic seeking.
- Browser checks wait for the actual rigged model before composition/export. The source video is available for comparison.
- Conditional GAN uses clip-disjoint train/validation/test windows with fixed training-only normalization; its true scope and all baseline errors are reported.

The paper's GAN only bridges five selected points (10 x/y features). It cannot by itself supply a good full 3D mesh, full hand configuration, facial grammar, or a correct French-to-LSF translation. Those are distinct parts of the system and must be assessed separately.
