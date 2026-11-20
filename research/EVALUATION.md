# Evaluation protocol and limits

Motion: same held-out windows and confidence masks for all methods; video-disjoint, signer-disjoint unavailable. Report normalized2DMPJPE, wrist/fingertip and hand-joint errors, derivative errors at24fps, boundary velocity, projected bone consistency and invalid fraction. No arbitrary window is a gold lexical transition. Future work needs multiple seeds, more independent videos, uncertainty intervals and reviewer assessments.

Text: preserve annotation conventions and source-recording groups; never use synthetic labels as gold. Report lexical WER, exact, BLEU/chrF and unknown prediction labels. Use training vocabulary only. Free translations can be many-to-many and omit nonmanual grammar; scorer cannot establish semantics. The generic HF evaluation entrypoint remains unexecuted, while the separate research/code/training/fsl_size_comparison.py runner actually evaluated and saved the completed mT5-small and T5-base feasibility predictions; both scored zero development exact matches. Existing test splits have historical exposure, so they cannot be advertised as untouched new benchmarks.

No fluent France-LSF signer assessment was available in this session. No naturalness or intelligibility claim is established.

Three narrow LSFB seeds repeat an aggregate advantage on the same 23 inspected examples; they do not supply 69 independent examples. All retain 0/6 multi-sign exact and worse multi-sign WER than retrieval. The source alignment audit verifies original-event consistency while 62.31% of retained rows contain nonlexical events omitted by the lexical target. Neither internal consistency nor low lexical WER establishes complete signing. See LSFB_SEED_REPLICATION_RESULT.md and LSFB_SOURCE_ALIGNMENT_AUDIT.md.
