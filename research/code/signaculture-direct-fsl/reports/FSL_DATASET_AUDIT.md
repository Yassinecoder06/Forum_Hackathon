# SignVerse-2M France-LSF audit

The public release was pinned to revision `0c7e956a8471acad3a9cebaeb66bb23bf5deaafa`. Both metadata CSVs contained 39,197 rows. An exact `fsl` filter identified 900 recordings (810 train, 45 validation, 45 test). The official archive naming and pinned Hub inventory mapped 860 recordings to 52 TARs totaling 46,820,157,440 bytes. Forty FSL recordings had no published archive mapping. The full 724-TAR inventory totals 598,541,107,200 bytes. Neither the recordings' usable duration nor subtitle clip totals and language distribution can be established from these metadata. The micro ZIP was listed but was not downloaded or used.

The first pilot used seven existing cached recordings and retained 87 cues from 95 candidates. To expand training, Colab downloaded one 255,692,800-byte pinned archive, `dataset/Sign_DWPose_NPZ_000637.tar`. The index identified 34 eligible train FSL recordings with matching French-caption files. Bounded staging selected eight and transferred 22,961,211 bytes of derived pose and caption files. Combined preprocessing retained 168 of 182 cues across 15 recordings, including 102 train, 47 validation and 19 test. No new held-out recordings were added. The supporting counts are in `dataset_audit.json` and `shard-pilot.log`.

Language was joined from source identifiers across release metadata. The selected French VTT tracks establish the released subtitle language, with transcription and translation provenance unknown. Poses contain no independent language or signer ID. There were zero verified gloss labels. Paired caption files and source identifiers remain private.

The release is CC BY-NC 4.0. This audit and experiment use released derived poses and captions for noncommercial research. Underlying recording permissions were not independently established. No original RGB recordings were downloaded or redistributed. [Release](https://huggingface.co/datasets/SignerX/SignVerse-2M)

## Additional official test recordings

A follow-up Colab session downloaded two pinned shards, `000644` (530,841,600 bytes) and `000667` (568,299,520 bytes). These shards indexed three and two official FSL test recordings, respectively. One per shard passed the bounded French-VTT and pose staging filters. Together they yielded 26 accepted cues out of 27 candidates. The checkpoint, original cue manifest, and train/template data remained unchanged. The two additional recordings do not occur in the original train, validation, or test sets. [Caption alignment review](ALIGNMENT_REVIEW.md)
