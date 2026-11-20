# SignaCulture direct France-LSF motion pilot

French captions condition continuous 2D DWPose predictions. This route runs alongside the gloss-based studio. The public metadata identify France-LSF and official recording splits. The French subtitle tracks provide weak temporal supervision with unknown transcription provenance.

- [Dataset audit](reports/FSL_DATASET_AUDIT.md)
- [Training results](reports/TRAINING_RESULTS.md)
- [Evaluation](reports/EVALUATION.md)
- [Caption alignment review](reports/ALIGNMENT_REVIEW.md)
- [Limitations](reports/LIMITATIONS.md)
- [Colab SSH runbook](COLAB_SSH.md)
- [Colab notebook](Colab_FSL_Pilot.ipynb)

From this directory, with a compatible PyTorch environment and the repository's private cue/pose cache:

```bash
python -m pip install -r requirements.txt
python scripts/check_gpu.py
python scripts/preprocess_fsl.py
python -m unittest discover -s tests -v
python train.py
python evaluate.py
python predict.py --text "Bonjour, bienvenue au musée." --output outputs/demo_001
```

Use `--config configs/frozen.yaml --out checkpoints/frozen` for the pretrained French encoder. For evaluation and prediction, pass its `--checkpoint checkpoints/frozen/best.pt` and use separate output directories. Outputs and checkpoints are local and excluded from Git. FFmpeg is required for MP4 rendering.
