"""Explicit opt-in inference using an existing local FunASR emotion2vec model."""
import argparse
import json
from pathlib import Path


def infer(audio, model_dir):
    model_dir = Path(model_dir)
    if not model_dir.is_dir() or not (model_dir / 'config.yaml').is_file():
        raise ValueError('Provide a complete local FunASR model directory with config.yaml')
    if not Path(audio).is_file():
        raise ValueError('Audio file unavailable')
    from funasr import AutoModel
    model = AutoModel(model=str(model_dir.resolve()), device='cpu', disable_update=True)
    return model.generate(input=str(Path(audio).resolve()), granularity='utterance', extract_embedding=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio'); p.add_argument('--model-dir', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    result = infer(a.audio, a.model_dir)
    Path(a.output).write_text(json.dumps({'model_version':str(Path(a.model_dir).resolve()),'emotion_hypothesis':result,'warning':'Experimental classifier output; not a measurement of internal emotion'},ensure_ascii=False))
