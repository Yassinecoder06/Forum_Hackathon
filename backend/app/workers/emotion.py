"""Real emotion2vec+ inference, isolated from FastAPI and pose dependencies."""
import argparse
import json
from pathlib import Path
import torch

def run(audio, model_path, output):
    torch.set_num_threads(4)
    from funasr import AutoModel
    model=AutoModel(model=str(Path(model_path).resolve()),device='cpu',disable_update=True,disable_pbar=True)
    results=model.generate(input=str(Path(audio).resolve()),granularity='utterance',extract_embedding=False)
    item=results[0]
    labels=item['labels'];scores=item['scores']
    values={str(label).split('/')[-1].lower().replace('<unk>','unknown'):float(score) for label,score in zip(labels,scores)}
    Path(output).write_text(json.dumps({'scores':values,'model':'emotion2vec_plus_seed','device':'cpu'},allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('model');p.add_argument('output');a=p.parse_args()
    run(a.audio,a.model,a.output)
