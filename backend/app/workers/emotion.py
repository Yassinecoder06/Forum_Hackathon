"""Real emotion2vec+ inference, isolated from FastAPI and pose dependencies."""
import argparse
import json
import os
from pathlib import Path
import torch
try:
    from .emotion_validation import verify_checkpoint, normalize_scores
except ImportError:  # Worker is launched as a standalone script by the API.
    from emotion_validation import verify_checkpoint, normalize_scores

MODEL_ID=os.getenv('SIGNACULTURE_EMOTION_MODEL_ID') or (Path(os.environ['SIGNACULTURE_EMOTION_MODEL_DIR']).name if os.getenv('SIGNACULTURE_EMOTION_MODEL_DIR') else 'emotion2vec_plus_seed')

def load(model_path):
    torch.set_num_threads(4)
    from funasr import AutoModel
    model=AutoModel(model=str(Path(model_path).resolve()),device='cpu',disable_update=True,disable_pbar=True)
    coverage=verify_checkpoint(model.model,Path(model_path)/'model.pt')
    if coverage['missing_unused_decoder_tensors']:
        def reject_untrained_decoder(*args):
            raise RuntimeError('Inference attempted to use the untrained reconstruction decoder')
        model.model.modality_encoders['AUDIO'].decoder.register_forward_pre_hook(reject_untrained_decoder)
    return model,coverage

def scores(model,audio):
    item=model.generate(input=str(Path(audio).resolve()),granularity='utterance',extract_embedding=False)[0]
    return normalize_scores(item['labels'],item['scores'])

def run_batch(audios,model_path,output):
    """One model load for many clips (film mode: one clip per subtitle line)."""
    model,coverage=load(model_path)
    Path(output).write_text(json.dumps({'items':[scores(model,a) for a in audios],'model':MODEL_ID,'device':'cpu','weight_coverage':coverage},allow_nan=False))

def run(audio, model_path, output):
    torch.set_num_threads(4)
    from funasr import AutoModel
    model=AutoModel(model=str(Path(model_path).resolve()),device='cpu',disable_update=True,disable_pbar=True)
    coverage=verify_checkpoint(model.model,Path(model_path)/'model.pt')
    if coverage['missing_unused_decoder_tensors']:
        def reject_untrained_decoder(*args):
            raise RuntimeError('Inference attempted to use the untrained reconstruction decoder')
        model.model.modality_encoders['AUDIO'].decoder.register_forward_pre_hook(reject_untrained_decoder)
    results=model.generate(input=str(Path(audio).resolve()),granularity='utterance',extract_embedding=False)
    item=results[0]
    labels=item['labels'];scores=item['scores']
    values=normalize_scores(labels,scores)
    Path(output).write_text(json.dumps({'scores':values,'model':MODEL_ID,'device':'cpu','weight_coverage':coverage},allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('model');p.add_argument('output')
    p.add_argument('--batch',action='store_true',help='audio is a JSON list of wav paths');a=p.parse_args()
    if a.batch: run_batch(json.loads(Path(a.audio).read_text()),a.model,a.output)
    else: run(a.audio,a.model,a.output)
