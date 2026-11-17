"""Real emotion2vec+ inference, isolated from FastAPI and pose dependencies."""
import argparse
import json
from pathlib import Path
import torch
try:
    from .emotion_validation import verify_checkpoint, normalize_scores
except ImportError:  # Worker is launched as a standalone script by the API.
    from emotion_validation import verify_checkpoint, normalize_scores

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
    Path(output).write_text(json.dumps({'scores':values,'model':'emotion2vec_plus_seed','device':'cpu','weight_coverage':coverage},allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('model');p.add_argument('output');a=p.parse_args()
    run(a.audio,a.model,a.output)
