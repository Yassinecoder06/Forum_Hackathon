"""Real emotion2vec+ inference, isolated from FastAPI and pose dependencies."""
import argparse
import json
import tempfile
import wave
from pathlib import Path
import numpy as np
import torch
try:
    from .emotion_validation import verify_checkpoint, normalize_scores
    from .emotion_timeline import PAUSE_RMS, build_timeline, window_bounds
except ImportError:  # Worker is launched as a standalone script by the API.
    from emotion_validation import verify_checkpoint, normalize_scores
    from emotion_timeline import PAUSE_RMS, build_timeline, window_bounds


def score(model, inputs):
    results = model.generate(input=inputs, granularity='utterance', extract_embedding=False)
    if len(results) != len(inputs):
        raise ValueError('emotion2vec returned an unexpected number of results')
    return [normalize_scores(item['labels'], item['scores']) for item in results]


def run(audio, model_path, output):
    torch.set_num_threads(4)
    from funasr import AutoModel
    model=AutoModel(model=str(Path(model_path).resolve()),device='cpu',disable_update=True,disable_pbar=True)
    coverage=verify_checkpoint(model.model,Path(model_path)/'model.pt')
    if coverage['missing_unused_decoder_tensors']:
        def reject_untrained_decoder(*args):
            raise RuntimeError('Inference attempted to use the untrained reconstruction decoder')
        model.model.modality_encoders['AUDIO'].decoder.register_forward_pre_hook(reject_untrained_decoder)
    audio=Path(audio).resolve()
    values=score(model,[str(audio)])[0]

    # Overlapping windows so the avatar can follow the voice; silent windows are skipped.
    with wave.open(str(audio)) as stream:
        rate=stream.getframerate();params=stream.getparams()
        pcm=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2')
    bounds=window_bounds(len(pcm),rate)
    window_scores=[None]*len(bounds)
    with tempfile.TemporaryDirectory(prefix='emotion-windows-') as temp:
        paths,indices=[],[]
        for i,(start,end) in enumerate(bounds):
            chunk=pcm[start:end]
            if float(np.sqrt(np.mean((chunk.astype(float)/32768)**2)))<PAUSE_RMS:
                continue
            path=Path(temp)/f'w{i:03d}.wav'
            with wave.open(str(path),'wb') as out:
                out.setparams(params);out.writeframes(chunk.tobytes())
            paths.append(str(path));indices.append(i)
        if paths:
            for i,item in zip(indices,score(model,paths)):
                window_scores[i]=item
    timeline=build_timeline(bounds,rate,window_scores)
    Path(output).write_text(json.dumps({'scores':values,'timeline':timeline,'model':'emotion2vec_plus_seed','device':'cpu','weight_coverage':coverage},allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('model');p.add_argument('output');a=p.parse_args()
    run(a.audio,a.model,a.output)
