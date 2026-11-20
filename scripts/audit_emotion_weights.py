"""Audit actual local inference and test the missing decoder's influence."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from backend.app.workers.emotion_validation import verify_checkpoint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,default=Path('.models/emotion2vec-seed'))
    ap.add_argument('--audio',type=Path,default=Path('.runtime/pipeline-recheck-voice.wav'))
    ap.add_argument('--output',type=Path,default=Path('reports/EMOTION2VEC_WEIGHT_AUDIT.json'))
    args=ap.parse_args();torch.set_num_threads(4)
    from funasr import AutoModel
    auto=AutoModel(model=str(args.model.resolve()),device='cpu',disable_update=True,disable_pbar=True)
    model=auto.model.eval()
    coverage=verify_checkpoint(model,args.model/'model.pt')
    calls=[];decoder=model.modality_encoders['AUDIO'].decoder
    hook=decoder.register_forward_hook(lambda *unused:calls.append(True))
    def infer():
        result=auto.generate(input=str(args.audio.resolve()),granularity='utterance',extract_embedding=False)[0]
        return dict(labels=result['labels'],scores=result['scores'])
    original=infer()
    # If the decoder is used through functional access instead of __call__, its
    # NaN weights would also contaminate output. Do not persist this diagnostic model.
    with torch.no_grad():
        for p in decoder.parameters():p.fill_(float('nan'))
    perturbed=infer();hook.remove()
    assert not calls,'Reconstruction decoder was called during inference'
    assert original==perturbed,'Unused-decoder perturbation changed inference'
    assert all(torch.isfinite(torch.tensor(original['scores'])))
    sources={}
    for component in [type(model),type(model.modality_encoders['AUDIO'])]:
        path=Path(inspect.getfile(component));sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(status='complete',coverage=coverage,funasr_version=importlib.metadata.version('funasr'),
                model_revision=(args.model/'.cache/huggingface/download/model.pt.metadata').read_text().splitlines()[0],
                checkpoint_sha256=hashlib.file_digest((args.model/'model.pt').open('rb'),'sha256').hexdigest(),
                audio_sha256=hashlib.sha256(args.audio.read_bytes()).hexdigest(),code_sha256=sources,
                reconstruction_decoder_forward_calls=len(calls),decoder_nan_perturbation_scores_identical=original==perturbed,
                result=original,qualified_emotion_accuracy_validated=False,
                scope='One existing 8-second French demo excerpt; validates weight coverage and execution, not emotional correctness.',
                license_status='Local model card declares other/model-license; no independent commercial-use permission conclusion.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
