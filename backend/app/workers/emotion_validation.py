"""Verify the local emotion2vec inference weights instead of trusting loader logs."""
import math
import re
import torch

# The audited FunASR inference path uses extract_features + proj and does not
# call this reconstruction decoder. Keep the exception narrower than 'decoder.*'.
_UNUSED_DECODER = re.compile(r'modality_encoders\.AUDIO\.decoder\.(?:blocks\.[0-3]\.0|proj)\.(?:weight|bias)\Z')


def normalize_scores(labels, scores):
    if len(labels) != 9 or len(scores) != len(labels):
        raise ValueError('Expected nine aligned emotion labels and scores')
    values = {}
    for label, score in zip(labels, scores):
        key = str(label).split('/')[-1].lower().replace('<unk>', 'unknown')
        value = float(score)
        if key in values or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid or duplicate emotion score')
        values[key] = value
    if set(values) != {'angry','disgusted','fearful','happy','neutral','other','sad','surprised','unknown'}:
        raise ValueError('Unexpected emotion label vocabulary')
    if abs(sum(values.values()) - 1) > 1e-4:
        raise ValueError('Emotion scores are not a probability distribution')
    return values


def verify_checkpoint(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for wrapper in ('state_dict', 'model_state_dict', 'model'):
        if wrapper in state:
            state = state[wrapper]
    missing, mapped = [], []
    elements = 0
    for key, value in model.state_dict().items():
        candidates = [k for k in (key, 'd2v_model.'+key, 'module.'+key) if k in state]
        if not candidates:
            if not _UNUSED_DECODER.fullmatch(key):
                raise ValueError('Missing emotion inference parameter: '+key)
            missing.append(key)
            continue
        if len(candidates) != 1:
            raise ValueError('Ambiguous checkpoint mapping: '+key)
        source = state[candidates[0]]
        if value.shape != source.shape or not torch.equal(value.detach().cpu(), source):
            raise ValueError('Emotion inference parameter differs from checkpoint: '+key)
        mapped.append(key)
        elements += value.numel()
    if 'proj.weight' not in mapped or 'proj.bias' not in mapped:
        raise ValueError('Emotion classification head is absent')
    return dict(verified_tensor_count=len(mapped), verified_elements=elements,
                missing_unused_decoder_tensors=missing,
                inference_parameter_check='exact tensor equality',
                semantic_accuracy_validated=False)
