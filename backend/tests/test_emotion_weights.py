"""Small counterfeit checkpoints test failure behavior without external weights."""
import pytest
torch = pytest.importorskip('torch', reason='Weight-audit checks require the training/audio PyTorch environment')
from backend.app.workers.emotion_validation import verify_checkpoint, normalize_scores

class Tiny:
    def state_dict(self):
        return {'proj.weight':torch.ones(9,2),'proj.bias':torch.zeros(9),
                'modality_encoders.AUDIO.decoder.proj.weight':torch.zeros(2,2)}


def checkpoint(tmp_path, state):
    path=tmp_path/'model.pt';torch.save({'model':state},path);return path


def test_exact_weights_and_narrow_unused_decoder_exception(tmp_path):
    s=Tiny().state_dict();del s['modality_encoders.AUDIO.decoder.proj.weight']
    r=verify_checkpoint(Tiny(),checkpoint(tmp_path,s))
    assert r['verified_tensor_count']==2
    assert r['missing_unused_decoder_tensors']==['modality_encoders.AUDIO.decoder.proj.weight']
    assert not r['semantic_accuracy_validated']


def test_missing_classification_head_rejected(tmp_path):
    s=Tiny().state_dict();del s['proj.weight']
    with pytest.raises(ValueError,match='Missing emotion inference'):
        verify_checkpoint(Tiny(),checkpoint(tmp_path,s))


def test_loader_random_initialization_cannot_pass_as_loaded(tmp_path):
    s=Tiny().state_dict();s['proj.weight'].zero_()
    with pytest.raises(ValueError,match='differs from checkpoint'):
        verify_checkpoint(Tiny(),checkpoint(tmp_path,s))


def test_ambiguous_scope_mapping_rejected(tmp_path):
    s=Tiny().state_dict();s['d2v_model.proj.weight']=s['proj.weight']
    with pytest.raises(ValueError,match='Ambiguous'):
        verify_checkpoint(Tiny(),checkpoint(tmp_path,s))


def test_scores_are_not_silently_truncated_or_relabelled():
    labels=['angry','disgusted','fearful','happy','neutral','other','sad','surprised','<unk>']
    values=normalize_scores(labels,[0,0,0,0,1,0,0,0,0]);assert values['neutral']==1
    for names,scores in [(labels,[1]),(labels,[float('nan')]*9),(labels,[.2]*9),(['angry']*9,[1/9]*9)]:
        with pytest.raises(ValueError):normalize_scores(names,scores)
