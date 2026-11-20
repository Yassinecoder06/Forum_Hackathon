import torch
from train_torch import Model,metrics,PAD,BOS,EOS

def test_causal_mask():
    torch.manual_seed(1);m=Model(20,dim=32,dropout=0.).eval()
    x=torch.tensor([[4,5,EOS]]);a=torch.tensor([[BOS,6,7]]);b=torch.tensor([[BOS,6,9]])
    with torch.no_grad():ya,yb=m(x,a),m(x,b)
    assert torch.allclose(ya[:,:2],yb[:,:2],atol=1e-6), 'Future target leakage'

def test_padding():
    torch.manual_seed(1);m=Model(20,dim=32,dropout=0.).eval()
    y=torch.tensor([[BOS,6]])
    with torch.no_grad():a=m(torch.tensor([[4,5,EOS]]),y);b=m(torch.tensor([[4,5,EOS,PAD,PAD]]),y)
    assert torch.allclose(a,b,atol=1e-5)

def test_metrics():
    score=metrics([['a','b'],['c']],[[],['x','x']],{'a','b','c'})
    assert score['all']['exact']==0 and score['all']['wer']==4/3
    assert score['unknown_prediction_labels']==2 and score['multi_sign']['wer']==1

if __name__=='__main__':
    test_causal_mask();test_padding();test_metrics();print('Three checks passed: causal mask, source padding, metrics retain errors.')
