import importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from prepare import canonical_event,candidates
from metrics import evaluate_arrays,distance
import numpy as np

def test_source_lemma_preserves_variation_and_nonlexical_identity():
    event={'gloss':'appeler(fxd)','lemma':'appeler','sign_type':'lexical','variation':'(fxd)'}
    assert canonical_event(event)=='lexical:appeler' and event['variation']=='(fxd)' and event['gloss']=='appeler(fxd)'
    assert canonical_event({'gloss':'pt:pro2','lemma':'pt','sign_type':'pointing'})=='pointing:pt:pro2'

def test_repeated_signs_kept_but_duplicate_events_removed():
    event={'start_ms':0,'end_ms':100,'gloss':'oui','lemma':'oui','sign_type':'lexical'}
    second={**event,'start_ms':150,'end_ms':250}
    source={'id':{'signer.txt':'S1','annotations.translation.json':[{'start_ms':0,'end_ms':300,'text':'Oui, oui.'}],'annotations.both_hands.json':[event,event,second]}}
    row=candidates(source)[0]
    assert row['units']==['lexical:oui','lexical:oui'] and len(row['events'])==2

def test_unseen_references_remain_errors_and_do_not_disappear_as_padding():
    scores=evaluate_arrays(np.array([[0,10,1,0]]),np.array([[10,2,1,-100]]),{10:'oui'})
    assert scores['wer']==.5 and scores['exact_match']==0 and scores['reference_oov_rate']==.5

def test_unknown_prediction_does_not_match_unknown_reference():
    scores=evaluate_arrays(np.array([[0,2,1]]),np.array([[2,1,-100]]),{10:'oui'})
    assert scores['wer']==1 and scores['exact_match']==0 and scores['invalid_sequence_rate']==1

def test_correct_atomic_sequence_requires_eos():
    reverse={10:'oui',11:'savoir'}
    labels=np.array([[10,11,1]])
    assert evaluate_arrays(np.array([[0,10,11,1]]),labels,reverse)['exact_match']==1
    assert evaluate_arrays(np.array([[0,10,11]]),labels,reverse)['exact_match']==0
