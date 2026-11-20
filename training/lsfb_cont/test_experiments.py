import pytest
from .experiments import metrics,normalize

def test_overlap_normalization_detects_punctuation_only_changes():
    assert normalize('Traduire: Oui !') == normalize('TRADUIRE: oui.')
    assert normalize('Je viens.') != normalize('Je ne viens pas.')

def test_valid_json_can_still_contain_unsupported_signs():
    m=metrics([{'target':['oui']}],['["not-a-sign"]'],{'oui'})
    assert m['invalid']==0 and m['unsupported']==1 and m['exact']==0

def test_empty_list_is_a_failed_prediction():
    m=metrics([{'target':['oui']}],['[]'],{'oui'})
    assert m['invalid']==1 and m['exact']==0 and m['wer']>=1

def test_metric_does_not_silently_drop_missing_predictions():
    with pytest.raises(ValueError):metrics([{'target':['oui']}],[],{'oui'})
