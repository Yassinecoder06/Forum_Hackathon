from .train_extracted import measure

def test_json_reference_scoring_and_length_groups():
    rows=[{'target':['oui'],'target_text':'["oui"]'},
          {'target':['oui','non'],'target_text':'["oui", "non"]'}]
    m=measure(rows,['["oui"]','["oui", "non"]'])
    assert m['exact']==1 and m['raw_token_wer']==0
    assert m['by_length']['multiple_signs']['exact']==1

def test_malformed_repeated_json_is_not_erased():
    row={'target':['oui'],'target_text':'["oui"]'}
    a=measure([row],['["oui", "oui", "oui"]'])
    b=measure([row],['["oui", "oui", "oui", "'])
    assert b['invalid']==1 and b['operation_wer']>=a['operation_wer']
    assert b['raw_token_wer']>=a['raw_token_wer']
