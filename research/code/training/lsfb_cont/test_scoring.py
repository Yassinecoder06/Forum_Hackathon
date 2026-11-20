from .scoring import score

def test_broken_quote_does_not_hide_repetitions():
    reference = ['SIGN "oui"']
    repeated = ' ; '.join(['SIGN "oui"']*4)
    assert score([repeated+' ; SIGN "'], reference)['operation_wer'] >= score([repeated], reference)['operation_wer']

def test_quoted_separator_and_fingerspelling():
    text = 'SIGN "a ; b" ; FINGERSPELL "ok" "fs:fk(ok)"'
    result = score([text], [text])
    assert result['exact_match'] == 1 and result['operation_wer'] == 0

def test_empty_and_malformed_are_not_exact():
    for p in ['', 'SIGN "oui" ;', 'garbage SIGN "oui"']:
        r = score([p], ['SIGN "oui"'])
        assert r['exact_match'] == 0
    assert score([''], ['SIGN "oui"'])['operation_wer'] == 1
