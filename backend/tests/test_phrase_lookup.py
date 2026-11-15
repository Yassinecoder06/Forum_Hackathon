from backend.app.services import text_lookup
from backend.app.schemas import GlossEntry

def entry(concept, identifier):
    return GlossEntry(gloss_id=identifier, concept_fr=concept, gloss=concept.upper(), language='LSF', source_id='software-test', variant='test', motion_id='M'+identifier, aliases=[], review_status='pending')

def test_dictionary_recovers_missing_inflections(monkeypatch):
    monkeypatch.setattr(text_lookup, 'model', lambda: None)
    entries = [entry(s, str(i)) for i,s in enumerate(['habiter','dormir','chercher'])]
    result = text_lookup.lookup_text('habite dort cherche Inconnu', entries)
    assert [g.concept_fr for g in result['gloss_sequence']] == ['habiter','dormir','chercher']
    assert result['unknowns'] == ['Inconnu']

def test_exact_match_and_ambiguity_are_preserved(monkeypatch):
    monkeypatch.setattr(text_lookup, 'model', lambda: None)
    assert text_lookup.lookup_text('lis', [entry('lis','1'),entry('lire','2')])['gloss_sequence'][0].gloss_id == '1'
    result=text_lookup.lookup_text('dort', [entry('dormir','1'),entry('dormir','2')])
    assert not result['gloss_sequence'] and result['unknowns']==['dort']
