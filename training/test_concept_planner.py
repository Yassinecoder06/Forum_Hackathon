import json
import pytest
from concept_planner import validate_response,load_vocabulary_objects,linguistic_guards,ROOT

V=[{'id':'real-A','concept':'fleur'},{'id':'real-B','concept':'écrire'}]
def test_exact_membership_and_no_repairs():
    x=validate_response('{"concept_ids":[1,0],"unsupported":[],"warnings":[]}',V)
    assert x['concept_ids']==['real-B','real-A']
    for ids in [[2],[-1],[True],['real-A'],['fleur']]:
        with pytest.raises(ValueError):validate_response(json.dumps(dict(concept_ids=ids,unsupported=[],warnings=[])),V)
def test_schema_rejects_unstructured_or_extra_fields():
    for obj in [{'concept_ids':[]},dict(concept_ids=[],unsupported=[],warnings=[],extra='yes'),dict(concept_ids=[],unsupported='oops',warnings=[])]:
        with pytest.raises(ValueError):validate_response(json.dumps(obj),V)
def test_duplicate_vocabulary_rejected():
    with pytest.raises(ValueError):load_vocabulary_objects([V[0],V[0]])
@pytest.mark.xfail(strict=True,reason="Known locked-run defect: fr_core_news_sm3.8.0 misses Alice as NOUN/nsubj; keep desired assertion, no comprehensive NER claim")
def test_guard_keeps_critical_meanings_explicit():
    r=linguistic_guards("Alice ne possède pas cinq fleurs ?")
    assert {'NEGATION_UNSUPPORTED','QUESTION_UNSUPPORTED','NUMBER_UNSUPPORTED','NAMED_ENTITY_UNSUPPORTED'}<=set(r['warnings'])
    assert r['unsupported']
def test_real_vocabulary_and_software_fixture_boundary():
    v=json.loads((ROOT/'research/concept_benchmark/vocabulary.json').read_text())
    assert len(v)==len({r['id'] for r in v})==440
    b=json.loads((ROOT/'research/concept_benchmark/benchmark.json').read_text())
    assert b['kind']=='AUTHORED_SOFTWARE_EXPECTATIONS_NOT_LSF_GROUND_TRUTH_NOT_TRAINING'
    assert len(b['cases'])==24


def test_advisory_false_positive_cannot_erase_known_bonjour():
    from concept_planner import merge_advisory_guards
    obj={'concept_ids':['real-bonjour-id'],'unsupported':[],'warnings':[]}
    guard={'named':['bonjour'],'unsupported':['Nom propre possible : Bonjour'],
           'warnings':['NAMED_ENTITY_UNSUPPORTED']}
    result=merge_advisory_guards(obj,guard,'Bonjour.')
    assert result['concept_ids']==['real-bonjour-id']
    assert result['unsupported']==guard['unsupported']
    assert 'NAME_DETECTION_IS_ADVISORY_AND_INCOMPLETE' in result['warnings']

def test_alice_empty_plan_cannot_silently_disappear():
    from concept_planner import merge_advisory_guards
    # Even when neither small NER nor candidate selection finds the unknown name,
    # a wholly empty result must retain the original input as unsupported.
    result=merge_advisory_guards({'concept_ids':[],'unsupported':[],'warnings':[]},
                                {'named':[],'unsupported':[],'warnings':[]},'Alice.')
    assert result['concept_ids']==[] and result['unsupported']==['Alice.']
    assert 'NO_SUPPORTED_MEANING_FOUND' in result['warnings']
