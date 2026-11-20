import pytest
from training.geoquery_lsfb.audit_baseline import validate_graph,score,normalize_query,retrieve

def graph(sign):return [['right-hand-articulation',sign+'-1','HNS'],['left-hand-articulation','hold-1','HNS2'],['during',sign+'-1','hold-1']]
def test_temporal_links_preserved():
 assert validate_graph(graph('a'))=={'articulations':2,'temporal_relations':1}
 broken=graph('a');broken[-1][-1]='missing-1'
 with pytest.raises(ValueError,match='Dangling'):validate_graph(broken)
def test_no_hidden_invalid_filtering():
 bad=graph('a');bad[-1][-1]='absent'
 r=score([bad,graph('b')],[bad,graph('c')]);assert r['examples']==2 and r['invalid_reference_graphs']==1 and not r['all_graphs_structurally_valid'];assert r['serialized_graph_exact']==.5
def test_full_graph_scoring_and_relation_order():
 g=graph('a');r=score([g],[list(reversed(g))]);assert r['serialized_graph_exact']==0 and r['triple_set_exact']==1
 assert normalize_query("state('texas')")==normalize_query("state('utah')")
def test_train_only_retrieval_handles_unknown_input():
 tr=[{'french':'rivière texas'},{'french':'montagne utah'}];assert retrieve(tr,[{'french':'rivière texas'}])==[0];assert len(retrieve(tr,[{'french':'xyznewword'}]))==1

def test_codec_roundtrip_both_hands_unknown_hns_and_edge_order():
 from training.geoquery_lsfb.codec import GraphCodec
 g=[['during','a-1','b-1'],['right-hand-articulation','a-1','\ue001\ue002'],['left-hand-articulation','b-1','\ue003']]
 c=GraphCodec.fit([graph('train')]);assert c.decode(c.encode(g))==g
 assert c.encode(g).count('|u')==2

def test_codec_rejects_missing_and_negative_indices():
 from training.geoquery_lsfb.codec import GraphCodec
 c=GraphCodec(['\ue001'])
 for text in ['R|a|h-1','R|a|h9','R|a|h0 A|0|2','R|a|h0 D|-1|0']:
  with pytest.raises(ValueError):c.decode(text)

def test_neural_graph_metrics_keeps_invalid_outputs():
 from training.geoquery_lsfb.train_graph import graph_metrics
 rows=[{'lsfb_graph':graph('a')},{'lsfb_graph':graph('b')}]
 r=graph_metrics(rows,[None,graph('b')])
 assert r['examples']==2 and r['graph_exact']==.5 and r['invalid_graphs']==1
 assert r['triple_micro_f1']==pytest.approx(2/3)
 with pytest.raises(ValueError,match='Missing predictions'):graph_metrics(rows,[None])
