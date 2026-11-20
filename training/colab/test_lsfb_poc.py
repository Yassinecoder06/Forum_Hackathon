import importlib.util,io,json,tarfile
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('lsfb_poc',Path(__file__).with_name('lsfb_poc.py'));poc=importlib.util.module_from_spec(spec);spec.loader.exec_module(poc)

def test_extract_alignment_and_group_split(tmp_path):
    archive=tmp_path/'sample.tar'
    with tarfile.open(archive,'w') as t:
        for i in range(9):
            for suffix,value in [('signer.txt',f'S{i}'),('annotations.translation.json',[{'start_ms':0,'end_ms':1000,'text':f'Phrase {i}'}]),('annotations.both_hands.json',[{'start_ms':0,'end_ms':500,'gloss':'A'},{'start_ms':0,'end_ms':500,'gloss':'A'},{'start_ms':800,'end_ms':2000,'gloss':'OUTSIDE'}])]:
                data=(value if isinstance(value,str) else json.dumps(value)).encode();info=tarfile.TarInfo(f'recording{i}.{suffix}');info.size=len(data);t.addfile(info,io.BytesIO(data))
    candidates=tmp_path/'pairs.jsonl';audit=poc.extract(archive,candidates)
    assert audit['candidates']==9
    assert all(json.loads(l)['gloss_sequence']=='A' for l in candidates.read_text().splitlines())
    with pytest.raises(ValueError,match='No eligible'):poc.split(candidates,tmp_path/'strict')
    report=poc.split(candidates,tmp_path/'splits',allow_weak=True)
    assert report['grouping']=='signer_id'
    groups=[set(v['groups']) for v in report['splits'].values()]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])

def test_insufficient_independent_groups_rejected(tmp_path):
    path=tmp_path/'pairs.jsonl';path.write_text(json.dumps({'review_status':'approved','signer_id':'one','recording_id':'one'})+'\n')
    with pytest.raises(ValueError,match='three independent'):poc.split(path,tmp_path/'splits')

def test_wer_distance_is_token_based():
    assert poc.distance(['A','B'],['A','C','D'])==2

@pytest.mark.parametrize("filename",["SignaCulture_mT5_LSFB_Drive.ipynb","SignaCulture_T5_Base_LSFB_Drive.ipynb"])
def test_notebook_embeds_current_helper_and_all_cells_compile(filename):
    import ast
    folder=Path(__file__).parent
    notebook=json.loads((folder/filename).read_text())
    embedded=None
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source']);tree=ast.parse(source);compile(tree,'<notebook>','exec')
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str) and node.args[0].value.startswith('"""Small,'):
                embedded=node.args[0].value
    assert embedded==(folder/'lsfb_poc.py').read_text()
