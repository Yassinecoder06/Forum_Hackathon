"""One locked official-test evaluation after validation checkpoint selection."""
import argparse,hashlib,json,time,collections
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from prepare import records_from_tar,candidates,REVISION,dump
from metrics import unpack,distance

def evaluate(run,data):
    assert (run/'TRAINING_COMPLETE.json').is_file() and (run/'selection_complete.json').is_file()
    config=json.loads((run/'RUN_PLAN.json').read_text());selection=json.loads((run/'selection_complete.json').read_text())
    if (run/'TEST_COMPLETE.json').exists():return json.loads((run/'test_results.json').read_text())
    dump(run/'TEST_PROTOCOL.json',{'source':'ppoitier/lsfb-cont','revision':REVISION,'official_shard':0,'selection':selection,'max_source_tokens':256,'max_target_units':96,'max_new_tokens':128,'beams':1,'precision':'float32','no_test_vocabulary_additions':True,'unknown_references':'remain full errors; never dropped','novel_text_subset':'reported alongside full eligible official test','linguistic_approval':False})
    path=Path(hf_hub_download('ppoitier/lsfb-cont','shards/annotated/shard_000000.tar',repo_type='dataset',revision=REVISION))
    records=records_from_tar(path);rows=candidates(records,source='shard_000000.tar')
    torch.set_num_threads(2);tok=AutoTokenizer.from_pretrained(run/'final',use_fast=False)
    model=AutoModelForSeq2SeqLM.from_pretrained(run/'final').to('cuda' if torch.cuda.is_available() else 'cpu').eval();model.config.use_cache=True
    vocab=json.loads((run/'unit_vocabulary.json').read_text());reverse={int(i):u for u,i in vocab['unit_to_id'].items()}
    train=[json.loads(l) for l in (data/'train.jsonl').read_text().splitlines()];val=[json.loads(l) for l in (data/'val.jsonl').read_text().splitlines()]
    seen_recordings={r['recording_id'] for r in train+val};test_recordings={r['recording_id'] for r in rows}
    if seen_recordings&test_recordings:raise ValueError('Official test overlaps a train/validation recording; evaluation stopped')
    eligible=[];excluded=[]
    for r in rows:
        n=len(tok('translate French to LSFB glosses: '+r['french_text'])['input_ids'])
        if n>256 or len(r['units'])>96:excluded.append({'id':r['id'],'source_tokens':n,'target_units':len(r['units'])})
        else:eligible.append(r)
    assert eligible
    all_predictions=[];started=time.monotonic()
    for begin in range(0,len(eligible),16):
        part=eligible[begin:begin+16];inputs=tok(['translate French to LSFB glosses: '+r['french_text'] for r in part],padding=True,return_tensors='pt',return_token_type_ids=False).to(model.device)
        for constrained in [True,False]:
            options={'prefix_allowed_tokens_fn':lambda batch_id,sequence:list(reverse)+[tok.eos_token_id]} if constrained else {}
            with torch.inference_mode():output=model.generate(**inputs,num_beams=1,max_new_tokens=128,use_cache=True,**options)
            for r,ids in zip(part,output.cpu().tolist()):
                units,invalid,eos=unpack(ids,reverse,tok.eos_token_id,tok.pad_token_id)
                all_predictions.append({'id':r['id'],'french_text':r['french_text'],'reference':r['units'],'prediction':units,'constrained':constrained,'invalid_tokens':invalid,'eos':eos,'signer_id':r['signer_id']})
        dump(run/'test_predictions.partial.json',all_predictions)
        print(json.dumps({'stage':'test','completed':min(begin+16,len(eligible)),'total':len(eligible),'seconds':time.monotonic()-started}),flush=True)
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=True)
    train_vectors=vectorizer.fit_transform([r['french_text'] for r in train]);test_vectors=vectorizer.transform([r['french_text'] for r in eligible]);indexes=np.asarray((test_vectors@train_vectors.T).argmax(axis=1)).ravel()
    baseline=[train[int(i)]['units'] for i in indexes]
    def score(predictions,refs,valid=None):
        return {'exact_match':sum(p==r and (valid is None or valid[i]) for i,(p,r) in enumerate(zip(predictions,refs)))/len(refs),'wer':sum(distance(r,p) for r,p in zip(refs,predictions))/max(1,sum(map(len,refs))),'examples':len(refs)}
    refs=[r['units'] for r in eligible];result={}
    for constrained in [True,False]:
        part=[r for r in all_predictions if r['constrained']==constrained]
        result['constrained' if constrained else 'raw']={**score([r['prediction'] for r in part],refs,[r['eos'] and not r['invalid_tokens'] for r in part]),'empty_rate':sum(not r['prediction'] for r in part)/len(part),'invalid_sequence_rate':sum(bool(r['invalid_tokens']) for r in part)/len(part),'eos_rate':sum(r['eos'] for r in part)/len(part)}
    result['train_only_char_tfidf_retrieval']=score(baseline,refs)
    train_inputs={r['french_text'].casefold() for r in train};novel=[i for i,r in enumerate(eligible) if r['french_text'].casefold() not in train_inputs]
    part=[r for r in all_predictions if r['constrained']]
    result['novel_french_text_subset']=score([part[i]['prediction'] for i in novel],[refs[i] for i in novel],[part[i]['eos'] and not part[i]['invalid_tokens'] for i in novel]) if novel else None
    total=sum(map(len,refs));known=set(reverse.values())
    result['dataset']={'source_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'source_candidates':len(rows),'eligible':len(eligible),'length_excluded':len(excluded),'inventory_coverage':sum(u in known for units in refs for u in units)/max(1,total),'signer_overlap_train':sorted({r['signer_id'] for r in eligible}&{r['signer_id'] for r in train}),'recording_disjoint':True,'test_vocabulary_added':False,'target_policy':'author lexical lemma plus retained nonlexical labels; raw morphology preserved in source snapshots','linguistic_approval':False}
    model_path=run/'final/model.safetensors';checksum=hashlib.file_digest(model_path.open('rb'),'sha256').hexdigest();best=Path(selection['best_checkpoint'])/'model.safetensors'
    assert checksum==hashlib.file_digest(best.open('rb'),'sha256').hexdigest()
    dump(run/'MODEL_VERIFIED.json',{'drive_model':str(model_path),'sha256':checksum,'bytes':model_path.stat().st_size,'matches_validation_selected_checkpoint':True})
    dump(run/'test_predictions.json',all_predictions);dump(run/'test_length_exclusions.json',excluded);dump(run/'test_results.json',result);dump(run/'TEST_COMPLETE.json',{'selected_by_validation':True,'official_test_shard':0,'model_sha256':checksum,'seconds':time.monotonic()-started})
    print(json.dumps(result),flush=True);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('data',type=Path);a=p.parse_args();evaluate(a.run,a.data)
