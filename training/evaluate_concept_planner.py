"""One fixed software benchmark pass. No training, prompt search or retries."""
import collections,json,statistics,subprocess,threading,time
from pathlib import Path
from catalog import ROOT,sha,dump
from concept_planner import ConceptPlanner,load_vocabulary,SYSTEM

class Monitor:
    def __init__(self):self.samples=[];self.stop=threading.Event()
    def run(self):
        while not self.stop.is_set():
            try:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
                procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
                own=[]
                for p in procs:
                    pid,mem=[s.strip() for s in p.split(',')]
                    cmd=Path('/proc/'+pid+'/cmdline').read_bytes().decode(errors='replace')
                    if '18195' in cmd and 'Qwen3.5-9B-UD-Q6_K_XL.gguf' in cmd:own.append(int(mem))
                self.samples.append(dict(elapsed=time.monotonic(),device_mib=int(gpu[0]),utilization=int(gpu[1]),own_server_mib=sum(own)))
            except Exception:pass
            self.stop.wait(.5)

def score(cases,outputs):
    exact=[];tp=fp=fn=0;flag_hits=flag_total=unknown_hits=unknown_total=0;lat=[];bad=0
    for c,r in zip(cases,outputs):
        expected=c['expected_concept_ids'];actual=r['concept_ids'];exact.append(actual==expected)
        a,e=collections.Counter(actual),collections.Counter(expected)
        tp+=sum((a&e).values());fp+=sum((a-e).values());fn+=sum((e-a).values())
        lat.append(r['latency_seconds']);bad+=r.get('status')=='abstained'
        for flag in c['required_warning_codes']:
            flag_total+=1;flag_hits+=bool(flag in r['warnings'] and r['unsupported'])
        unsupported=' '.join(r['unsupported']).casefold()
        for term in c['required_unsupported_terms']:
            unknown_total+=1;unknown_hits+=term.casefold() in unsupported
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return dict(examples=len(cases),ordered_exact=sum(exact)/len(exact),micro_precision=precision,micro_recall=recall,micro_f1=2*precision*recall/max(1e-9,precision+recall),critical_flag_recall=flag_hits/max(1,flag_total),critical_flag_hits=flag_hits,critical_flag_total=flag_total,unknown_term_coverage=unknown_hits/max(1,unknown_total),abstentions=bad,latency_mean_seconds=statistics.mean(lat),latency_median_seconds=statistics.median(lat),latency_max_seconds=max(lat),latency_first_seconds=lat[0],per_category_ordered_exact={category:sum(ok for c,ok in zip(cases,exact) if c['category']==category)/sum(c['category']==category for c in cases) for category in sorted({c['category'] for c in cases})})

def main():
    folder=ROOT/'research/concept_benchmark';manifest=json.loads((folder/'manifest.json').read_text())
    for f,h in manifest['files'].items():assert sha(folder/f)==h
    runtime=json.loads((ROOT/'research/concept_planner_runtime.json').read_text())
    assert sha(ROOT/'training/concept_planner.py')==runtime['code_sha256']
    assert sha(folder/'manifest.json')==runtime['benchmark_manifest_sha256']
    out=ROOT/'research/runs/pretrained-concept-planner';out.mkdir(exist_ok=False)
    cases=json.loads((folder/'benchmark.json').read_text())['cases'];vocab=load_vocabulary(folder/'vocabulary.json')
    dump(out/'evaluation_lock.json',dict(runtime=runtime,benchmark_sha256=sha(folder/'benchmark.json'),vocabulary_sha256=sha(folder/'vocabulary.json'),planner_sha256=sha(ROOT/'training/concept_planner.py'),literal_snapshot_sha256=sha(ROOT/'training/literal_lookup_snapshot.py'),evaluator_sha256=sha(Path(__file__)),policy='ONE call per benchmark case; no retry/prompt/model/settings changes after results; no STK examples or LSF evaluation'))
    lead='/home/jmal/Projects/Active/Forum_Hackathon/.venv/bin/python'
    baseline=subprocess.run([lead,str(ROOT/'training/run_literal_concept_baseline.py')],input=json.dumps(dict(cases=[{'case_id':c['case_id'],'text':c['text']} for c in cases],vocabulary=vocab),ensure_ascii=False),capture_output=True,text=True,check=True)
    b=json.loads(baseline.stdout);dump(out/'literal_baseline.json',b)
    planner=ConceptPlanner(folder/'vocabulary.json',runtime['model_path'],device=runtime['device'])
    monitor=Monitor();thread=threading.Thread(target=monitor.run,daemon=True);thread.start();start=time.monotonic();results=[]
    try:
        for c in cases:
            r=planner.plan(c['text']);results.append(r)
            with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(dict(case=c,model=r,baseline_raw=b['results'][len(results)-1]['raw'],baseline_shared_guards=b['results'][len(results)-1]['shared_guards']),ensure_ascii=False)+'\n')
            print(json.dumps(dict(completed=len(results),case_id=c['case_id'],seconds=r['latency_seconds'],status=r['status'])),flush=True)
    finally:monitor.stop.set();thread.join()
    allowed={r['id'] for r in vocab};assert all(set(r['concept_ids'])<=allowed for r in results)
    metrics={name:score(cases,rs) for name,rs in {'model':results,'literal_existing':[r['raw'] for r in b['results']],'literal_shared_guards':[r['shared_guards'] for r in b['results']]}.items()}
    useful=(metrics['model']['ordered_exact']>metrics['literal_shared_guards']['ordered_exact'] and metrics['model']['critical_flag_recall']==1 and metrics['model']['unknown_term_coverage']==1 and not metrics['model']['abstentions'])
    report=dict(kind='SOFTWARE_CONCEPT_PLANNING_NOT_LSF_TRANSLATION',runtime=runtime,metrics=metrics,model_inference_calls=len(results),vocabulary_count=len(vocab),invalid_external_ids=0,total_seconds=time.monotonic()-start,device_peak_mib=max((s['device_mib'] for s in monitor.samples),default=None),own_server_peak_mib=max((s['own_server_mib'] for s in monitor.samples),default=None),gpu_samples=monitor.samples,measured_useful_against_shared_guard_baseline=useful,integration_decision='Candidate for lead review only' if useful else 'Do not integrate: fixed benchmark did not meet the predeclared usefulness checks',limitations=['Authored software expectations, not LSF ground truth or STK test','Only24cases; no general French or signing quality claim','Deterministic linguistic guards are incomplete on unrestricted language; manual review always required','Model/guard failures preserved; no tuning on benchmark results'])
    dump(out/'metrics.json',report);print(json.dumps(report['metrics'],ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
