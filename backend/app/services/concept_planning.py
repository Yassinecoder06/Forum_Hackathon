"""Bounded local semantic proposals, separate from linguistic approval."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
from threading import BoundedSemaphore
from .capture_library import ROOT

LOCK=BoundedSemaphore(1)

def runtime():
    owner=ROOT/'.worktrees/wt-codex2'
    script=ROOT/'training/concept_planner.py'
    python=owner/'training/.venv-train/bin/python'
    manifest=ROOT/'research/concept_planner_runtime.json'
    if not manifest.is_file():
        raise FileNotFoundError('Le modèle de proposition local n’est pas encore disponible. Vous pouvez choisir les concepts manuellement.')
    try:
        config=json.loads(manifest.read_text())
        weights=Path(config['model_path']).expanduser().resolve()
    except (ValueError,KeyError,OSError): raise FileNotFoundError('Configuration locale du modèle absente.')
    if not script.is_file() or not python.is_file() or not weights.exists():
        raise FileNotFoundError('Le modèle de proposition local n’est pas installé. Vous pouvez choisir les concepts manuellement.')
    return python,script,weights

def propose(text,vocabulary):
    python,script,weights=runtime()
    if not LOCK.acquire(blocking=False): raise BlockingIOError('Une proposition est déjà en cours.')
    try:
        with tempfile.TemporaryDirectory(prefix='signaculture-concepts-') as folder:
            vocab=Path(folder)/'vocabulary.json'
            vocab.write_text(json.dumps(vocabulary,ensure_ascii=False))
            result=subprocess.run([str(python),str(script),'--text',text,'--vocabulary',str(vocab),'--model',str(weights)],capture_output=True,text=True,timeout=270,cwd=ROOT)
        if result.returncode: raise RuntimeError('La proposition locale a échoué. La sélection manuelle reste disponible.')
        try: output=json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError,IndexError): raise RuntimeError('La réponse du modèle est illisible.')
        ids=output.get('concept_ids');allowed={c['id'] for c in vocabulary}
        if not isinstance(ids,list) or len(ids)>40 or any(not isinstance(i,str) or i not in allowed for i in ids):
            raise RuntimeError('Le modèle a proposé un concept absent de la bibliothèque.')
        unsupported=output.get('unsupported',[])
        if not isinstance(unsupported,list) or any(not isinstance(s,str) for s in unsupported): raise RuntimeError('La réponse du modèle est invalide.')
        # A draft must never silently erase meaning carried by unsupported grammar.
        guards=[]
        if re.search(r'\b(ne|pas|jamais|aucun|sans|ni)\b',text,re.I): guards.append('Négation : traitement grammatical LSF à revoir')
        if '?' in text: guards.append('Question : marqueurs non manuels à revoir')
        if re.search(r'\d',text): guards.append('Nombres : vérifier leur représentation')
        from .text_lookup import model
        nlp=model()
        if nlp is not None:
            names=[e.text for e in nlp(text).ents if e.label_ in {'PER','PERSON','ORG','LOC','GPE'}]
            guards.extend('Nom propre : '+name+' — représentation à revoir' for name in names)
        output.update(unsupported=list(dict.fromkeys(unsupported+guards)),candidate=True,translation_validated=False,linguistic_review='pending',render_allowed_for_publication=False)
        return output
    finally: LOCK.release()
