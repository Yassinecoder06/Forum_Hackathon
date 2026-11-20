"""Controlled lookup with optional installed French spaCy model; never downloads."""
from functools import lru_cache
import os
import re
import simplemma

TOKEN = re.compile(r'[^\W_]+', re.UNICODE)
ELISION = {'j': 'je', 't': 'te', 'm': 'me', 's': 'se', 'n': 'ne', 'c': 'ce', 'qu': 'que', 'jusqu': 'jusque', 'lorsqu': 'lorsque'}

@lru_cache(maxsize=4)
def load_model(name):
    if not name:
        return None
    try:
        import spacy
        model = spacy.load(name)
        return model if model.lang == 'fr' and 'lemmatizer' in model.pipe_names else None
    except (ImportError, OSError, ValueError):
        return None

def model():
    return load_model(os.getenv('SIGNACULTURE_SPACY_MODEL', 'fr_core_news_sm'))

def mode():
    return 'spacy-fr-lemmatizer' if model() is not None else 'unicode-regex-fallback-no-lemmatizer'

def tokenize(text):
    spans = list(TOKEN.finditer(text))
    tokens = [m.group() for m in spans]
    lemmas = [t.casefold() for t in tokens]
    nlp = model()
    if nlp is not None:
        doc = nlp(text)
        for i, span in enumerate(spans):
            aligned = [t for t in doc if t.idx == span.start() and t.idx+len(t.text) == span.end()]
            if len(aligned) == 1 and aligned[0].lemma_:
                lemmas[i] = aligned[0].lemma_.casefold()
    # French elision: "j'aime" tokenizes as "j" + "aime"; restore the full word
    # so it can match its dictionary entry (j' -> je, t' -> te, qu' -> que ...).
    for i, span in enumerate(spans):
        surface = tokens[i].casefold()
        elided = text[span.end():span.end() + 1] in ("'", "’")
        if elided and surface in ELISION and lemmas[i] == surface:
            lemmas[i] = ELISION[surface]
    return tokens, lemmas

@lru_cache(maxsize=4096)
def phrase_tokens(phrase, model_name):
    tokens, lemmas = tokenize(phrase)
    return tuple(tokens), tuple(lemmas)

def lookup_text(text, entries):
    tokens, lemmas = tokenize(text)
    surfaces = [t.casefold() for t in tokens]
    dictionary_lemmas = [simplemma.lemmatize(t, lang='fr') for t in surfaces]
    candidates = []
    for entry in entries:
        for phrase in [entry.concept_fr, *entry.aliases]:
            ts, ls = phrase_tokens(phrase, os.getenv('SIGNACULTURE_SPACY_MODEL', 'fr_core_news_sm'))
            if ts:
                candidates.append(([t.casefold() for t in ts], ls, entry))
    sequence, spans, unknowns, i = [], [], [], 0
    while i < len(tokens):
        matches = [(len(ts), entry) for ts, ls, entry in candidates
                   if surfaces[i:i+len(ts)] == ts or tuple(lemmas[i:i+len(ls)]) == ls]
        # Dictionary recovery only when contextual lookup found nothing. Never
        # override an exact match or resolve an existing ambiguity by guessing.
        if not matches:
            matches = [(len(ts), entry) for ts, ls, entry in candidates
                       if dictionary_lemmas[i:i+len(ts)] == ts]
        if matches:
            size = max(n for n, _ in matches)
            best = {g.gloss_id:g for n,g in matches if n == size}
            if len(best) == 1:
                sequence.append(next(iter(best.values())))
                spans.append((i, i + size))
                i += size
                continue
        unknowns.append(tokens[i])
        i += 1
    return dict(source_text=text, tokens=tokens, lemmas=lemmas, gloss_sequence=sequence,
                gloss_token_spans=spans, unknowns=unknowns, linguistic_review='pending', translation_validated=False)
