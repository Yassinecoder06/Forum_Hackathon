"""French sentence -> LSF-ordered gloss sequence, by simple published word-order rules (no model).

This is the "E" step of the LSF plan: before the B2 engine looks signs up, the French words are put in an
order closer to LSF and words LSF does not sign are dropped. The rules are the textbook tendencies of LSF:

1. time first (DEMAIN, HIER, CE-MATIN, LUNDI, "à 8 heures"...),
2. then place ("à Paris", "au musée", "dans la salle"...),
3. then the topic / subject and the object, the verb after them (S O V tendency, modal verbs after the main verb),
4. negation after the verb (PAS, JAMAIS, RIEN, PERSONNE, PLUS), "ne" dropped,
5. question words at the end (OÙ, QUAND, QUI, QUOI, COMMENT, POURQUOI, COMBIEN), "est-ce que" dropped,
6. no articles, no copula "être", no auxiliary before a past participle (a completed action gets FINI when no time
   word says when it happened), subject pronouns become MOI / TOI / IL / NOUS / VOUS / EUX.

It also returns the clause's non-manual markers (yes/no question, wh-question, negation), which the motion converter
turns into eyebrow and head movement. These are simplified rules from LSF teaching material, applied without a parser;
they have NOT been checked by an LSF reader and are no substitute for a translation.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

ARTICLES = {"le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux"}
ETRE = {"être", "suis", "es", "est", "sommes", "êtes", "sont", "était", "étais", "étaient", "étions", "étiez", "sera", "seront",
        "serai", "seras", "serons", "serez", "été", "soit", "soient", "serait", "seraient"}
AVOIR = {"ai", "as", "a", "avons", "avez", "ont", "avait", "avais", "avaient", "aura", "auront", "eu"}
SUBJECT_PRONOUNS = {"je": "MOI", "j": "MOI", "tu": "TOI", "il": "IL", "elle": "ELLE", "on": "ON", "nous": "NOUS", "vous": "VOUS",
                    "ils": "EUX", "elles": "EUX", "moi": "MOI", "toi": "TOI", "lui": "LUI", "eux": "EUX"}
CLITICS = {"me", "m", "te", "t", "se", "s", "y", "en", "lui", "leur", "nous", "vous", "le", "la", "les", "l"}
NEGATORS = {"pas": "PAS", "jamais": "JAMAIS", "rien": "RIEN", "personne": "PERSONNE", "plus": "PLUS", "aucun": "AUCUN",
            "aucune": "AUCUN"}
WH = {"où": "OÙ", "quand": "QUAND", "qui": "QUI", "quoi": "QUOI", "que": "QUOI", "qu": "QUOI", "comment": "COMMENT",
      "pourquoi": "POURQUOI", "combien": "COMBIEN", "quel": "QUEL", "quelle": "QUEL", "quels": "QUEL", "quelles": "QUEL",
      "lequel": "QUEL", "laquelle": "QUEL"}
MODALS = {"vouloir", "pouvoir", "devoir", "falloir", "aller", "savoir", "aimer", "préférer", "espérer", "venir"}
TIME_WORDS = {"aujourd'hui", "aujourd", "demain", "hier", "maintenant", "bientôt", "toujours", "souvent", "parfois", "déjà",
              "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "janvier", "février", "mars", "avril",
              "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre", "matin", "soir", "midi",
              "après-midi", "nuit", "semaine", "mois", "an", "année", "week-end", "heure", "heures", "tard", "tôt",
              "avant-hier", "après-demain", "ensuite", "puis", "désormais", "actuellement"}
TIME_LEAD = {"ce", "cet", "cette", "la", "le", "l", "à", "en", "dans", "depuis", "pendant", "chaque", "tous", "toutes",
             "prochain", "prochaine", "dernier", "dernière", "tout"}
PLACE_PREPS = {"à", "au", "aux", "en", "dans", "chez", "sur", "sous", "devant", "derrière", "près"}
PLACE_NOUNS = {"maison", "école", "musée", "ville", "gare", "bureau", "hôpital", "travail", "salle", "pays", "france", "rue",
               "cinéma", "théâtre", "festival", "restaurant", "magasin", "parc", "plage", "classe", "université", "mairie",
               "village", "région", "europe", "monde", "ministère", "palais", "élysée", "assemblée", "sénat", "port",
               "aéroport", "hôtel", "église", "marché", "bibliothèque", "quartier", "campagne", "mer", "montagne"}
PARTICIPLE_END = ("é", "ée", "és", "ées", "i", "ie", "is", "ies", "u", "ue", "us", "ues", "it", "ite", "its", "ert", "erte",
                  "ort", "orte", "int", "ainte")
DROP = ARTICLES | {"ne", "n", "est-ce", "ce"}
POSSESSIVES = {"ma": "MON", "mon": "MON", "mes": "MON", "ta": "TON", "ton": "TON", "tes": "TON", "sa": "SON", "son": "SON",
               "ses": "SON", "notre": "NOTRE", "nos": "NOTRE", "votre": "VOTRE", "vos": "VOTRE", "leur": "LEUR", "leurs": "LEUR"}


def _norm(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", w.lower()) if not unicodedata.combining(c))


def _lemma(tok: str) -> str:
    try:
        import simplemma
        return simplemma.lemmatize(tok, lang="fr")
    except Exception:  # noqa: BLE001
        return tok


def _gloss(tok: str) -> str:
    return _lemma(tok.lower()).upper() if not tok[:1].isupper() or tok.isupper() else tok.upper()


def _tokens(clause: str) -> list[str]:
    # split elisions (l'eau -> l eau, j'ai -> j ai, qu'il -> qu il) but keep "aujourd'hui"
    clause = re.sub(r"(?i)aujourd['’]hui", "aujourd'hui", clause)
    out = []
    for t in re.findall(r"aujourd'hui|[\wÀ-ÿ]+(?:-[\wÀ-ÿ]+)*['’]?|\?", clause):
        if t.endswith(("'", "’")) and t.lower() != "aujourd'hui":
            out.append(t[:-1])
        else:
            out.append(t)
    return out


def _looks_like_verb(tok: str, prev: str | None) -> bool:
    low, lem = tok.lower(), _lemma(tok.lower())
    if low in ETRE or low in AVOIR:
        return True
    if prev and prev.lower() in ARTICLES | {"ma", "mon", "mes", "ta", "ton", "tes", "sa", "son", "ses", "notre", "votre", "nos", "vos",
                                            "leur", "leurs", "ce", "cette", "ces", "cet"}:
        return False
    return lem != low and lem.endswith(("er", "ir", "re", "oir")) or low.endswith(("er", "ir", "re")) and prev and \
        prev.lower() in set(SUBJECT_PRONOUNS) | MODAL_FORMS


MODAL_FORMS = {"veux", "veut", "voulons", "voulez", "veulent", "voudrais", "peux", "peut", "pouvons", "pouvez", "peuvent",
               "dois", "doit", "devons", "devez", "doivent", "faut", "vais", "vas", "va", "allons", "allez", "vont", "sais",
               "sait", "aime", "aimes", "aimons", "aimez", "aiment"}


@dataclass
class Clause:
    glosses: list[str]
    markers: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)


def order_clause(text: str) -> Clause:
    toks = _tokens(text)
    rules: list[str] = []
    markers: list[str] = []
    question = "?" in toks
    toks = [t for t in toks if t != "?"]
    low = [t.lower() for t in toks]
    # "est-ce que" / "est-ce qu"
    joined = " ".join(low)
    if "est-ce" in low:
        i = low.index("est-ce")
        drop = 2 if i + 1 < len(low) and low[i + 1] in ("que", "qu") else 1
        toks, low = toks[:i] + toks[i + drop:], low[:i] + low[i + drop:]
        question = True
        rules.append("« est-ce que » supprimé (question)")
    # wh-words (only at clause start or after a preposition/verb in a question; "que" as conjunction is not a wh-word)
    wh = []
    for i, t in enumerate(low):
        if t in WH and (question or i == 0) and not (t in ("que", "qu") and not question):
            wh.append(i)
    # negation: ne/n' ... pas/jamais/rien/plus/personne
    neg = []
    if "ne" in low or "n" in low:
        for i, t in enumerate(low):
            if t in NEGATORS:
                neg.append(i)
                break
    # time spans
    time_idx = set()
    for i, t in enumerate(low):
        if t in TIME_WORDS or re.fullmatch(r"\d{1,2}h\d{0,2}", t) or (t.isdigit() and i + 1 < len(low) and low[i + 1].startswith("heure")):
            time_idx.add(i)
            j = i - 1
            while j >= 0 and low[j] in TIME_LEAD:
                time_idx.add(j)
                j -= 1
            k = i + 1
            while k < len(low) and low[k] in {"prochain", "prochaine", "dernier", "dernière", "matin", "soir", "midi"}:
                time_idx.add(k)
                k += 1
    # place spans: preposition + (article/possessive) + noun that is a place noun or a capitalised name
    place_idx = set()
    for i, t in enumerate(low):
        if t in PLACE_PREPS and i + 1 < len(toks) and i not in time_idx:
            j = i + 1
            while j < len(toks) and low[j] in ARTICLES | {"ma", "mon", "mes", "ta", "ton", "sa", "son", "notre", "votre", "leur", "ce",
                                                           "cette", "l"}:
                j += 1
            if j < len(toks) and (low[j] in PLACE_NOUNS or _norm(low[j]) in {_norm(p) for p in PLACE_NOUNS} or toks[j][:1].isupper()):
                place_idx.update(range(i, j + 1))
    # main verb: first verb-looking token outside time/place/neg/wh; modal + infinitive grouped
    verb_idx: list[int] = []
    for i, t in enumerate(toks):
        if i in time_idx | place_idx or i in wh or low[i] in DROP | set(NEGATORS) | set(SUBJECT_PRONOUNS) | CLITICS:
            continue
        if _looks_like_verb(t, toks[i - 1] if i else None) or low[i] in MODAL_FORMS:
            verb_idx.append(i)
            nxt = next((k for k in range(i + 1, len(toks)) if low[k] not in NEGATORS and low[k] not in CLITICS), None)
            if nxt is not None and (low[i] in MODAL_FORMS or _lemma(low[i]) in MODALS) and _looks_like_verb(toks[nxt], low[i]):
                verb_idx.append(nxt)
            break
    # auxiliary + participle: drop the auxiliary, mark completed action
    completed = False
    if verb_idx and low[verb_idx[0]] in AVOIR | ETRE:
        v = verb_idx[0]
        nxt = next((k for k in range(v + 1, len(toks)) if low[k] not in NEGATORS and low[k] not in CLITICS), None)
        if nxt is not None and low[nxt].endswith(PARTICIPLE_END) and low[nxt] not in TIME_WORDS:
            verb_idx = [nxt]
            completed = True
            rules.append("auxiliaire supprimé (passé composé)")
        elif low[v] in ETRE:
            rules.append("copule « être » supprimée")
            verb_idx = []
    subject = [i for i, t in enumerate(low) if t in SUBJECT_PRONOUNS and (not verb_idx or i < verb_idx[0]) and i not in place_idx]
    used = set(time_idx) | place_idx | set(wh) | set(neg) | set(verb_idx) | set(subject)

    def gl(idx):
        out = []
        for i in sorted(idx):
            t = low[i]
            if t in DROP and i not in time_idx:
                continue
            if t in ("à", "au", "aux", "en", "dans", "chez") and i in place_idx:
                continue                                  # the place itself is signed, the preposition usually not
            if t in TIME_LEAD and i in time_idx and t not in ("prochain", "prochaine", "dernier", "dernière", "chaque", "tous", "toutes"):
                continue
            out.append(SUBJECT_PRONOUNS.get(t) or POSSESSIVES.get(t) or _gloss(toks[i]))
        return out

    rest = [i for i in range(len(toks)) if i not in used and low[i] not in DROP and low[i] not in CLITICS and low[i] not in ETRE
            and low[i] not in ("est-ce",) and not (low[i] in AVOIR and completed)]
    verbs = [_lemma(low[i]).upper() for i in verb_idx]
    if len(verbs) == 2 and _lemma(low[verb_idx[0]]) in MODALS:
        verbs = [verbs[1], verbs[0]]                      # modal after the main verb
        rules.append("verbe modal après le verbe principal")
    glosses = gl(time_idx)
    if time_idx:
        rules.append("temps en premier")
    glosses += gl(place_idx)
    if place_idx:
        rules.append("lieu après le temps")
    glosses += gl(subject) + gl(rest) + verbs
    if verb_idx and rest:
        rules.append("verbe après l'objet")
    if completed and not time_idx:
        glosses.append("FINI")
        rules.append("FINI marque l'action accomplie")
    if neg:
        glosses.append(NEGATORS[low[neg[0]]])
        markers.append("negation")
        rules.append("négation après le verbe")
    if wh:
        glosses += [WH[low[i]] for i in wh]
        markers.append("question_wh")
        rules.append("mot interrogatif à la fin")
    elif question:
        markers.append("question_yn")
    if any(t in ARTICLES for t in low):
        rules.append("articles supprimés")
    glosses = [g for g in glosses if g and g != "?"]
    return Clause(glosses=glosses, markers=markers, rules=rules)


def order(text: str) -> dict:
    """-> {"clauses": [{"text", "glosses", "markers", "rules"}], "glosses": [...], "gloss_text": "...", "validated": False}."""
    parts = [p.strip() for p in re.split(r"(?<=[.;!])\s+|(?<=\?)\s*|,\s+(?=(?:mais|et|puis|donc|alors)\b)", text) if p and p.strip()]
    clauses = []
    for p in parts or [text]:
        c = order_clause(p)
        if c.glosses:
            clauses.append({"text": p, "glosses": c.glosses, "markers": c.markers, "rules": c.rules})
    glosses = [g for c in clauses for g in c["glosses"]]
    return {"clauses": clauses, "glosses": glosses, "gloss_text": " ".join(glosses),
            "method": "règles simplifiées d'ordre LSF (temps, lieu, sujet, objet, verbe, négation, interrogatif)",
            "validated": False}
