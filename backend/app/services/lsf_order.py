"""French sentence -> LSF-ordered gloss sequence, by simple published word-order rules (no model).

This is the "E" step of the LSF plan: before the B2 engine looks signs up, the French words are put in an
order closer to LSF and words LSF does not sign are dropped. The rules are the textbook tendencies of LSF:

1. time first (DEMAIN, HIER, CE-MATIN, LUNDI, "à 8 heures"...),
2. then place ("à Paris", "au musée", "dans la salle"...),
3. then the topic / subject and the object, the verb after them (S O V tendency, modal verbs after the main verb),
4. negation after the verb (PAS, JAMAIS, RIEN, PERSONNE, PLUS), "ne" dropped,
5. question words at the end (OÙ, QUAND, QUI, QUOI, COMMENT, POURQUOI, COMBIEN), "est-ce que" dropped,
6. no articles, no copula "être", no auxiliary before a past participle (a completed action gets FINI once per
   sentence when no time word says when it happened), subject pronouns become MOI / TOI / IL / NOUS / VOUS / EUX.

The rules apply per CLAUSE. A sentence is first cut into clauses (quote-aware):

- coordinated clauses ("... et il est entré ...", "mais", "puis", "donc", "alors", "car") become separate clauses,
  each ordered on its own, so a pronoun and a place stay in their own clause; "et" between clauses is not signed;
- direct speech (« ... », “ ... ”, "...", or "Paul dit : ...") and reported speech ("Marie répond qu'elle ...",
  "il demande si ...") are clauses of their own, marked as a ROLE SHIFT of their speaker (prise de rôle); a speech
  incise after the quote ("« Tu viens ? » demande Paul") is moved before it, so the speaker is introduced first.

A small deterministic REFERENT TABLE follows the people, places and things of the text: names, nouns introduced
with a determiner, the current speaker. Pronouns (il, elle, ils, elles, lui, eux; je/tu for the speaker and the
addressee) are resolved to the most recent compatible referent (gender and number from articles, a first-name list
and noun endings; people preferred for personal pronouns; the previous subject preferred on ties; inside a quote the
quoted speaker is never "il/elle"). Each person gets a proposed locus in signing space (L / R, in the viewer's frame,
the same as the film cast sheet), so a later pointing layer can index them. The table can be passed back in (`scene`)
so successive subtitles share referents.

It also returns the clause's non-manual markers (yes/no question, wh-question, negation), which the motion converter
turns into eyebrow and head movement. These are simplified rules from LSF teaching material, applied without a parser.
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
# Conjunctions that are not signed: "et" between clauses, "que" / "qu'" / "si" introducing a clause.
DROP = ARTICLES | {"ne", "n", "est-ce", "ce", "et"}
POSSESSIVES = {"ma": "MON", "mon": "MON", "mes": "MON", "ta": "TON", "ton": "TON", "tes": "TON", "sa": "SON", "son": "SON",
               "ses": "SON", "notre": "NOTRE", "nos": "NOTRE", "votre": "VOTRE", "vos": "VOTRE", "leur": "LEUR", "leurs": "LEUR"}
MODAL_FORMS = {"veux", "veut", "voulons", "voulez", "veulent", "voudrais", "peux", "peut", "pouvons", "pouvez", "peuvent",
               "dois", "doit", "devons", "devez", "doivent", "faut", "vais", "vas", "va", "allons", "allez", "vont", "sais",
               "sait", "aime", "aimes", "aimons", "aimez", "aiment"}
# Irregular past participles whose dictionary lemma is the adjective (ouvert -> OUVRIR, not OUVERT).
PARTICIPLE_LEMMA = {"ouvert": "ouvrir", "ouverte": "ouvrir", "ouverts": "ouvrir", "couvert": "couvrir", "offert": "offrir",
                    "découvert": "découvrir", "souffert": "souffrir", "écrit": "écrire", "écrite": "écrire", "décrit": "décrire",
                    "inscrit": "inscrire", "parti": "partir", "partie": "partir", "sorti": "sortir", "sortie": "sortir",
                    "fini": "finir", "choisi": "choisir", "réussi": "réussir", "dormi": "dormir", "servi": "servir",
                    "construit": "construire", "conduit": "conduire", "produit": "produire", "traduit": "traduire",
                    "mort": "mourir", "né": "naître", "née": "naître", "peint": "peindre", "craint": "craindre",
                    "atteint": "atteindre", "éteint": "éteindre", "plu": "plaire", "lu": "lire", "bu": "boire", "su": "savoir",
                    "pu": "pouvoir", "dû": "devoir", "cru": "croire", "connu": "connaître", "paru": "paraître", "vécu": "vivre",
                    "couru": "courir", "tenu": "tenir", "obtenu": "obtenir", "voulu": "vouloir", "fallu": "falloir"}
# Speech and thought verbs: their quote / complement clause is a role shift of their subject.
SPEECH_VERBS = {"dire", "répondre", "demander", "expliquer", "annoncer", "affirmer", "déclarer", "raconter", "ajouter",
                "promettre", "crier", "penser", "croire", "assurer", "prévenir", "rappeler", "préciser", "confirmer",
                "indiquer", "avouer", "jurer", "estimer", "répéter", "murmurer", "chuchoter", "hurler", "s'exclamer",
                "exclamer", "insister", "lancer", "souffler", "répliquer", "rétorquer", "conclure", "admettre", "reconnaître"}
_SPEECH_STEMS = tuple(sorted({v[:-2] for v in SPEECH_VERBS if v.endswith("er") and len(v) >= 7}, key=len, reverse=True))
# First names (gender) for capitalised words at the start of a clause, where capitals do not tell a name.
FIRST_NAMES = {n: "f" for n in (
    "marie anne sophie julie claire camille léa emma chloé manon sarah laura lucie inès louise alice jeanne zoé lina "
    "nathalie isabelle sandrine céline émilie catherine christine martine sylvie valérie véronique hélène élise juliette "
    "margaux pauline charlotte mathilde amélie agathe rose lou nina jade eva clara océane morgane audrey aurélie "
    "fatima aïcha amina yasmine leïla nadia samira khadija salma myriam elisa elsa louna maëlle gabrielle brigitte "
    "monique françoise nicole danielle josiane mireille chantal patricia stéphanie caroline virginie mélanie").split()}
FIRST_NAMES.update({n: "m" for n in (
    "paul pierre jean jacques louis lucas hugo léo gabriel arthur jules adam raphaël nathan thomas nicolas julien antoine "
    "alexandre maxime mathieu matthieu guillaume romain sébastien olivier philippe michel alain bernard patrick daniel "
    "françois henri marc éric eric laurent christophe david vincent yves serge didier frédéric emmanuel benjamin "
    "victor martin noah ethan louis-paul théo tom enzo samuel simon bruno gérard rené claude andré robert georges "
    "mohamed mohammed ahmed karim yassine hamza youssef omar ali mehdi rachid sami adem amine bilal ilyes "
    "adrien baptiste clément damien florian kevin quentin valentin xavier yann").split()})
PERSON_NOUNS = {
    "directeur", "directrice", "président", "présidente", "ministre", "homme", "femme", "enfant", "fille", "garçon", "fils",
    "frère", "sœur", "soeur", "mère", "père", "maman", "papa", "ami", "amie", "copain", "copine", "voisin", "voisine",
    "élève", "professeur", "prof", "maîtresse", "maître", "médecin", "docteur", "infirmier", "infirmière", "policier",
    "policière", "journaliste", "maire", "roi", "reine", "prince", "princesse", "patron", "patronne", "client", "cliente",
    "serveur", "serveuse", "chef", "collègue", "interprète", "juge", "avocat", "avocate", "soldat", "député", "députée",
    "sénateur", "sénatrice", "mari", "épouse", "femme", "bébé", "grand-mère", "grand-père", "oncle", "tante", "cousin",
    "cousine", "étudiant", "étudiante", "sourd", "sourde", "inspecteur", "inspectrice", "commissaire", "capitaine",
    "vendeur", "vendeuse", "acteur", "actrice", "chanteur", "chanteuse", "secrétaire", "porte-parole", "candidat",
    "candidate", "gens", "enfants", "parents", "habitants", "citoyens", "joueur", "joueuse", "entraîneur", "fermier",
    "boulanger", "boulangère", "facteur", "pompier", "chauffeur", "pilote", "dame", "monsieur", "madame", "garde"}
FEMININE_NOUNS = {"femme", "fille", "sœur", "soeur", "mère", "maman", "amie", "copine", "voisine", "maîtresse", "reine",
                  "princesse", "patronne", "cliente", "épouse", "tante", "cousine", "étudiante", "sourde", "dame", "madame",
                  "maison", "porte", "salle", "école", "ville", "gare", "rue", "classe", "mairie", "mer", "montagne", "voiture",
                  "lettre", "table", "chambre", "fenêtre", "clé", "eau", "loi", "réunion", "famille", "équipe", "fête",
                  "nuit", "semaine", "journée", "histoire", "question", "réponse", "boîte", "photo", "radio", "télé", "pomme"}
MASCULINE_NOUNS = {"homme", "garçon", "fils", "frère", "père", "papa", "ami", "copain", "voisin", "roi", "prince", "mari",
                   "oncle", "cousin", "monsieur", "musée", "bureau", "hôpital", "pays", "parc", "village", "livre", "projet",
                   "travail", "train", "bus", "téléphone", "film", "repas", "jardin", "magasin", "restaurant", "cinéma"}
PRENOMINAL_ADJ = {"petit", "petite", "grand", "grande", "vieux", "vieil", "vieille", "jeune", "nouveau", "nouvel", "nouvelle",
                  "bon", "bonne", "beau", "bel", "belle", "gros", "grosse", "premier", "première", "dernier", "dernière",
                  "autre", "même", "seul", "seule", "meilleur", "meilleure", "pauvre", "cher", "chère"}
DETERMINERS = {"le": ("m", "sg"), "la": ("f", "sg"), "l": (None, "sg"), "les": (None, "pl"), "un": ("m", "sg"),
               "une": ("f", "sg"), "des": (None, "pl"), "du": ("m", "sg"), "au": ("m", "sg"), "aux": (None, "pl"),
               "ce": ("m", "sg"), "cet": ("m", "sg"), "cette": ("f", "sg"), "ces": (None, "pl"), "mon": (None, "sg"),
               "ton": (None, "sg"), "son": (None, "sg"), "ma": ("f", "sg"), "ta": ("f", "sg"), "sa": ("f", "sg"),
               "mes": (None, "pl"), "tes": (None, "pl"), "ses": (None, "pl"), "notre": (None, "sg"), "votre": (None, "sg"),
               "leur": (None, "sg"), "nos": (None, "pl"), "vos": (None, "pl"), "leurs": (None, "pl")}
# Third-person pronouns -> (gender, number); None = either.
THIRD_PRONOUNS = {"il": ("m", "sg"), "elle": ("f", "sg"), "ils": ("m", "pl"), "elles": ("f", "pl"), "lui": (None, "sg"),
                  "eux": ("m", "pl")}
FIRST_PERSON = {"je", "j", "moi", "me", "m"}
SECOND_PERSON = {"tu", "toi", "te", "t"}
COORDINATORS = ("et", "mais", "puis", "donc", "alors", "car", "ensuite")
# Verbs whose passé composé takes "être" (completed action, unlike a passive "est fermé").
ETRE_VERBS = {"aller", "venir", "arriver", "partir", "entrer", "sortir", "monter", "descendre", "tomber", "rester", "naître",
              "mourir", "retourner", "devenir", "revenir", "rentrer", "passer", "repartir", "intervenir", "parvenir"}
_ABBREVIATIONS = {"m", "mme", "mlle", "dr", "st", "pr", "me"}
QUOTE_PAIRS = {"«": "»", "“": "”", "\"": "\""}
LOCI = ("L", "R")


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


def _speech_lemma(tok: str, prev: str | None = None) -> str | None:
    """Infinitive of a speech/thought verb form ("dit" -> "dire", "demande" -> "demander"), else None."""
    low = tok.lower()
    if prev and prev.lower() in ARTICLES | set(POSSESSIVES) | {"ce", "cette", "ces", "cet"}:
        return None                                        # "la demande", "sa réponse": nouns
    lem = _lemma(low)
    if lem in SPEECH_VERBS:
        return lem
    for stem in _SPEECH_STEMS:
        if low.startswith(stem) and len(low) - len(stem) <= 4:
            return stem + "er"
    return None


def _looks_like_verb(tok: str, prev: str | None) -> bool:
    low, lem = tok.lower(), _lemma(tok.lower())
    if low in ETRE or low in AVOIR:
        return True
    if prev and prev.lower() in ARTICLES | {"ma", "mon", "mes", "ta", "ton", "tes", "sa", "son", "ses", "notre", "votre", "nos", "vos",
                                            "leur", "leurs", "ce", "cette", "ces", "cet"}:
        return False
    if _speech_lemma(tok, prev):
        return True
    return lem != low and lem.endswith(("er", "ir", "re", "oir")) or low.endswith(("er", "ir", "re")) and prev and \
        prev.lower() in set(SUBJECT_PRONOUNS) | MODAL_FORMS


def _verb_lemma(low: str, after_avoir: bool) -> str:
    lem = _lemma(low)
    if after_avoir and lem == low:
        if low in PARTICIPLE_LEMMA:
            return PARTICIPLE_LEMMA[low]
        base = re.sub(r"(ée|és|ées|é)$", "er", low)
        if base != low:
            return base
    if lem == low and _speech_lemma(low):
        return _speech_lemma(low) or lem
    return lem


def _name_indices(toks: list[str]) -> set[int]:
    """Capitalised words that are names: anywhere after the first token, or a known first name / an unknown
    capitalised word followed by a verb at the start of the clause."""
    low = [t.lower() for t in toks]
    function = set(ARTICLES) | set(SUBJECT_PRONOUNS) | set(WH) | TIME_WORDS | set(POSSESSIVES) | set(DETERMINERS) | \
        {"ne", "n", "est-ce", "ce", "et", "mais", "puis", "donc", "alors", "car", "si", "oui", "non", "bonjour", "merci",
         "selon", "pour", "avec", "sans", "chez", "dans", "sur", "sous", "en", "à", "depuis", "pendant", "quand"}
    out = set()
    for i, t in enumerate(toks):
        if not t[:1].isupper() or low[i] in function:
            continue
        if i > 0:
            out.add(i)
            continue
        if low[i] in FIRST_NAMES:
            out.add(i)
        elif _lemma(low[i]) == low[i] and not low[i].endswith(("er", "ir", "oir")) and i + 1 < len(toks) and \
                (low[i + 1] in ETRE | AVOIR | MODAL_FORMS or _looks_like_verb(toks[i + 1], None)):
            out.add(i)
    return out


@dataclass
class Clause:
    glosses: list[str]
    markers: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    # token index each gloss comes from (None for added glosses: FINI, question words...)
    sources: list[int | None] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    verb_index: int | None = None
    completed: bool = False
    has_time: bool = False
    names: set[int] = field(default_factory=set)
    time_idx: set[int] = field(default_factory=set)


def order_clause(text: str, fini: bool = True) -> Clause:
    """One clause -> LSF-ordered glosses. fini=False leaves the completed-action marker to the caller."""
    toks = _tokens(text)
    rules: list[str] = []
    markers: list[str] = []
    question = "?" in toks
    toks = [t for t in toks if t != "?"]
    low = [t.lower() for t in toks]
    # "est-ce que" / "est-ce qu"
    if "est-ce" in low:
        i = low.index("est-ce")
        drop = 2 if i + 1 < len(low) and low[i + 1] in ("que", "qu") else 1
        toks, low = toks[:i] + toks[i + drop:], low[:i] + low[i + drop:]
        question = True
        rules.append("« est-ce que » supprimé (question)")
    names = _name_indices(toks)
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
            if j < len(toks) and (low[j] in PLACE_NOUNS or _norm(low[j]) in {_norm(p) for p in PLACE_NOUNS}
                                  or toks[j][:1].isupper() and low[j] not in FIRST_NAMES):          # "à Marie" is a person
                place_idx.update(range(i, j + 1))
    # main verb: first verb-looking token outside time/place/neg/wh/names; modal + infinitive grouped
    verb_idx: list[int] = []
    for i, t in enumerate(toks):
        if i in time_idx | place_idx | names or i in wh or low[i] in DROP | set(NEGATORS) | set(SUBJECT_PRONOUNS) | CLITICS:
            continue
        if _looks_like_verb(t, toks[i - 1] if i else None) or low[i] in MODAL_FORMS:
            verb_idx.append(i)
            nxt = next((k for k in range(i + 1, len(toks)) if low[k] not in NEGATORS and low[k] not in CLITICS), None)
            if nxt is not None and (low[i] in MODAL_FORMS or _lemma(low[i]) in MODALS) and _looks_like_verb(toks[nxt], low[i]):
                verb_idx.append(nxt)
            break
    # auxiliary + participle: drop the auxiliary, mark completed action
    completed = False
    after_avoir = False
    if verb_idx and low[verb_idx[0]] in AVOIR | ETRE:
        v = verb_idx[0]
        nxt = next((k for k in range(v + 1, len(toks)) if low[k] not in NEGATORS and low[k] not in CLITICS), None)
        if nxt is not None and low[nxt].endswith(PARTICIPLE_END) and low[nxt] not in TIME_WORDS and nxt not in names:
            after_avoir = low[v] in AVOIR
            verb_idx = [nxt]
            # "être" + participle is a completed action only for the verbs conjugated with "être" (il est entré);
            # otherwise it is a state or a passive (la porte est fermée): no FINI
            completed = after_avoir or _lemma(low[nxt]) in ETRE_VERBS
            rules.append("auxiliaire supprimé (passé composé)" if completed else "copule « être » supprimée")
        elif low[v] in ETRE:
            rules.append("copule « être » supprimée")
            verb_idx = []
    subject = [i for i, t in enumerate(low) if t in SUBJECT_PRONOUNS and (not verb_idx or i < verb_idx[0]) and i not in place_idx]
    used = set(time_idx) | place_idx | set(wh) | set(neg) | set(verb_idx) | set(subject)
    conj = {"que", "qu", "si"} if not question else set()

    def gl(idx) -> list[tuple[str, int | None]]:
        out = []
        for i in sorted(idx):
            t = low[i]
            if (t in DROP or t in conj) and i not in time_idx:
                continue
            if t in ("à", "au", "aux", "en", "dans", "chez") and i in place_idx:
                continue                                  # the place itself is signed, the preposition usually not
            if t in TIME_LEAD and i in time_idx and t not in ("prochain", "prochaine", "dernier", "dernière", "chaque", "tous", "toutes"):
                continue
            g = toks[i].upper() if i in names else SUBJECT_PRONOUNS.get(t) or POSSESSIVES.get(t) or _gloss(toks[i])
            out.append((g, i))
        return out

    rest = [i for i in range(len(toks)) if i not in used and low[i] not in DROP and low[i] not in conj and low[i] not in CLITICS
            and low[i] not in ETRE and low[i] not in ("est-ce",) and not (low[i] in AVOIR and completed)
            and not (low[i] == "à" and i + 1 in names)]           # "dire à Marie": the preposition is not signed
    verbs: list[tuple[str, int | None]] = [(_verb_lemma(low[i], after_avoir).upper(), i) for i in verb_idx]
    if len(verbs) == 2 and _lemma(low[verb_idx[0]]) in MODALS:
        verbs = [verbs[1], verbs[0]]                      # modal after the main verb
        rules.append("verbe modal après le verbe principal")
    # a clause-initial coordinator (mais, donc, alors, car) stays at the start of its clause
    lead: list[tuple[str, int | None]] = []
    if low and low[0] in COORDINATORS and low[0] != "et" and 0 not in time_idx:
        lead = [(low[0].upper(), 0)]
        rest = [i for i in rest if i != 0]
        subject = [i for i in subject if i != 0]
    items = lead + gl(time_idx)
    if time_idx:
        rules.append("temps en premier")
    items += gl(place_idx)
    if place_idx:
        rules.append("lieu après le temps")
    items += gl(subject) + gl(rest) + verbs
    if verb_idx and rest:
        rules.append("verbe après l'objet")
    if completed and not time_idx and fini:
        items.append(("FINI", None))
        rules.append("FINI marque l'action accomplie")
    if neg:
        items.append((NEGATORS[low[neg[0]]], neg[0]))
        markers.append("negation")
        rules.append("négation après le verbe")
    if wh:
        items += [(WH[low[i]], i) for i in wh]
        markers.append("question_wh")
        rules.append("mot interrogatif à la fin")
    elif question:
        markers.append("question_yn")
    if any(t in ARTICLES for t in low):
        rules.append("articles supprimés")
    items = [(g, s) for g, s in items if g and g != "?"]
    return Clause(glosses=[g for g, _ in items], markers=markers, rules=rules, sources=[s for _, s in items], tokens=toks,
                  verb_index=verb_idx[0] if verb_idx else None, completed=completed, has_time=bool(time_idx), names=names,
                  time_idx=time_idx)


# ---------------------------------------------------------------------------------------------------------------
# Clause segmentation: sentences, quotes, reported speech, coordination
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Unit:
    text: str
    sentence: int
    quote: str | None = None          # "quote" (direct speech) / "reported" (que/si clause) / None
    voice: int | None = None          # index of the unit whose subject speaks this unit
    rules: list[str] = field(default_factory=list)
    incise: bool = False              # speech frame verb-first ("demande Paul"): speaker after the verb


def _invert_pronoun(text: str) -> str:
    """'dit-il' -> 'il dit', 'demande-t-elle' -> 'elle demande', 'Viens-tu' -> 'tu viens'."""
    def swap(m):
        return f"{m.group(2)} {m.group(1).lower()}"
    return re.sub(r"\b([\wÀ-ÿ]+)-(?:t-)?(il|elle|on|ils|elles|je|tu|nous|vous)\b", swap, text)


def _split_sentences(text: str) -> list[str]:
    """Sentence ends (. ! ? ; and "..."), never inside a quotation, never after M. / Mme / Dr."""
    out, buf, stack, i = [], [], [], 0
    while i < len(text):
        ch = text[i]
        buf.append(ch)
        if stack and ch == stack[-1]:
            stack.pop()
        elif ch in QUOTE_PAIRS and ch not in ("»", "”"):
            stack.append(QUOTE_PAIRS[ch])
        elif not stack and ch in ".!?;…":
            j = i + 1
            while j < len(text) and text[j] in ".!?;…":
                buf.append(text[j])
                j += 1
            word = re.findall(r"[\wÀ-ÿ]+$", "".join(buf).rstrip(".!?;…"))
            abbrev = ch == "." and word and word[0].lower() in _ABBREVIATIONS
            if not abbrev and (j >= len(text) or text[j].isspace() or ch in "?!"):
                out.append("".join(buf))
                buf = []
            i = j
            continue
        i += 1
    if "".join(buf).strip():
        out.append("".join(buf))
    return [s.strip() for s in out if s.strip() and re.search(r"[\wÀ-ÿ]", s)]


def _has_verb(segment: str) -> bool:
    toks = _tokens(segment)
    toks = [t for t in toks if t != "?"]
    names = _name_indices(toks)
    for i, t in enumerate(toks):
        if i in names:
            continue
        low = t.lower()
        if low in ETRE | AVOIR | MODAL_FORMS:
            return True
        if low in SUBJECT_PRONOUNS or low in CLITICS or low in DROP or low in NEGATORS:
            continue
        if _looks_like_verb(t, toks[i - 1] if i else None):
            return True
    return False


def _coord_split(text: str) -> list[tuple[str, str | None]]:
    """Split coordinated CLAUSES (both sides have a verb): -> [(clause text, coordinator)]; "la porte et la fenêtre"
    stays whole. The coordinator is kept at the start of its clause, except "et" (not signed)."""
    cuts = []
    for m in re.finditer(r"(?:\s*,\s*|\s+)\b(" + "|".join(COORDINATORS) + r")\b\s+", text, flags=re.IGNORECASE):
        cuts.append(m)
    if not cuts:
        return [(text, None)]
    out, start, coord = [], 0, None
    for k, m in enumerate(cuts):
        left = text[start:m.start()]
        right = text[m.end(): cuts[k + 1].start() if k + 1 < len(cuts) else len(text)]
        if _has_verb(left) and _has_verb(right):
            out.append((left, coord))
            coord = m.group(1).lower()
            start = m.end() if coord == "et" else m.start(1)
    out.append((text[start:], coord))
    return [(t.strip(" ,"), c) for t, c in out if t.strip(" ,")]


def _speech_split(text: str) -> tuple[str, str, str] | None:
    """'Marie répond qu'elle a vu Paul' -> ('Marie répond', 'elle a vu Paul', 'reported');
    'Paul dit : il viendra' -> ('Paul dit', 'il viendra', 'quote')."""
    toks = list(re.finditer(r"[\wÀ-ÿ]+(?:-[\wÀ-ÿ]+)*['’]?", text))
    for k, m in enumerate(toks):
        word = m.group(0).rstrip("'’")
        prev = toks[k - 1].group(0).rstrip("'’") if k else None
        if k == 0 or not _speech_lemma(word, prev):
            continue
        tail = text[m.end():]
        colon = re.match(r"\s*:\s*(.+)$", tail, flags=re.S)
        if colon and _has_verb(colon.group(1)):
            return text[:m.end()].strip(), colon.group(1).strip(), "quote"
        conj = re.match(r"\s*(?:à\s+[\wÀ-ÿ]+\s+)?(que\s+|qu['’]\s*|si\s+|s['’](?=il))", tail, flags=re.IGNORECASE)
        if conj and _has_verb(tail[conj.end():]):
            frame = (text[:m.end()] + tail[:conj.start(1)]).strip()
            return frame, tail[conj.end():].strip(), "reported"
    return None


def _segment(text: str) -> list[Unit]:
    text = _invert_pronoun(" ".join(text.split()))
    units: list[Unit] = []

    def push(unit: Unit) -> int | None:
        if not re.search(r"[\wÀ-ÿ]", unit.text):
            return None
        units.append(unit)
        return len(units) - 1

    def add_plain(t: str, sentence: int, quote: str | None = None, voice: int | None = None, rules=None):
        speech = _speech_split(t) if quote is None else None
        if speech:
            frame, inner, kind = speech
            n = len(units)
            add_plain(frame, sentence)
            voice_at = len(units) - 1 if len(units) > n else None
            label = "citation" if kind == "quote" else "discours rapporté"
            add_plain(inner, sentence, kind, voice_at, [f"{label} : proposition à part (prise de rôle)"])
            return
        for k, (part, coord) in enumerate(_coord_split(t)):
            r = list(rules or [])
            if k and coord == "et":
                r.append("« et » entre deux propositions non signé")
            if k:
                r.append("proposition coordonnée ordonnée à part")
            push(Unit(part, sentence, quote, voice, r))

    for s_idx, sentence in enumerate(_split_sentences(text)):
        rest = sentence
        while rest:
            m = re.search(r"[«“\"]", rest)
            if not m:
                add_plain(rest.strip(" ,"), s_idx)
                break
            close = QUOTE_PAIRS[m.group(0)]
            end = rest.find(close, m.end())
            end = len(rest) if end < 0 else end
            before, inner, after = rest[:m.start()].strip(), rest[m.end():end].strip(), rest[end + 1:].strip()
            before = before.rstrip(" :,")
            # speech incise after the quote ("« ... » demande Paul", "», dit-il")
            after_clean = after.lstrip(" ,.;")
            inc = re.match(r"([\wÀ-ÿ]+(?:-[\wÀ-ÿ]+)*)", after_clean)
            incise_frame = None
            if not before and inc and _speech_lemma(inc.group(1)):
                stop = re.search(r"[.;!?]|[«“\"]", after_clean)
                incise_frame = after_clean[:stop.start()] if stop else after_clean
                after = after_clean[len(incise_frame):]
            elif not before and inc and inc.group(1).lower() in THIRD_PRONOUNS | {"je": 0, "tu": 0, "on": 0} and \
                    len(after_clean.split()) > 1 and _speech_lemma(after_clean.split()[1].strip(".,;")):
                stop = re.search(r"[.;!?]|[«“\"]", after_clean)
                incise_frame = after_clean[:stop.start()] if stop else after_clean
                after = after_clean[len(incise_frame):]
            voice = None
            if before:
                n = len(units)
                add_plain(before, s_idx)
                voice = len(units) - 1 if len(units) > n else None
            if incise_frame:
                voice = push(Unit(incise_frame.strip(" ,"), s_idx, incise=inc.group(1).lower() not in THIRD_PRONOUNS,
                                  rules=["incise placée avant la citation (le locuteur est présenté d'abord)"]))
            if inner:
                for sub in _split_sentences(inner) or [inner]:
                    add_plain(sub.strip(" ,"), s_idx, "quote", voice, ["citation : proposition à part (prise de rôle)"])
            rest = after.strip(" ,.;")
    return units


# ---------------------------------------------------------------------------------------------------------------
# Referent table
# ---------------------------------------------------------------------------------------------------------------

def _ref_id(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _norm(label)).strip("-") or "x"


def _noun_gender(low: str, det_gender: str | None) -> str | None:
    if det_gender:
        return det_gender
    if low in FEMININE_NOUNS:
        return "f"
    if low in MASCULINE_NOUNS:
        return "m"
    if low.endswith(("trice", "euse", "ière", "enne", "esse", "tion", "sion", "ure", "ette", "elle", "ance", "ence", "té")):
        return "f"
    if low.endswith(("eur", "ier", "ien", "ment", "age", "isme", "eau")):
        return "m"
    return None


class Tracker:
    """Deterministic referent table for one scene (successive subtitles may share it)."""

    def __init__(self, scene: list[dict] | None = None, speaker: str | dict | None = None, addressee: str | dict | None = None):
        self.refs: dict[str, dict] = {}
        self.clock = 0
        for r in scene or []:
            if not isinstance(r, dict) or not r.get("label"):
                continue
            rid = str(r.get("id") or _ref_id(str(r["label"])))
            self.refs[rid] = {"id": rid, "label": str(r["label"])[:40].upper(), "kind": r.get("kind") or "person",
                              "gender": r.get("gender"), "number": r.get("number") or "sg", "locus": r.get("locus"),
                              "speaker": bool(r.get("speaker")), "mentions": int(r.get("mentions") or 0),
                              "last_mention": int(r.get("last_mention") or 0), "first_clause": None, "last_clause": None}
            self.clock = max(self.clock, self.refs[rid]["last_mention"])
        self.sentence_of_mention: dict[str, int] = {}
        self.speaker = self._person(speaker, speaker=True) if speaker else None
        self.addressee = self._person(addressee, speaker=True) if addressee else None
        self.last_subject: str | None = None

    def _person(self, who: str | dict, speaker: bool = False) -> str:
        label = who.get("label") if isinstance(who, dict) else who
        label = str(label or "").strip()[:40]
        rid = _ref_id(label)
        ref = self.refs.get(rid)
        if not ref:
            ref = self.refs[rid] = {"id": rid, "label": label.upper(), "kind": "person",
                                    "gender": FIRST_NAMES.get(label.lower()), "number": "sg", "locus": None, "speaker": speaker,
                                    "mentions": 0, "last_mention": 0, "first_clause": None, "last_clause": None}
        if isinstance(who, dict) and who.get("locus"):
            ref["locus"] = who["locus"]
        ref["speaker"] = ref["speaker"] or speaker
        return rid

    def _locus(self, ref: dict):
        if ref["locus"] or ref["kind"] != "person":
            return
        taken = [r["locus"] for r in self.refs.values() if r["kind"] == "person" and r["locus"] in LOCI]
        free = [l for l in LOCI if l not in taken]
        ref["locus"] = free[0] if free else LOCI[len(taken) % 2]

    def mention(self, label: str, kind: str, gender: str | None, number: str, clause: int, sentence: int) -> tuple[str, bool]:
        rid = _ref_id(label)
        new = rid not in self.refs
        ref = self.refs.setdefault(rid, {"id": rid, "label": label.upper(), "kind": kind, "gender": gender, "number": number,
                                         "locus": None, "speaker": False, "mentions": 0, "last_mention": 0,
                                         "first_clause": clause, "last_clause": clause})
        if ref["gender"] is None and gender:
            ref["gender"] = gender
        if ref["kind"] == "thing" and kind != "thing":
            ref["kind"] = kind
        new = new or ref["mentions"] == 0
        self.touch(rid, clause, sentence)
        self._locus(ref)
        return rid, new

    def touch(self, rid: str, clause: int, sentence: int):
        ref = self.refs[rid]
        self.clock += 1
        ref["mentions"] += 1
        ref["last_mention"] = self.clock
        if ref["first_clause"] is None:
            ref["first_clause"] = clause
        ref["last_clause"] = clause
        self.sentence_of_mention[rid] = sentence

    def resolve(self, pronoun: str, exclude: set[str], sentence: int) -> str | None:
        gender, number = THIRD_PRONOUNS[pronoun]
        cands = [r for r in self.refs.values() if r["mentions"] > 0 and r["id"] not in exclude
                 and (gender is None or r["gender"] in (None, gender))
                 and (r["number"] == number or (number == "pl" and r["kind"] == "group"))]
        if not cands:
            return None
        recent = [r for r in cands if self.sentence_of_mention.get(r["id"], -10) >= sentence - 1]
        pool = recent or cands
        best = max(pool, key=lambda r: (r["kind"] == "person", r["id"] == self.last_subject, r["last_mention"]))
        return best["id"]

    def table(self) -> list[dict]:
        rows = sorted(self.refs.values(), key=lambda r: (r["first_clause"] is None, r["first_clause"] or 0, r["id"]))
        return [dict(r) for r in rows]


def _track_clause(tr: Tracker, c: Clause, k: int, unit: Unit, sentence: int, voice: str | None, addressee: str | None) -> dict:
    """Mentions and pronoun links of one clause -> events; returns {'events', 'subject', 'addressee'}."""
    toks, low = c.tokens, [t.lower() for t in c.tokens]
    gloss_at = {s: g for g, s in enumerate(c.sources) if s is not None}
    events, mentions = [], []                        # mentions: (token index, ref id)
    skip: set[int] = set()
    for i, t in enumerate(toks):
        if i in skip:
            continue
        if i in c.names:
            j = i
            while j + 1 < len(toks) and j + 1 in c.names:
                j += 1
            label = " ".join(toks[i:j + 1])
            skip.update(range(i, j + 1))
            prev = low[i - 1] if i else ""
            kind = "place" if prev in {"à", "en", "au", "aux", "vers", "dans"} and low[i] not in FIRST_NAMES else "person"
            rid, new = tr.mention(label, kind, FIRST_NAMES.get(low[i]) if kind == "person" else None, "sg", k, sentence)
            events.append({"type": "introduce" if new else "mention", "referent": rid, "token": label,
                           "gloss_index": gloss_at.get(i)})
            mentions.append((i, rid))
            continue
        if low[i] in THIRD_PRONOUNS:
            exclude = {voice} if (unit.quote == "quote" and voice) else set()
            # "lui" right before a verb is the dative clitic (him OR her)
            rid = tr.resolve(low[i], exclude, sentence)
            if rid:
                tr.touch(rid, k, sentence)
                mentions.append((i, rid))
            events.append({"type": "index", "referent": rid, "token": t, "gloss_index": gloss_at.get(i)})
            continue
        if low[i] in FIRST_PERSON and voice and not (low[i] in {"me", "m"} and i + 1 < len(toks) and low[i + 1] in ETRE):
            tr.touch(voice, k, sentence)
            mentions.append((i, voice))
            events.append({"type": "index", "referent": voice, "token": t, "gloss_index": gloss_at.get(i), "person": 1})
            continue
        if low[i] in SECOND_PERSON and addressee:
            tr.touch(addressee, k, sentence)
            mentions.append((i, addressee))
            events.append({"type": "index", "referent": addressee, "token": t, "gloss_index": gloss_at.get(i), "person": 2})
            continue
        if low[i] in DETERMINERS and i + 1 < len(toks) and i not in c.time_idx:
            prev = low[i - 1] if i else ""
            if prev in SUBJECT_PRONOUNS or prev in {"ne", "n", "me", "te", "se", "m", "t", "s"}:
                continue                                    # object clitic ("il le voit"), not a determiner
            j = i + 1
            while j < len(toks) and low[j] in PRENOMINAL_ADJ:
                j += 1
            if j >= len(toks) or j in c.time_idx or j in c.names or low[j] in ETRE | AVOIR | MODAL_FORMS or \
                    low[j] in SUBJECT_PRONOUNS or low[j] in NEGATORS or low[j] in WH or low[j] in DETERMINERS or \
                    not re.fullmatch(r"[a-zà-ÿœ-]+", low[j]) or (j == c.verb_index):
                continue
            det_gender, number = DETERMINERS[low[i]]
            noun = low[j]
            if number == "pl" or noun.endswith(("s", "x")) and noun not in PERSON_NOUNS | PLACE_NOUNS:
                lem = _lemma(noun)
                noun = lem if lem in PERSON_NOUNS | PLACE_NOUNS or number == "pl" else noun
            kind = "person" if noun in PERSON_NOUNS else "place" if noun in PLACE_NOUNS or _norm(noun) in {_norm(p) for p in PLACE_NOUNS} \
                else "thing"
            if number == "pl" and kind == "person":
                kind = "group"
            rid, new = tr.mention(noun, kind, _noun_gender(noun, det_gender), number, k, sentence)
            skip.update(range(i, j + 1))
            events.append({"type": "introduce" if new else "mention", "referent": rid, "token": " ".join(toks[i:j + 1]),
                           "gloss_index": gloss_at.get(j)})
            mentions.append((j, rid))
    # subject: first mention before the verb (verb-first speech incise "demande Paul": first mention after it)
    v = c.verb_index if c.verb_index is not None else len(toks)
    before = [rid for i, rid in mentions if i < v]
    after = [(i, rid) for i, rid in mentions if i > v and not (i > 0 and low[i - 1] == "à")]
    subject = before[0] if before else (after[0][1] if unit.incise and after else None)
    if subject is None and any(t in FIRST_PERSON for t in low[:v]) and voice:
        subject = voice
    to = next((rid for i, rid in mentions if i > 0 and low[i - 1] == "à" and tr.refs[rid]["kind"] in ("person", "group")), None)
    if subject:
        tr.last_subject = subject
    return {"events": events, "subject": subject, "addressee": to}


def order(text: str, scene: list[dict] | None = None, speaker: str | dict | None = None,
          addressee: str | dict | None = None) -> dict:
    """-> {"clauses": [{"text", "glosses", "markers", "rules", "role_shift", "voice", "events"}], "glosses": [...],
    "gloss_text": "...", "referents": [...], "validated": False}.

    scene: the referent table of the previous subtitles (same scene); speaker / addressee: who says this text
    (film cast sheet: a label, or {"label", "locus"}), so "je" / "tu" resolve to them."""
    tr = Tracker(scene, speaker, addressee)
    units = _segment(text)
    parsed: list[tuple[Unit, Clause]] = [(u, order_clause(u.text, fini=False)) for u in units]
    # FINI once per sentence (and per quotation), only when no time word in that stretch says when it happened
    groups: dict[tuple[int, str | None], list[int]] = {}
    for k, (u, c) in enumerate(parsed):
        groups.setdefault((u.sentence, "q" if u.quote else None), []).append(k)
    for ks in groups.values():
        if any(parsed[k][1].has_time for k in ks):
            continue
        first = next((k for k in ks if parsed[k][1].completed), None)
        if first is not None:
            c = parsed[first][1]
            # FINI right after the verb, before the negation and the question word
            pos = next((n for n, s in enumerate(c.sources) if s is not None and
                        (c.tokens[s].lower() in NEGATORS or ("question_wh" in c.markers and c.tokens[s].lower() in WH))), len(c.glosses))
            c.glosses.insert(pos, "FINI")
            c.sources.insert(pos, None)
            c.rules.append("FINI marque l'action accomplie")
    clauses, subjects, addressees = [], {}, {}
    current = tr.speaker
    for k, (u, c) in enumerate(parsed):
        voice = current
        role_shift = None
        if u.quote and u.voice is not None and subjects.get(u.voice):
            who = subjects[u.voice]
            if who != current and tr.refs[who]["kind"] in ("person", "group"):
                role_shift = {"referent": who, "label": tr.refs[who]["label"], "kind": u.quote,
                              "locus": tr.refs[who]["locus"]}
            voice = who
        info = _track_clause(tr, c, k, u, u.sentence, voice, addressees.get(u.voice) if u.quote else tr.addressee)
        subjects[k], addressees[k] = info["subject"], info["addressee"]
        events = info["events"]
        if role_shift:
            events.insert(0, {"type": "role_shift", "referent": role_shift["referent"], "kind": role_shift["kind"],
                              "locus": role_shift["locus"], "gloss_index": 0})
            c.rules.append(f"prise de rôle : {role_shift['label']}")
        if c.glosses:
            clauses.append({"text": u.text, "glosses": c.glosses, "markers": c.markers, "rules": u.rules + c.rules,
                            "role_shift": role_shift, "voice": voice, "events": events})
    # role-shift locus is the referent's final locus (it may be assigned on its first mention)
    for cl in clauses:
        if cl["role_shift"]:
            cl["role_shift"]["locus"] = tr.refs[cl["role_shift"]["referent"]]["locus"]
            cl["events"][0]["locus"] = cl["role_shift"]["locus"]
    glosses = [g for c in clauses for g in c["glosses"]]
    return {"clauses": clauses, "glosses": glosses, "gloss_text": " ".join(glosses), "referents": tr.table(),
            "method": "règles simplifiées d'ordre LSF (propositions, temps, lieu, sujet, objet, verbe, négation, interrogatif, "
                      "prise de rôle, référents)",
            "validated": False}


def clause_times(analysis: dict, sign_spans: list[dict] | None, n_frames: int, fps: float = 25.0) -> list[tuple[float, float] | None]:
    """Time span (ms) of each clause in the generated clip: B2 from its first to last sign; otherwise the clip is split
    between clauses in proportion to their text length (C0 has no sign boundaries). Same split as the non-manual spans."""
    clauses = analysis.get("clauses") or []
    if not clauses:
        return []
    if sign_spans:
        out = []
        for k in range(len(clauses)):
            mine = [sp for sp in sign_spans if sp["clause"] == k]
            out.append((mine[0]["start_frame"] / fps * 1000.0, mine[-1]["end_frame"] / fps * 1000.0) if mine else None)
        return out
    total_ms = (n_frames - 1) / fps * 1000.0
    lengths = [max(1, len(c["text"])) for c in clauses]
    head = tail = min(300.0, 0.1 * total_ms)
    usable, t, out = max(1.0, total_ms - head - tail), head, []
    for n in lengths:
        d = usable * n / sum(lengths)
        out.append((t, t + d))
        t += d
    return out


def role_shift_spans(analysis: dict, sign_spans: list[dict] | None, n_frames: int, fps: float = 25.0) -> list[dict]:
    """Clauses spoken by a quoted / reported speaker -> [{start_ms, end_ms, referent, label, locus, kind}] for the
    avatar's role-shift layer (torso, head and gaze toward the speaker's locus)."""
    out = []
    for c, span in zip(analysis.get("clauses") or [], clause_times(analysis, sign_spans, n_frames, fps)):
        rs = c.get("role_shift")
        if rs and span:
            out.append({"start_ms": round(span[0]), "end_ms": round(span[1]), "referent": rs["referent"], "label": rs["label"],
                        "locus": rs.get("locus"), "kind": rs["kind"]})
    return out
