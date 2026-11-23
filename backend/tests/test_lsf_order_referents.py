"""Clause segmentation (coordination, quotes, reported speech), role-shift triggers and the referent table of lsf_order.

Expected orders follow the simplified rules of the module; the role-shift and pronoun conventions are design choices
(film-mode speaker work), to be confirmed by an LSF reader."""
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.services import lsf_order as O
from backend.tests.test_lsf_pose_convert import fixture

client = TestClient(main.app)


def glosses(r):
    return [c["glosses"] for c in r["clauses"]]


def index_of(clause, token):
    return next(e["referent"] for e in clause["events"] if e["type"] == "index" and e["token"].lower() == token)


def test_coordinated_clauses_keep_their_pronoun_and_place():
    r = O.order("Le directeur a ouvert la porte et il est entré dans la salle.")
    assert glosses(r) == [["DIRECTEUR", "PORTE", "OUVRIR", "FINI"], ["SALLE", "IL", "ENTRER"]]
    assert "ET" not in r["glosses"]
    assert index_of(r["clauses"][1], "il") == "directeur"
    assert {x["id"]: x["kind"] for x in r["referents"]} == {"directeur": "person", "porte": "thing", "salle": "place"}


def test_noun_coordination_is_not_split_and_passive_gets_no_fini():
    r = O.order("La porte et la fenêtre sont fermées.")
    assert glosses(r) == [["PORTE", "FENÊTRE", "FERMÉ"]]


def test_fini_once_per_sentence_and_never_with_a_time_word():
    r = O.order("Le directeur a ouvert la porte et il a allumé la lumière.")
    assert r["glosses"].count("FINI") == 1 and r["clauses"][0]["glosses"][-1] == "FINI"
    assert "FINI" not in O.order("Hier, j'ai mangé et j'ai bu.")["glosses"]


def test_mais_starts_its_clause_and_gender_picks_the_antecedent():
    r = O.order("Sophie a appelé son frère mais il n'a pas répondu.")
    assert glosses(r)[1] == ["MAIS", "IL", "RÉPONDRE", "PAS"]
    assert r["clauses"][1]["markers"] == ["negation"]
    assert index_of(r["clauses"][1], "il") == "frere"


def test_people_are_preferred_for_personal_pronouns():
    r = O.order("La ministre a visité l'école et elle a rencontré les élèves.")
    assert index_of(r["clauses"][1], "elle") == "ministre"


def test_direct_quote_is_a_role_shift_of_its_speaker():
    r = O.order("Paul dit : « Il ne viendra pas ». Marie répond qu'elle a vu Paul hier.")
    assert glosses(r) == [["PAUL", "DIRE"], ["IL", "VENIR", "PAS"], ["MARIE", "RÉPONDRE"], ["HIER", "ELLE", "PAUL", "VOIR"]]
    quote, reported = r["clauses"][1], r["clauses"][3]
    assert quote["role_shift"]["referent"] == "paul" and quote["role_shift"]["kind"] == "quote"
    assert quote["markers"] == ["negation"]
    assert index_of(quote, "il") is None                 # inside Paul's quote, "il" is never Paul himself
    assert reported["role_shift"]["referent"] == "marie" and reported["role_shift"]["kind"] == "reported"
    assert index_of(reported, "elle") == "marie"
    assert r["clauses"][0]["role_shift"] is None and r["clauses"][2]["role_shift"] is None
    # two people -> two different loci, and each role shift points at its speaker's locus
    loci = {x["id"]: x["locus"] for x in r["referents"]}
    assert {loci["paul"], loci["marie"]} == {"L", "R"}
    assert quote["role_shift"]["locus"] == loci["paul"] and reported["role_shift"]["locus"] == loci["marie"]


def test_speech_incise_is_moved_before_the_quote():
    r = O.order("« Tu viens ? » demande Paul.")
    assert glosses(r) == [["PAUL", "DEMANDER"], ["TOI", "VENIR"]]
    assert r["clauses"][1]["markers"] == ["question_yn"] and r["clauses"][1]["role_shift"]["referent"] == "paul"
    r = O.order("« Je suis fatigué », dit-il.")
    assert glosses(r) == [["IL", "DIRE"], ["MOI", "FATIGUÉ"]]


def test_reported_speech_variants():
    r = O.order("Le ministre a annoncé que les écoles seront fermées demain.")
    assert r["clauses"][1]["role_shift"]["referent"] == "ministre"
    assert glosses(r)[1][0] == "DEMAIN"
    # the signer's own voice ("je pense que") is no role shift
    assert all(c["role_shift"] is None for c in O.order("Je pense que tu as raison.")["clauses"])


def test_addressee_of_a_quote_is_tu():
    r = O.order("Le directeur est entré. Il a dit à Marie : « Tu es en retard ! »")
    frame, quote = r["clauses"][1], r["clauses"][2]
    assert "À" not in frame["glosses"] and index_of(frame, "il") == "directeur"
    assert quote["role_shift"]["referent"] == "directeur" and index_of(quote, "tu") == "marie"


def test_scene_and_speaker_carry_over_between_subtitles():
    first = O.order("Le directeur a ouvert la porte.")
    second = O.order("Il est parti.", scene=first["referents"])
    assert index_of(second["clauses"][0], "il") == "directeur"
    assert {x["id"]: x["locus"] for x in second["referents"]}["directeur"] == \
        {x["id"]: x["locus"] for x in first["referents"]}["directeur"]          # same referent, same locus
    mine = O.order("Je pense qu'il a raison.", speaker={"label": "Marie", "locus": "R"})
    assert index_of(mine["clauses"][0], "je") == "marie" and mine["clauses"][0]["voice"] == "marie"
    marie = next(x for x in mine["referents"] if x["id"] == "marie")
    assert marie["locus"] == "R" and marie["speaker"] and marie["gender"] == "f"


def test_deterministic_and_gloss_indices_point_at_the_glosses():
    text = "Paul dit : « Il ne viendra pas ». Marie répond qu'elle a vu Paul hier."
    assert O.order(text) == O.order(text)
    for c in O.order(text)["clauses"]:
        for e in c["events"]:
            if e["type"] in ("index", "introduce", "mention") and e["gloss_index"] is not None:
                assert c["glosses"][e["gloss_index"]].startswith(e["token"].split()[-1].upper()[:3])


def test_role_shift_spans_follow_the_clause_split():
    r = O.order("Paul dit : « Il ne viendra pas ».")
    spans = O.role_shift_spans(r, None, 101)                       # C0: 4 s clip, proportional split
    assert len(spans) == 1 and spans[0]["referent"] == "paul" and spans[0]["kind"] == "quote"
    assert 300 < spans[0]["start_ms"] < spans[0]["end_ms"] <= 3700
    signs = [{"start_frame": 2, "end_frame": 10, "clause": 0}, {"start_frame": 12, "end_frame": 30, "clause": 1}]
    assert O.role_shift_spans(r, signs, 40) == [{"start_ms": 480, "end_ms": 1200, "referent": "paul", "label": "PAUL",
                                                 "locus": spans[0]["locus"], "kind": "quote"}]


def test_order_endpoint_accepts_speaker_and_scene():
    r = client.post("/lsf/order", json={"text": "Je viens.", "speaker": {"label": "Paul", "locus": "L"}})
    assert r.status_code == 200
    body = r.json()
    assert body["clauses"][0]["voice"] == "paul" and body["referents"][0]["locus"] == "L"
    r = client.post("/lsf/order", json={"text": "Il est parti.", "referents": body["referents"]})
    assert index_of(r.json()["clauses"][0], "il") == "paul"


def test_generate_returns_role_shift_spans(monkeypatch):
    monkeypatch.setattr(main.lsf_engines, "generate", lambda text, engine, seed=0, groups=None: (fixture(T=50), {"engine": engine}))
    m = client.post("/lsf/generate", json={"text": "Marie dit qu'elle viendra demain.", "engine": "c0"}).json()
    assert [s["referent"] for s in m["role_shift"]] == ["marie"]
    assert m["generation"]["gloss_analysis"]["referents"][0]["id"] == "marie"
    assert "role_shift" not in client.post("/lsf/generate", json={"text": "Je viens demain.", "engine": "c0"}).json()
