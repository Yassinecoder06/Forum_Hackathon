# SignaCulture AI

Studio local de recherche pour explorer du texte français, des mouvements LSF capturés et un avatar 3D. Le lecteur fonctionne avec des vidéos réelles ; la traduction générale français → LSF n’est pas résolue.

## Essayer

```bash
make setup
python3 scripts/setup-avatar.py   # personnage local pour revue, voir docs/AVATAR_PROVENANCE.md
bash scripts/setup-models.sh all   # capture MediaPipe et emotion2vec local, téléchargements explicites
make dev
```

Ouvrir **http://127.0.0.1:5173**. Dans Studio, essayer « Bonjour ». Bibliothèque permet de sélectionner un mouvement, de comparer sa vidéo source et de télécharger un aperçu WebM. Capturer accepte une vidéo autorisée, un JSON de mouvements ou un export de landmarks Animics. Recherche affiche les expériences terminées et permet d’interroger les checkpoints locaux.

La session actuelle contient **440 mouvements réels** : Bonjour, fourni avec attribution, et 439 captures importées. Les vidéos et dérivés du corpus restent locaux dans `.runtime/`. Sur un nouveau checkout, seul Bonjour est livré ; pour reconstruire la bibliothèque :

```bash
.venv/bin/python scripts/ingest_commons_signs.py --all --author-mirror --workers 2
```

L’inventaire contient 447 concepts France-LSF avec licences par fichier, crédits et sommes de contrôle. Toutes les vidéos ont été récupérées dans cette session. Sept captures échouent au seuil technique de suivi des mains et restent hors bibliothèque. Le [rapport d’ingestion](data/source_inventory/ingestion_report.json) précise les critères. Une capture admise par ce seuil n’est pas une validation linguistique.

## Ce qui fonctionne

- Indexation de concepts avec lemmatisation française spaCy lorsqu’installée, expressions les plus longues et inconnus explicites. Ce lookup ne traduit pas la grammaire LSF.
- Extraction MediaPipe du corps, des doigts et de quelques mesures faciales. Les occultations restent visibles ; profondeur et retargeting sont approximatifs.
- Avatar articulé Three.js : IK des bras, orientation des paumes, os des doigts, longueurs préservées. Liaisons quintiques de 333 ms entre clips, timeline, comparaison avec la source et vrai export WebM avec filigrane et attribution. Le modèle local de revue et ses limites sont décrits dans [la provenance](docs/AVATAR_PROVENANCE.md).
- Décodage audio/vidéo FFmpeg, estimation F0, RMS et pauses. Synchronisation approximative aux portions voisées, sans alignement mot à mot.
- Inférence locale emotion2vec, affichée comme hypothèse. Les signaux faciaux capturés priment sur la modulation expressive expérimentale.
- Catalogue SQLite séparant vidéos lexicales, annotations et paires de recherche ; provenance, licences, splits et évaluations conservés.

## Composition assistée et revue

Dans Studio, ouvrir « Préparer et corriger les concepts ». On peut sélectionner les signes réels, les réordonner, prévisualiser la sélection et télécharger une fiche JSON pour une revue linguistique. Ces fiches restent des brouillons, jamais des données expertes automatiquement approuvées. Une proposition sémantique locale Qwen3.5-9B est disponible en option : démarrer `python3 scripts/start-concept-server.py`, puis cliquer « Proposer des concepts ». Le modèle utilise environ 5,2 Go de VRAM avec offload partiel. Le premier essai logiciel donne 13/24 séquences exactes contre 12/24 pour le lookup ; le runtime corrigé ensuite reste non scoré. Voir [la solution](docs/SEMANTIC_SOLUTION.md) et [le rapport mesuré](research/CONCEPT_PLANNER_REPORT.md).

## Résultats mesurés

128 paires de recherche issues d’annotations publiées UD/SUD STKAutogramm : **61 entraînement / 28 validation / 39 test**. Les traductions françaises libres ne forment pas un corpus parallèle validé. Les vidéos lexicales ne deviennent jamais des paires de phrases inventées. Zéro approbation linguistique locale.

| Expérience | Exact match test | WER test | BLEU |
|---|---:|---:|---:|
| T5-small LoRA CPU, smoke | 0 % | 207,28 % | 0 |
| mT5-small LoRA RTX 4060, 8 époques | 0 % | 100 % | 0 |
| mT5-small, sélection par validation, arrêt à 21 époques | 0 % | 517,96 % | 0,042 |
| Retrieval sur le corpus élargi | 0 % | 120,87 % | 0,301 |

Ces modèles **ne sont pas des traducteurs utilisables**. Le troisième essai réduit la perte de validation mais produit des répétitions ; son test est une réutilisation d’un test d’ingénierie, pas une confirmation indépendante. Aucun résultat état de l’art n’est revendiqué. Voir [l’investigation](research/ADAPTATION_INVESTIGATION.md), [la carte des données](research/DATASET_CARD.md), [les résultats](research/IMPLEMENTATION_STATUS.md) et [les commandes de reproduction](training/README.md).

Le PDF fourni `research_paper.pdf` a maintenant été lu intégralement. La [comparaison vérifiée](research/PAPER_B_VERIFIED_REVIEW.md) distingue les composants du papier de notre implémentation ; un CGAN à cinq points est maintenant entraîné pour une expérience intra-signe, sans activation comme modèle inter-signes. Les anciennes mentions d’un PDF absent décrivent un état antérieur et sont corrigées par cette revue. Dicta-Sign, Rosetta et MEDIAPI sont catalogués avec leurs limites d’accès ; aucun accès institutionnel n’a été simulé.

## Vérification

```bash
make test
CHROME_PATH=/chemin/vers/chrome npm --prefix e2e test
# Environnement ML séparé, avec dépendances training installées :
.worktrees/wt-codex2/training/.venv-train/bin/python -m pytest training/test_catalog.py training/test_adaptation.py
```

67 tests API/schémas/mouvement et 24 tests de recherche passent, avec des vérifications frontend des transitions et du rig. Un test strictement marqué xfail conserve le défaut connu de détection du prénom Alice. Le contrôle navigateur couvre composition, recherche temporelle, téléchargement réel, audio, import, persistance, mobile et erreurs réseau. Les [captures et aperçu vidéo](artifacts/README.md) documentent le logiciel exécuté.

Les modèles `.models/`, environnements, checkpoints et captures `.runtime/` sont ignorés par Git. Aucun téléchargement automatique n’a lieu lors d’une requête API. Le modèle français optionnel se configure dans [backend/README.md](backend/README.md).

Développement : exactement deux worktrees, racine `main` pour le lead et `.worktrees/wt-codex2` pour `codex2`, GPT-6.1 Sol medium. Voir [l’organisation](docs/team_worktree_organization.md). L’API reste locale : le système de revue n’est pas un service d’authentification. La publication d’une traduction reste bloquée sans validation réelle.

## Réparation après revue du papier

Le [diagnostic](research/FAILURE_ANALYSIS_AND_REPAIR.md) distingue les erreurs du prototype des composants manquants. Les fixtures synthétiques sont exclues du lookup normal ; les clips réels conservent leurs deux extrémités et disposent d’une liaison de 333 ms ; le personnage est articulé avec IK et doigts. Le [CGAN entraîné](research/TRANSITION_REPORT.md) améliore légèrement la reconstruction intra-signe, sans démontrer la coarticulation inter-signes. Les données originales du papier et la validation linguistique restent manquantes.
