# SignaCulture AI — Plan complet de développement et d'exécution

> **Projet :** SignaCulture AI — génération d'animations 3D en langue des signes française (LSF), enrichies par l'analyse de la prosodie et de l'expressivité vocale.  
> **Cadre :** Hackathon IA Éthique 360 (13–25 novembre 2026)  
> **Version :** Plan opérationnel v1.0 — 15 novembre 2026  
> **Matériel cible :** NVIDIA RTX 4060 (8 Go de VRAM supposés)  
> **Statut :** proposition de développement. Les fonctionnalités, métriques et validations décrites comme objectifs ne sont pas des résultats déjà obtenus.

---

## 0. Synthèse du projet

**Objectif :** construire une plateforme qui reçoit une phrase française (et éventuellement son audio), prépare une représentation *gloss* pour un vocabulaire LSF contrôlé, retrouve des mouvements signés dans une bibliothèque documentée, les anime sur un avatar 3D, puis expérimente l'ajout de paramètres expressifs issus de l'audio (intonation, intensité, pauses, scores d'émotion).

**Pipeline initial :**

```text
Texte français ──> spaCy / appariement d'expressions ──> gloss candidats
                                                    │
Bibliothèque LSF : gloss ↔ assets de mouvements ────┤
                                                    ▼
                                 Plan d'animation vérifiable
                                                    │
Audio (optionnel) ──> emotion2vec+ + prosodie ──> paramètres expressifs proposés
                                                    │
                                                    ▼
                                Contraintes linguistiques / validation
                                                    │
                                                    ▼
                         Avatar 3D → aperçu → rendu vidéo
```

**Contraintes essentielles :**

- La **lemmatisation n'est pas une traduction linguistiquement correcte en LSF**. Elle constitue une baseline limitée pour du vocabulaire et des expressions contrôlés.
- Les **glosses ne décrivent pas intégralement** la grammaire spatiale et non manuelle de la LSF.
- L'extraction de points-clés 2D/3D ne suffit pas à garantir des gestes intelligibles ou une animation exacte.
- L'émotion prédite par la voix est **une hypothèse d'expressivité**, pas une lecture fiable de l'état intérieur du locuteur.
- Les marqueurs grammaticaux de la LSF (visage, regard, tête, rythme) ont **priorité** sur les effets émotionnels ajoutés.
- L'utilisation de corpus accessibles publiquement n'autorise pas nécessairement le téléchargement massif, le réentraînement, la diffusion des dérivés ou la commercialisation.
- Pour tout contenu réellement publié comme accessible en LSF : contrôle humain compétent, droits validés et statut d'approbation explicite.

**MVP première soumission :** démontrer un flux réel *texte → entrée gloss → animation 3D* sur quelques signes autorisés, compléter l'analyse audio comme expérience indépendante si possible, et remettre les livrables imposés.

---

## 1. Organisation — quatre équipes parallèles

Une personne assure le rôle de **lead technique / intégration** et coordonne les interfaces, les tests end-to-end et la présentation. Dans une équipe de quatre, chaque membre prend une filière ; dans une équipe de cinq, le cinquième peut assurer l'intégration, la validation terrain et le pitch.

| Équipe | Mission | Livrable clé | Dépendances |
|---|---|---|---|
| **A — LSF Data Engineering** | Construire le registre gloss ↔ signification ↔ motion, documenter sources et droits | `glosses.json`, `motions_manifest.json`, `sources.csv` | Accord format avec B, disponibilité de ressources autorisées |
| **B — Computer Vision & Avatar 3D** | Extraire, normaliser, retargeter et jouer les mouvements | Animation 3D testée sur rig commun | Assets de A / vidéos autorisées |
| **C — NLP Text-to-Gloss** | Lemmatisation, correspondance d'expressions, gloss lookup, gestion des inconnus | Service de lookup et `AnimationPlan` | Schéma et gloss IDs de A |
| **D — Audio & Expressiveness** | Estimer prosodie et expressions vocales, tester modulation contrainte | Features audio timestampées et expérience A/B | Contrat temporel avec B |
| **Lead / produit / validation** | Architecture commune, API, Git, intégration, témoignages, éthique, présentation | Démo end-to-end, README, slides, vidéo | Toutes les équipes |

### 1.1 Équipe A — LSF Data Engineering

**Responsabilité :** bâtir une base utilisable, traçable et juridiquement admissible, avec des identifiants stables de gloss et de mouvements.

| Ticket | Actions concrètes | Critère d'acceptation |
|---|---|---|
| `DATA-01` Explorer les ressources | Examiner Dicta-Sign Concepts FR, Dicta-Sign-LSF-v2 ; relever attribution, licence, accès et contraintes | Registre `sources.csv` renseigné, usages autorisés explicités |
| `DATA-02` Choisir un domaine restreint | Sélectionner 5 à 10 signes ou expressions adaptés à une courte vidéo culturelle | Vocabulaire limité défini, sens et variante linguistique indiqués |
| `DATA-03` Créer le schéma | `gloss_id`, `concept_fr`, `variant`, `source_id`, `motion_id`, statut de validation | JSON validé par schéma et relu par B/C |
| `DATA-04` Associer les fichiers | Indexer uniquement les assets qu'il est permis de traiter | Chaque ID résout vers un fichier existant ou état indisponible |
| `DATA-05` Contrôler la qualité | Signaler les mouvements incomplets, doublons, erreurs de suivi et ambiguïtés linguistiques | Rapport qualité + flags par asset |
| `DATA-06` Documenter | Provenance, limitations, restrictions de redistribution, éventuels consentements | `DATASET_CARD.md`, registre des licences |

**Sources initiales :**

- Dicta-Sign French Concepts : https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_fre.html
- Dicta-Sign-LSF-v2 : https://hdl.handle.net/11403/dicta-sign-lsf-v2/v1
- Échantillons originaux obtenus avec accord éclairé de signants compétents en LSF.

**Attention :** ne pas scraper ni redistribuer automatiquement les vidéos du portail Dicta-Sign. Vérifier et obtenir les permissions pertinentes avant toute extraction et transformation de corpus en bibliothèque de mouvements. Les licences logicielles d'Animics ne transfèrent pas les droits sur des performances filmées.

**Definition of done :** Team C peut retrouver un `gloss_id` et Team B peut retrouver l'asset associé via un identifiant partagé ; aucun contenu non autorisé dans le dépôt public.

### 1.2 Équipe B — Computer Vision, Motion & Avatar 3D

**Responsabilité :** transformer des enregistrements autorisés en clips réutilisables sur un avatar 3D.

| Ticket | Actions concrètes | Critère d'acceptation |
|---|---|---|
| `MOTION-01` Installer Animics | Cloner, lancer, explorer Keyframe Mode et Script Mode | Application fonctionnelle, démo d'animation basique |
| `MOTION-02` Tester l'extraction | Prendre une vidéo autorisée, vérifier mains, torse, tête et visage | Landmarks et taux d'échec documentés |
| `MOTION-03` Normalisation | Uniformiser coordonnées, échelles, repères et temps ; détecter inversions de mains | Séquence lisible sur rig cible |
| `MOTION-04` Retargeting | Mapper les mouvements à un squelette commun | Animation de référence correctement jouée |
| `MOTION-05` Export | Export GLB/BVH compatible avec le lecteur choisi | `motion_id` → asset jouable dans l'app |
| `MOTION-06` Composition | Tester concaténation et transitions sur séquences approuvées | Pas de sauts majeurs ni de déformation critique |
| `MOTION-07` Vérification | Contrôler formes de mains, occultations, sourcils, regards et vitesse | Rapport d'anomalies + statut de validation |

**Outils :**

- Animics : https://github.com/upf-gti/animics
- Performs : https://github.com/upf-gti/performs
- MediaPipe Holistic : https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker
- Blender : https://www.blender.org/

**Points techniques à documenter :** FPS, rig commun, système de coordonnées, représentation des rotations, précision des doigts, interpolation, occlusions et qualité de retargeting.

**Important :** les modes Script et Keyframe d'Animics sont des flux distincts. Le fait d'exporter tous les clips vers un format commun ne garantit pas automatiquement leur compatibilité linguistique ou géométrique.

**Definition of done :** un clip signé autorisé est accessible depuis un `motion_id` et s'anime de façon reproductible sur l'avatar.

### 1.3 Équipe C — NLP, Text-to-Gloss & Lookup

**Responsabilité :** transformer les entrées françaises du périmètre contrôlé en références vérifiées, sans inventer des traductions.

| Ticket | Actions concrètes | Critère d'acceptation |
|---|---|---|
| `NLP-01` Installer spaCy FR | Tokenisation et lemmatisation | Pipeline de base déterministe |
| `NLP-02` Matcher des expressions | Priorité aux expressions multi-mots validées avant tokens isolés | Expressions couvertes retrouvées sans fausses substitutions |
| `NLP-03` Canonicaliser les glosses | Harmoniser casse, variantes, formes canoniques | ID de gloss stable pour la bibliothèque |
| `NLP-04` Résoudre gloss → motion | Interroger le registre A | Liste d'assets existants, pas d'ID fictif |
| `NLP-05` Gérer les inconnus | Détecter couverture partielle, ambiguïtés et données absentes | Réponse `unmatched` explicite |
| `NLP-06` Produire un plan | Schéma standard `TranslationDraft` / `AnimationPlan` | Sortie testée par B |
| `NLP-07` Prévoir revue humaine | Gloss candidats et statut non approuvé par défaut | Aucune publication automatisée non validée |

**Outils :**

- spaCy français : https://spacy.io/models/fr
- Baseline de comparaison : https://github.com/sign-language-processing/spoken-to-signed-translation

**Contrôle linguistique :** la sortie du lemmatiseur est une **approximation de lookup**, non une traduction LSF. Un gloss ne suffit pas à spécifier l'espace, les marqueurs du visage et la grammaire complète. La prise en charge de phrases non validées doit provoquer un échec explicite ou une demande de revue.

**Definition of done :** un exemple validé de la liste blanche renvoie des IDs d'animation corrects ; une phrase non couverte renvoie `unmatched` et `translation_validated: false`.

### 1.4 Équipe D — Audio, Emotion2vec & Prosody

**Responsabilité :** analyser la voix, produire un signal temporel d'expressivité, puis tester son utilité sans dégrader le sens signé.

| Ticket | Actions concrètes | Critère d'acceptation |
|---|---|---|
| `AUDIO-01` Prétraitement | Charger, normaliser, gérer mono/stéréo et fréquences d'échantillonnage | Audio reproductible, timestamps conservés |
| `AUDIO-02` Tester emotion2vec+ | Exécuter un checkpoint et obtenir catégories / scores | JSON de prédictions datées, version du modèle stockée |
| `AUDIO-03` Analyser la prosodie | Extraire F0, RMS, silences, rythme, durée des segments | Courbes temporelles exploitables |
| `AUDIO-04` Aligner les sorties | Transformer en ms, associer segments audio aux unités linguistiques | `AudioFeatures` compatible avec le composeur |
| `AUDIO-05` Proposer contrôles | Intensité, pauses, emphase, vitesse dans limites autorisées | Contrôleur expérimental sans altération grammaticale imposée |
| `AUDIO-06` Comparer | Animation neutre, prosodie seule, puis prosodie+émotion | Tableau d'ablation et retours linguistiques |

**Outils :**

- emotion2vec / emotion2vec+ : https://github.com/ddlBoJack/emotion2vec
- librosa : https://librosa.org/
- Parselmouth : https://parselmouth.readthedocs.io/
- VAD ou timestamps audio si nécessaire.

**Règle impérative :** les marqueurs grammaticaux de la LSF priment sur toute animation expressive automatique. Le contrôleur audio propose une modulation contrainte, et le réviseur compétent conserve la décision finale. Les prédictions émotionnelles peuvent être inexactes et sensibles au contexte / à l'intonation jouée.

**Definition of done :** fichier de caractéristiques audio versionné, aligné en millisecondes, comparatif de plusieurs enregistrements. Le contrôle de l'avatar constitue une étape additionnelle.

### 1.5 Lead technique et validation produit

- Définir l'API et les contrats JSON ; valider les IDs, FPS, timestamps et version de rig.
- Mettre en place les branches Git, revue de code et pipeline de tests.
- Organiser au moins une vérification du besoin par un professionnel audiovisuel et un retour d'une personne compétente en LSF, si accessible.
- Documenter droits/consentement, sécurité, transparence, limitations et six piliers éthiques.
- Garantir les livrables du concours : slides, vidéo, dépôt GitHub.
- Vérifier que toute démo distingue clairement fonctionnalités actuelles et développements futurs.

---

## 2. Contrats d'intégration et architecture

### 2.1 Contrats communs

| Objet | Produit par | Utilisé par | Champs obligatoires |
|---|---|---|---|
| `GlossEntry` | A | C / B | `gloss_id`, `concept_fr`, `language`, `variant`, `source_id`, `review_status` |
| `MotionAsset` | B / A | B / UI | `motion_id`, `gloss_id`, `path`, `rig_version`, `fps`, `duration_ms`, `license`, `quality_status` |
| `TranslationDraft` | C | UI / validateur / B | `source_text`, `gloss_sequence`, `unknowns`, `linguistic_review` |
| `AudioFeatures` | D | B / UI | `start_ms`, `end_ms`, `pitch`, `energy`, `emotion_hypothesis`, `model_version` |
| `AnimationPlan` | Intégration | B / rendu | `motion_id`, `start_ms`, `duration_ms`, `linguistic_review`, `rights_status`, `render_allowed_for_publication` |
| `ReviewRecord` | Réviseur | Backend export | `reviewer`, `status`, `notes`, `version`, `date` |

**Règles :**

1. Références de gloss et de mouvement stables et globalement uniques dans le projet.
2. Horodatage en **millisecondes** partout ; FPS stocké explicitement par asset.
3. Rig 3D versionné et partagé par B et le frontend.
4. Les champs `confidence`, `emotion`, `translation` sont des estimations tant qu'un examen humain n'est pas fait.
5. La licence et le statut d'approbation sont attachés aux données et aux dérivés.
6. Aucun export de contenu présenté comme une traduction approuvée sans validation linguistique, validation animation et droits de publication.

### 2.2 Exemple d'entrée gloss

```json
{
  "gloss_id": "LSF_BONJOUR_001",
  "language": "LSF",
  "concept_fr": "bonjour",
  "gloss": "BONJOUR",
  "variant_id": "V001",
  "source_id": "SRC001",
  "motion_id": "MOTION_BONJOUR_001",
  "license": "PERMISSION_REQUIRED",
  "linguistic_review": "pending",
  "motion_review": "pending"
}
```

### 2.3 Exemple de plan d'animation

```json
{
  "project_id": "demo_001",
  "source_text": "Bonjour",
  "target_language": "LSF",
  "segments": [
    {
      "gloss_id": "LSF_BONJOUR_001",
      "motion_id": "MOTION_BONJOUR_001",
      "start_ms": 0,
      "duration_ms": 900
    }
  ],
  "audio_features_id": "AUDIO_001",
  "linguistic_review": "pending",
  "animation_review": "pending",
  "rights_status": "pending",
  "render_allowed_for_publication": false
}
```

Les valeurs et exemples ci-dessus illustrent le **contrat logiciel**, pas une traduction LSF vérifiée.

### 2.4 Stack et composants

| Couche | Choix MVP | Notes |
|---|---|---|
| Frontend | React + TypeScript | Formulaire de texte, visualisation gloss, audio, avatar, statuts |
| API | FastAPI (Python) | Service unique pour MVP, séparation modulaire des fonctions |
| Base de données | JSON et SQLite | Migration éventuelle vers PostgreSQL |
| NLP | spaCy `fr_core_news_sm` | Lemmatisation / expressions sous contrôle |
| Reconnaissance audio optionnelle | Faster-Whisper | Nécessaire si input uniquement voix, pas pour première démo texte |
| Emotion/prosodie | emotion2vec+, librosa, Parselmouth | Exécution locale, analyse différée |
| Extraction de points-clés | MediaPipe / Animics | Vérification de la qualité et des mains obligatoire |
| Avatar et lecture | Animics + Performs ou moteur Three.js compatible | Tester les exports réellement supportés |
| Montage vidéo | FFmpeg | Étape après le rendu de base |
| Matériel | RTX 4060 | Ne pas charger simultanément des modèles lourds ; profiler VRAM |

### 2.5 API minimum

| Méthode / endpoint | Résultat |
|---|---|
| `POST /analyze-text` | Tokens, lemmas, expression matches |
| `POST /gloss-lookup` | Gloss IDs, motion IDs, inconnus |
| `GET /motions/{motion_id}` | Manifest et chemin d'asset autorisé |
| `POST /analyze-audio` | Timeline prosodie + scores d'expression |
| `POST /compose-animation` | Aperçu d'un plan d'animation |
| `POST /review` | Approbation, rejet ou commentaire |
| `POST /render` | Aperçu expérimental ou export avec contrôles d'autorisation |

### 2.6 Structure du dépôt

```text
signaculture-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── schemas/
│   │   └── services/
│   │       ├── text_to_gloss.py
│   │       ├── gloss_lookup.py
│   │       ├── motion_registry.py
│   │       ├── audio_emotion.py
│   │       ├── audio_prosody.py
│   │       └── animation_composer.py
│   └── tests/
├── frontend/
│   └── src/
├── data/
│   ├── glossary.json
│   ├── motions_manifest.json
│   ├── sources.csv
│   └── approved_motion_assets/
├── experiments/
│   ├── motion_extraction/
│   └── emotion2vec/
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── data_governance.md
│   ├── ethics.md
│   └── validation.md
└── README.md
```

Conserver les enregistrements identifiables et les consentements **hors** dépôt public, dans un stockage à accès limité.

---

## 3. Définition exacte du MVP

### 3.1 Trois démonstrations

**Démo 1 — Texte vers avatar (P0)**

1. L'utilisateur saisit une phrase française appartenant à une liste autorisée.
2. spaCy fournit des informations linguistiques basiques.
3. Un matcher trouve l'expression/gloss connu.
4. Le registre renvoie un asset signé existant.
5. Le player 3D joue cet asset.
6. La UI affiche « expérimental / en attente de validation » si nécessaire.

**Démo 2 — Analyse audio (P1)**

1. L'utilisateur charge un court enregistrement vocal.
2. Le programme retourne une timeline (F0, RMS, pauses, scores de l'émotion classifiée).
3. La UI affiche ces mesures et leurs limites.

**Démo 3 — Avatar expressif (P2, stretch)**

1. Charger le même clip signé dans deux modes : neutre et expressif.
2. Appliquer une modulation limitée compatible avec le contrat d'animation.
3. Présenter la comparaison, sans prétendre à une correction linguistique non validée.

### 3.2 Priorités

| Niveau | Fonctionnalité | Importance |
|---|---|---|
| **P0** | Démarrage avatar + lecture d'un clip réel | Critique |
| **P0** | Registre gloss → motion avec sources et droits | Critique |
| **P0** | Endpoint de lookup simple, gestion des inconnus | Critique |
| **P0** | README français, slides, vidéo 1–2 min et dépôt GitHub | Obligatoire concours |
| **P1** | Courbes audio, prédictions emotion2vec+ | Forte valeur démonstrative |
| **P1** | Plusieurs clips enchaînés et revue linguistique | Important |
| **P2** | Paramétrage audio des expressions/rythmes de l'avatar | Expérimental |
| **P2** | GAN / transitions apprises / text-to-gloss neuronal | Au-delà du MVP |

**Règle de réduction de périmètre :** si Team B ne réussit pas le premier rendu animé, ne pas bloquer l'équipe sur une génération complexe : tester une animation autorisée manuelle et montrer honnêtement l'état de l'extraction automatisée.

---

## 4. Planning du concours et coordination

> Dates issues du cahier des charges IA Éthique 360 : première remise le **15 novembre 2026 à 23:59 GMT+1**, sélection des dix équipes le 17, workshop le 18, demi-finale le 21, soumission finale le 24 à 23:59, finale le 25.

### 4.1 15 novembre — Première soumission

| Bloc | Team A | Team B | Team C | Team D | Lead |
|---|---|---|---|---|---|
| Bloc 1 | Sources / droits | Lancer Animics | spaCy / baseline | Installer emotion2vec | Créer repo, contrats |
| Bloc 2 | Gloss IDs, manifest | Exporter un clip | API lookup | Tester 1 audio | Tester contrats |
| Bloc 3 | Corriger métadonnées | Lecteur avatar | Connecter motion IDs | Pitch / energy | Test vertical slice |
| Bloc 4 | Documenter jeu de données | Filmer démo | Tests + README | Visualiser courbes | Slides, vidéo française |
| Bloc final | Vérifier licences | Démo reproductible | Finaliser API | Résultats honnêtes | Dépôt complet avant 23:59 |

**Livrables officiels :**

- Slides en **français** sur problème réel, justification, solution IA, six piliers éthiques, faisabilité et perspective économique.
- Vidéo en **français de 1–2 minutes** présentant l'idée et l'IA (une démo complète n'est pas requise à cette première étape).
- Dépôt **GitHub**, code éventuellement incomplet, README détaillé en français avec démarche éthique.

### 4.2 16–17 novembre — Consolider

- Intégrer l'interface et les endpoints.
- Étendre la bibliothèque de quelques glosses avec permissions vérifiées.
- Tester les chemins d'échec et les motion IDs.
- Finaliser le premier pipeline audio de Team D.

### 4.3 18–21 novembre — Préparer la demi-finale

- Ajouter composition multi-clips, gestion de transitions et détection des anomalies.
- Mettre en place interface de validation d'expert.
- Obtenir un premier retour sur la compréhension LSF, avec limites méthodologiques reconnues.
- Préparer **pitch 7 minutes + questions 5 minutes**.

### 4.4 22–24 novembre — Finaliser

- Stabiliser avatar, UI et démo reproductible.
- Ajouter export vidéo si l'animation est fiable.
- Documenter tests utilisateurs, éthique, sécurité, coûts et business model.
- Préparer et remettre : **vidéo de démonstration 1–2 min, pitch deck français, GitHub finalisé**, avant 23:59 le 24 novembre.

### 4.5 25 novembre — Finale

- Présenter **6 minutes de pitch**, **1 minute de démonstration**, **4 minutes de questions**, puis **1 minute de vote du public**.
- Utiliser une copie de secours (vidéo locale, slides, éventuels assets hors-ligne).

---

## 5. Évaluation scientifique et technique

### 5.1 Hypothèse principale

> *L'intégration de caractéristiques acoustiques de prosodie et d'indices d'expressivité vocale dans une animation LSF fondée sur des mouvements réels permet-elle d'améliorer la naturalité perçue sans diminuer l'intelligibilité ni altérer les marqueurs grammaticaux de la LSF ?*

**La recherche existante ne permet pas d'affirmer à l'avance que oui.** L'étude doit comparer plusieurs variantes et recueillir une validation de signants compétents en LSF.

### 5.2 Expériences

| Expérience | Baseline | Variante | Mesure |
|---|---|---|---|
| E1 — Gloss lookup | Lemmatisation simple | Matching expressions + canonicalisation | Couverture et faux appariements |
| E2 — Motion extraction | Landmarks bruts | Normalisation + retargeting | Taux de tracking réussi, erreur de reconstruction |
| E3 — Coarticulation | Concaténation rigide | Interpolation / transitions | Discontinuité, précision des mains, retour humain |
| E4 — Prosodie | Avatar neutre | Prosodie contrôlée | Naturalité et compréhension |
| E5 — Affect | Prosodie seule | Prosodie + emotion2vec+ | Gain perceptuel, erreurs grammaticales |

### 5.3 Critères de validation

- **Technique :** animation jouable, formats compatibles, absence d'erreurs critiques, temps de chargement, VRAM maximale, FPS/latence.
- **Linguistique :** compréhension sans sous-titres français, précision du sens, structure spatiale, configuration manuelle, marqueurs non manuels, expressions culturelles.
- **Produit :** temps de production, modifications demandées, facilité d'export, besoins réels de producteurs.
- **Éthique :** licences, consentement, confidentialité, journal de validation et transparence.

Prévoir un test avec au moins une personne compétente en LSF ; solliciter des utilisateurs sourds signants lorsque possible. Un test sur un nombre très réduit de participants demeure exploratoire, non représentatif.

### 5.4 Référence scientifique : T5-GAN 2026

Le papier fourni par l'équipe, *Signing Avatar Synthesis via a Hybrid T5-GAN Approach: Automatic Speech-to-French Sign Language Translation*, décrit déjà une chaîne de traduction française vers gloss, récupération de landmarks, transitions GAN et rendu. Il rapporte notamment 319 signes isolés, 2 500 paires phrase–gloss et 379 séquences. Il reconnaît comme limites l'absence de marqueurs non manuels, une représentation de main simplifiée, un avatar 2D stylisé et des problèmes d'occlusion.

**Positionnement différenciant :** construire un système 3D traçable, avec une couche expérimentale prosodie/émotion **soumise à des contraintes linguistiques et à des tests d'intelligibilité** ; ne pas revendiquer l'invention de text-to-gloss ou gloss-to-pose.

---

## 6. Risques, mesures d'atténuation et décision de repli

| Risque | Priorité | Action / repli |
|---|---|---|
| Licence Dicta-Sign incompatible | Critique | N'utiliser que les éléments autorisés ; enregistrer des exemples consentis ; ne pas scraper |
| Réutilisation de données personnelles | Critique | Minimer la collecte, contrôle d'accès, consentement, suppression selon conditions |
| Gloss incorrect grammaticalement | Critique | Liste blanche approuvée ; erreur explicite sinon ; revue par spécialiste |
| 3D mal retargetée | Haute | Corriger manuellement ou choisir asset de démo autorisé |
| Doigts / visage mal reconstruits | Haute | Contrôler image à image, ne pas publier comme LSF correct |
| Emotion2vec instable ou trompeur | Moyenne | Garder un mode prosodie seule ; afficher hypothèses et incertitude |
| Expression qui modifie la grammaire | Critique | Priorité aux contraintes LSF, désactiver modulation, validation humaine |
| Intégration en retard | Haute | Une seule animation verticale fonctionnelle, autres équipes présentent expériences distinctes |
| VRAM insuffisante | Moyenne | Exécution séquentielle, CPU pour audio/pose si nécessaire, quantification pertinente |
| Pas de preuve du besoin réel | Haute | Entretien rapide avec producteur + revue d'utilisateur cible, témoigner avec rigueur |

---

## 7. Éthique à 360° — responsabilités techniques

| Pilier officiel | Équipes | Preuve technique/organisationnelle |
|---|---|---|
| **Créateurs** | B, C, lead | L'utilisateur contrôle, refuse ou édite l'animation et ses variantes |
| **Droits** | A, lead | Registre de licences, consentements, attribution et conditions d'usage |
| **Données** | A, D | Origine documentée, accès limité, stockage séparé des informations personnelles |
| **Cultures et langues** | A, B, C | Définition précise de LSF/variantes, avis d'experts signants, limites reconnues |
| **Transparence** | Tous | Mention IA, source et statut de validation, mesures et incertitudes |
| **Planète** | C, D, lead | Logs de temps, VRAM, modèles choisis, réduction des traitements inutiles |

Le cahier des charges évalue, à la première sélection, **30 % problème réel**, **25 % innovation/IA**, **25 % éthique** et **20 % faisabilité**. La documentation de la détection du besoin et des six dimensions éthiques doit donc avancer en parallèle du code.

---

## 8. Checklist d'acceptation — première soumission

- [ ] Dépôt GitHub partagé, branches et responsables définis.
- [ ] Contrats `GlossEntry`, `MotionAsset`, `AudioFeatures`, `AnimationPlan` validés.
- [ ] Au moins un asset signé d'usage autorisé.
- [ ] Registre gloss → motion et `sources.csv` traçables.
- [ ] Une animation 3D jouée de manière reproductible.
- [ ] Endpoint français → lookup des signes supportés, erreurs inconnus testées.
- [ ] Une expérience d'analyse audio documentée (si P1 atteint).
- [ ] Statut « expérimental / non validé » clairement visible quand nécessaire.
- [ ] Problème terrain expliqué, vérifications réelles ou limites de validation indiquées.
- [ ] Six dimensions éthiques traitées.
- [ ] Slides en français.
- [ ] Vidéo en français de 1–2 minutes.
- [ ] README en français détaillé + code, même incomplet.
- [ ] Remise avant **15 novembre 2026, 23:59 (GMT+1)**.

---

## 9. Tableau d'affectation initial

| Ticket | Owner | Premier résultat attendu | Test de passage |
|---|---|---|---|
| `DATA-01` | A | Gloss registry minimal + source registry | Un gloss résout une référence documentée |
| `MOTION-01` | B | Clip animé jouable | Lecture 3D sans l'enregistrement brut |
| `NLP-01` | C | Lemmatisation + lookup | Entrée connue → ID, entrée inconnue → erreur claire |
| `AUDIO-01` | D | Tests emotion2vec+ et prosodie | JSON daté et visualisable |
| `CORE-01` | Lead | Premier vertical slice | Texte → gloss → mouvement → avatar |
| `VALIDATION-01` | Produit | Données terrain et éthique | Témoignages/sources documentés, limites signalées |

**Premier point de synchronisation :** Team A et Team B définissent ensemble l'identifiant de mouvement et le format jouable. Team C utilise temporairement une base mock correspondant à ce schéma. Team D avance de façon autonome avec des enregistrements de démonstration autorisés.

---

## 10. Bibliographie et ressources de départ

### Ressources techniques

1. **Animics** — UPF GTI, éditeur d'animation signé : https://github.com/upf-gti/animics
2. **Performs** — UPF GTI, synthèse/lecture d'avatars : https://github.com/upf-gti/performs
3. **MediaPipe Holistic** — suivi corps, mains et visage : https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker
4. **emotion2vec / emotion2vec+** — représentations de l'émotion vocale : https://github.com/ddlBoJack/emotion2vec
5. **spaCy French** — modèles de traitement du français : https://spacy.io/models/fr
6. **librosa** — analyse audio : https://librosa.org/
7. **Parselmouth** — analyse acoustique Praat : https://parselmouth.readthedocs.io/
8. **Spoken-to-Signed Translation** — baseline : https://github.com/sign-language-processing/spoken-to-signed-translation
9. **Faster-Whisper** — transcription : https://github.com/SYSTRAN/faster-whisper

### Données et corpus

10. **Dicta-Sign Concepts French** — lexique illustré en LSF : https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_fre.html
11. **Dicta-Sign-LSF-v2** — corpus de recherche : https://hdl.handle.net/11403/dicta-sign-lsf-v2/v1
12. **Rosetta-LSF** — corpus français–LSF / avatar : https://hdl.handle.net/11403/rosetta-lsf/v3
13. **MEDIAPI-SKEL** — vidéos/landmarks recherche : https://hdl.handle.net/11403/mediapi-skel/v2

**Vérifier séparément** : disponibilité des ressources, droit de téléchargement, licence de données, droits des personnes filmées, droits des dérivés et réutilisation commerciale. Les ressources LSF France et LSF-CH Suisse ne sont pas automatiquement interchangeables.

### Travaux scientifiques

14. **Dicta-Sign-LSF-v2**, ressource de référence sur l'annotation en LSF et ses défis : https://aclanthology.org/2020.lrec-1.740.pdf
15. **Muanji et al. (2026)**, *Signing Avatar Synthesis via a Hybrid T5-GAN Approach: Automatic Speech-to-French Sign Language Translation*, Procedia Computer Science, vol. 290, pp. 578–584. DOI : https://doi.org/10.1016/j.procs.2026.08.433 — article fourni par l'équipe.
16. **PGloss-LSF** — annotation formelle pour la synthèse de LSF : https://www.sign-lang.uni-hamburg.de/lrec/pub/26046.html
17. **Facial Expressions for Sign Language Synthesis using FACSHuman and AZee** : https://aclanthology.org/2024.signlang-1.39/

---

## 11. Conclusion opérationnelle

1. **Commencer par la donnée et l'animation :** une entrée gloss autorisée et un clip 3D jouable.
2. **Relier immédiatement le lookup au player :** prouver une chaîne end-to-end limitée.
3. **Faire progresser l'audio indépendamment :** emotion2vec+ et prosodie produisent des mesures, pas encore une animation signée valide.
4. **N'ajouter l'expression qu'avec des contraintes :** préserver la grammaire LSF et tester la compréhension auprès de signants compétents.
5. **Ne pas surpromettre :** le système est un prototype expérimental d'assistance à l'accessibilité audiovisuelle, non un traducteur LSF autonome universel.
6. **Conserver les trois livrables obligatoires à temps :** slides, vidéo française 1–2 min et GitHub commenté.

**Formulation du projet pour le pitch :**

> *SignaCulture AI explore une chaîne de synthèse de la langue des signes française fondée sur des mouvements réels : des glosses reliés à une bibliothèque d'animations 3D, enrichis expérimentalement par les caractéristiques prosodiques et émotionnelles de la voix. La conception privilégie la traçabilité des ressources, le contrôle humain et la validation linguistique.*
