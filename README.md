# SignaCulture AI — Synthèse 3D en LSF Assistée par la Prosodie Vocale

[![CI](https://github.com/signaculture-ai/signaculture-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/signaculture-ai/signaculture-ai/actions/workflows/ci.yml)
[![Licence](https://img.shields.io/badge/licence-CC--BY--NC--SA--4.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node-20.x-green.svg)](https://nodejs.org/)

> **Projet :** SignaCulture AI — Génération d'animations 3D en Langue des Signes Française (LSF), enrichies par l'analyse de la prosodie et de l'expressivité vocale.  
> **Cadre :** Hackathon IA Éthique 360 (13–25 novembre 2026) — Première remise : 15 novembre 2026  
> **Avertissement de transparence :** Ce projet constitue un prototype de recherche en cours de développement. Les fonctionnalités présentées correspondent à un MVP vérifiable. Aucune validation humaine native par des personnes sourdes, ni aucune publication publique définitive ou enregistrement vidéo finalisé n'est revendiqué comme achevé à ce jour.

---

## 1. Synthèse du Projet & Vision

SignaCulture AI a pour objectif de faciliter la médiation culturelle accessible en explorant une chaîne de synthèse gestuelle en LSF fondée sur des mouvements réels capturés, enrichie expérimentalement par les dynamiques acoustiques et prosodiques de la parole humaine (F0 estimé, énergie RMS, pauses).

### Principes directeurs :
1. **Primauté de la grammaire LSF :** La langue des signes française possède sa propre syntaxe spatiale et tridimensionnelle. La lemmatisation mot-à-mot ne constitue pas une traduction LSF, mais un simple indexeur de concepts contrôlés.
2. **Priorité aux marqueurs non manuels :** Les mouvements du visage, des sourcils et du regard sont grammaticaux en LSF ; ils prévalent absolument sur toute modulation expressive issue de la voix.
3. **Humain dans la boucle et Gating strict :** Tout rendu destiné à la publication est bloqué par défaut (`HTTP 403`) tant que des revues linguistique, cinématique et juridique n'ont pas été formellement enregistrées.
4. **Fixtures Synthétiques Documentées :** L'ensemble des six mouvements actuels correspond à des fixtures techniques synthétiques (`signaculture-v1`), formellement non validées en LSF.

---

## 2. Architecture & Contrat d'Intégration

Le projet s'appuie sur une architecture découplée régie par le [Contrat d'intégration v1](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/INTEGRATION_CONTRACT.md) :
- **Backend :** API Python FastAPI (port `8000`), endpoints JSON en `snake_case`, temps exprimés en **millisecondes**.
- **Frontend :** Application React / Vite / TypeScript (port `5173`) avec moteur de rendu 3D Three.js sur le rig squelettique normalisé `signaculture-v1`. Proxy Vite `/api` redirigeant vers le backend.

```text
Texte FR ──> spaCy / Matcher ──> Glosses LSF ──> Lookup Mouvements (6 fixtures synthétiques)
                                                                 │
Audio WAV ──> DSP CPU (F0 estimé / RMS / Pauses) ───────────────┼──> Plan d'Animation 3D
                                                                 │         │
                                                                 ▼         ▼
                                                         Revue Humaine ──> Rendu Gated (403 si non validé)
```

### Endpoints Contractuels Clés :
- `GET /health` : Contrôle de disponibilité (`{"status": "ok"}`).
- `GET /glossary` : Liste des 6 entrées de gloss contrôlées (`GlossEntry`).
- `GET /motions/{motion_id}` : Métadonnées et statut d'un mouvement (`MotionAsset`).
- `GET /assets/{filename}` : Distribution sécurisée des fichiers JSON de mouvement approuvés pour prévisualisation locale.
- `POST /analyze-text` & `POST /gloss-lookup` : Analyse textuelle, appariement d'expressions et détection explicite des mots inconnus (`unknowns`).
- `POST /compose-animation` : Génération d'un plan d'animation (`AnimationPlan`) avec `render_allowed_for_publication: false`.
- `POST /analyze-audio` : Analyse acoustique déterministe sur CPU d'un fichier WAV multipart (estimation de F0, énergie RMS, détection des silences).
- `POST /review` : Enregistrement persistant d'une décision de révision dans SQLite.
- `POST /render` : Rendu d'aperçu (`preview`) ou blocage strict de publication (`publication`).

---

## 3. Commandes Locales Reproductibles

L'environnement utilise `python3` (et `uv` si présent), ainsi que Node.js / npm.

### Installation de l'Environnement
```bash
# Configuration automatique du venv Python (.venv) et des paquets npm frontend
make setup
# ou alternativement :
bash scripts/setup.sh
```

### Lancement du Développement Local
```bash
# Démarre le backend FastAPI (port 8000) et le frontend Vite (port 5173, --strictPort)
make dev
# ou alternativement :
bash scripts/dev.sh
```

### Exécution des Tests
```bash
# Exécute les tests de schémas, tests backend, tests experiments, build frontend et smoke tests
make test
# ou alternativement :
bash scripts/test.sh

# Lancement ciblé de la suite de smoke tests d'intégration contre l'API réelle :
make smoke
# ou avec URL spécifique :
python3 scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

---

## 4. Statut d'Avancement Réel : Actual vs Pending

Conformément à nos exigences éthiques de vérité scientifique, nous distinguons rigoureusement ce qui est opérationnel de ce qui est planifié :

| Composant | Statut Réel Actuel (Prototype) | Statut Cible / En Attente | Référence |
|---|---|---|---|
| **API & Contrats** | **Validé** : Endpoints opérationnels et vérifiés par les tests d'intégration (7/7 passés). | Déploiement d'infrastructure haute disponibilité | [INTEGRATION_CONTRACT.md](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/INTEGRATION_CONTRACT.md) |
| **Gating de Rendu** | **Opérationnel** : Blocage HTTP 403 strict sur toute tentative de publication non approuvée. | Signature cryptographique des rendus validés | [validation.md](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/validation.md) |
| **Analyse Audio (CPU)** | **Opérationnel** : DSP déterministe sur CPU (RMS, pauses, F0 estimé testé à ~220 Hz et ~440 Hz). | Modèle local emotion2vec+ quantifié | [methodology.md](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/methodology.md) |
| **Lexique & Mouvements** | **Opérationnel** : 6 concepts contrôlés et 6 fixtures cinématiques synthétiques non validées LSF. | Extension à 50+ concepts issus de corpus autorisés | [data_governance.md](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/data_governance.md) |
| **Validation Humaine LSF** | **En attente** : Protocole méthodologique formalisé ; relecteurs réels non encore mobilisés. | Sessions d'évaluation formelles avec des locuteurs sourds natifs | [validation.md](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/validation.md) |
| **Vidéo de Démonstration** | **En attente** : Scénario et narration minutés rédigés (90s) ; captation studio à réaliser. | Enregistrement de la vidéo finale de démo | [demo_narration.md](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/demo_narration.md) |
| **Dépôt GitHub Public** | **En attente** : Configuration Git et CI locales prêtes ; mise en ligne sous contrôle du lead. | Publication publique du code source | [ci.yml](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/.github/workflows/ci.yml) |

---

## 5. Documentation du Projet

- [Gouvernance des Données & Traçabilité](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/data_governance.md)
- [Cadre Éthique 360° & Responsabilité](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/ethics.md)
- [Méthodologie Scientifique & Positionnement T5-GAN](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/methodology.md)
- [Protocole de Validation & Gating de Publication](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/validation.md)
- [Organisation Multi-Worktree](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/team_worktree_organization.md)
- [Diapositives de Présentation (Markdown)](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/slides.md)
- [Script de Narration de la Démo](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/demo_narration.md)
- [Plan de Développement Initial](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/SignaCulture_AI_Plan_Developpement.md)
- [Contrat d'Intégration Faisant Foi](file:///home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-gemini-2/docs/INTEGRATION_CONTRACT.md)

---

## 6. Organisation en Worktrees de Développement

Le projet a été développé en parallèle sur 4 worktrees Git sous la coordination du Lead Orchestrateur :
1. **Worktree 1 (`feat/architecture`) :** Schémas Pydantic du contrat, registres de données et manifestes cinématiques.
2. **Worktree 2 (`feat/backend`) :** API FastAPI, NLP spaCy, analyse du signal audio DSP sur CPU, base SQLite de revues.
3. **Worktree 3 (`feat/frontend`) :** Interface React, lecteur 3D Three.js, visualiseur de signal et panneau de révision.
4. **Worktree 4 (`feat/qa`) :** Configuration racine, GitHub Actions CI, suite de tests d'intégration, outillage et documentation complète.
