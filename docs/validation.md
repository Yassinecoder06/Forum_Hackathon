# Stratégie de Validation et Gating de Publication — SignaCulture AI

> **Document :** Protocole de validation et contrôle qualité v1.0  
> **Cadre :** Hackathon IA Éthique 360  
> **Avertissement Formel :** Aucune validation humaine par des signants natifs n'est déclarée comme achevée dans ce document. Le protocole décrit les garde-fous automatisés en place et la méthodologie de certification humaine en attente.

---

## 1. Niveaux de Contrôle Qualité

Le système SignaCulture AI implémente une validation en quatre couches successives garantissant qu'aucune animation non vérifiée ne puisse être publiée comme accessible en LSF :

```mermaid
graph TD
    A["Niveau 1 : Validation Technique & Schémas"] --> B["Niveau 2 : Validation Cinématique & Rig"]
    B --> C["Niveau 3 : Validation Juridique & Droits"]
    C --> D["Niveau 4 : Validation Linguistique Humaine LSF"]
    D --> E["Déverrouillage Publication (Gating)"]
```

### Niveau 1 — Validation Technique & Schémas (Automatisée en CI)
- **Contrat d'intégration :** Strict respect de [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md).
- **Gestion des types et temps :** Tous les timestamps sont en millisecondes entières.
- **Suite de Smoke Tests (`scripts/smoke_test.py`) :**
  - Vérification de l'API `/health` (HTTP 200).
  - Validation du lexique `/glossary` et des assets `/motions/{id}`.
  - Détection systématique des tokens inconnus lors du lookup textuel (`translation_validated: false`).
  - Analyse du signal audio WAV synthétique (440 Hz + silence + 220 Hz) : vérification de l'estimation de F0 à ~220 Hz et détection effective des pauses.
  - Sécurité des assets : refus d'accès aux fichiers non enregistrés ou tentatives d'évasion de chemin (`GET /assets/...`).

### Niveau 2 — Validation Cinématique & Rig 3D
- **Conformité `signaculture-v1` :** Présence des 9 articulations indispensables (`head`, `neck`, `hips`, épaules, coudes, poignets).
- **Repère spatial :** Coordonnées en mètres (Y vers le haut, X droite, Z avant).
- **Détection des discontinuités :** Vérification de l'absence de téléportations d'articulations ou d'inversions droite/gauche lors de l'enchaînement des segments.

### Niveau 3 — Validation Juridique & Droits d'Auteur
- **Attribution des sources :** Chaque mouvement doit être rattaché à un `source_id` documenté dans `data/sources.csv`.
- **Statut des droits (`rights_status`) :** Valeurs autorisées : `"cleared"` ou `"pending"`.
- **Règle d'or :** Tant que `rights_status != "cleared"`, tout rendu public est impossible.

### Niveau 4 — Validation Linguistique LSF (Humains dans la boucle)
- **Critères d'évaluation :**
  1. Configuration des mains (chiremes / cherèmes).
  2. Orientation de la paume et direction du mouvement.
  3. Emplacement dans l'espace de signation (thorax, visage, espace neutre).
  4. Marqueurs non manuels : sourcils, direction du regard, mimique faciale grammaticale.
- **Réviseur assermenté :** Enregistrement dans SQLite via `POST /review` (`reviewer`, `status`, `notes`).

---

## 2. Système de Gating de Publication (`POST /render`)

Le contrat d'intégration distingue formellement deux finalités de rendu :

```python
# Comportement garanti par le contrat
if purpose == "preview":
    # Autorise l'affichage dans le lecteur 3D local pour revue technique
    return {"status": "preview_ready", "render_allowed_for_publication": False}

elif purpose == "publication":
    # VERROUILLÉ : bloque la génération de vidéo finale
    if not (linguistic_review == "approved" and animation_review == "approved" and rights_status == "cleared"):
        raise HTTPException(status_code=403, detail="Publication blocked until ALL approvals and rights checked")
```

### Règle Anti-Bypass
L'approbation isolée d'un réviseur via `POST /review` **ne peut pas outrepasser** le statut des droits ou la revue d'animation. L'interdiction de publication reste active tant que l'ensemble des conditions n'est pas réuni.

---

## 3. Matrice Statut Réel (Actuel) vs En Attente (Pending)

| Processus de Validation | Statut Réel Actuel (MVP) | Statut Cible / En Attente | Preuve / Livrable Associé |
|---|---|---|---|
| **Contrôle d'Intégration API** | Opérationnel et testé à 100% via `scripts/smoke_test.py` | Surveillance continue en production | `python3 scripts/smoke_test.py` (7/7 tests validés) |
| **Gating de Rendu Publication** | Opérationnel : HTTP 403 strict sur toute tentative de publication | Déblocage conditionné par double validation certifiée (relecteur sourd + droits) | Testé dans la suite de smoke tests |
| **Sécurité d'Accès aux Assets** | Opérationnel : blocage 403/404 des assets non approuvés | Quotas et contrôle de session | Endpoint `GET /assets/{filename}` |
| **Revue Linguistique LSF** | **En attente** (protocole défini, statut `pending` par défaut) | Séances d'évaluation avec experts sourds natifs | Grille d'évaluation chiremique et spatiale |
| **Validation Droits Corpus** | Fixtures synthétiques libres (CC0), sources réelles tracées | Audit juridique formalisé des licences tierces | Registre `data/sources.csv` |
| **Enregistrement Démo Vidéo** | Script minuté formalisé (`docs/demo_narration.md`) | Enregistrement de la démo 1–2 minutes | Script prêt pour captation finale |
| **Dépôt Public GitHub** | Dépôt Git local et CI configurés | Mise en ligne publique après validation lead | `.github/workflows/ci.yml` configuré |
