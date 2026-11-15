# Organisation en Worktrees de Développement — SignaCulture AI

> **Document :** Spécification d'implémentation multi-worktree v1.1  
> **Objet :** Traduction opérationnelle de l'organisation d'équipe du plan initial en quatre worktrees Git isolés sous la direction d'un lead orchestrateur, sans modifier les objectifs du plan d'origine.  
> **Principe directeur :** Aucun chemin absolu codé en dur ; travail exclusif via des branches et des chemins relatifs au dépôt (`REPO_ROOT`).

---

## 1. Cartographie des Responsabilités et Modèles Effectifs par Worktree

Pour concilier parallélisme strict et cohérence d'intégration, l'implémentation est structurée en **quatre worktrees Git spécialisés** pilotés par le Lead Orchestrateur.

Conformément à la topologie de session et à la correction utilisateur :
- Les chemins historiques des worktrees (`.worktrees/wt-luna-1` et `.worktrees/wt-luna-2`) sont conservés pour assurer la continuité des environnements de travail.
- Les modèles effectifs pour les tâches d'architecture et de backend ont été mis à jour de Codex vers **GPT 6.1 Sol low**.
- Les tâches de frontend et de QA / documentation sont assurées par **Gemini 3.8 Flash high**.

| Tâche / Rôle | Modèle Effectif | Worktree (Relatif) | Branche Git | Périmètres et Fichiers Possédés | Responsabilités Clés |
|---|---|---|---|---|---|
| **Frontend et Lecteur 3D** | Gemini 3.8 Flash high | `.worktrees/wt-gemini-1` | `feat/frontend` | `frontend/**` | - Application React / Vite / TypeScript<br/>- Rendu WebGL Three.js sur rig `signaculture-v1`<br/>- Visualiseur audio et panneau de révision |
| **QA, Outillage & Documentation** | Gemini 3.8 Flash high | `.worktrees/wt-gemini-2` | `feat/qa` | Racine configs (`Makefile`, `pytest.ini`, etc.)<br/>`.github/**`<br/>`scripts/**`<br/>`docs/**` (hors architecture/contrat)<br/>`README.md` | - Scripts locaux reproductibles (`setup`, `dev`, `test`)<br/>- GitHub Actions CI unifié sans mocks<br/>- Suite de smoke tests contre endpoints réels<br/>- Documentation éthique, gouvernance, slides et pitch |
| **Architecture, Schémas & Données** | GPT 6.1 Sol low | `.worktrees/wt-luna-1` *(historique)* | `feat/architecture` | `backend/app/schemas/**`<br/>`data/**`<br/>`experiments/**`<br/>`docs/architecture.md` | - Schémas Pydantic du contrat d'intégration<br/>- Registres `glossary.json` et `motions_manifest.json`<br/>- Expérimentations d'extraction cinématique |
| **API, Lookup & Signal Audio** | GPT 6.1 Sol low | `.worktrees/wt-luna-2` *(historique)* | `feat/backend` | `backend/app/**` (hors schemas)<br/>`backend/tests/**` | - Application FastAPI (`app.main:app`)<br/>- Endpoints text-to-gloss et lookup d'inconnus<br/>- Traitement acoustique déterministe (DSP)<br/>- Base SQLite des revues et logique de gating |
| **Lead Orchestrateur** | Lead coordination | Dépôt racine | `main` | `docs/INTEGRATION_CONTRACT.md`<br/>`SignaCulture_AI_Plan_Developpement.md`<br/>`ORCHESTRATION.md` | - Contrat d'intégration faisant foi<br/>- Fusion et relecture des diffs entre branches<br/>- Contrôles d'intégration et gating de release |

```mermaid
flowchart TD
    Lead["Lead Orchestrateur (main)<br/>Contrat d'intégration & Fusion"]
    WT1["Worktree 1 : Architecture & Données<br/>GPT 6.1 Sol low (.worktrees/wt-luna-1)"]
    WT2["Worktree 2 : Backend & Signal<br/>GPT 6.1 Sol low (.worktrees/wt-luna-2)"]
    WT3["Worktree 3 : Frontend & Avatar 3D<br/>Gemini 3.8 Flash high (.worktrees/wt-gemini-1)"]
    WT4["Worktree 4 : QA, Outillage & Docs<br/>Gemini 3.8 Flash high (.worktrees/wt-gemini-2)"]

    Lead -->|INTEGRATION_CONTRACT.md| WT1 & WT2 & WT3 & WT4
    WT1 -.->|Schémas & Données| Lead
    WT2 -.->|API FastAPI & DSP| Lead
    WT3 -.->|UI React & Three.js| Lead
    WT4 -.->|CI, Tests & Docs| Lead
```

---

## 2. Règles d'Isolation et Workflow Git

1. **Isolation stricte :** Chaque agent travaille exclusivement dans son worktree assigné. Aucune modification directe sur les branches voisines.
2. **Chemins relatifs universels :** Les scripts et configurations s'appuient sur la racine Git dynamique (`REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"`) sans aucun chemin absolu codé en dur.
3. **Autorité du Contrat d'Intégration :** En cas d'ambiguïté entre branches, le document `docs/INTEGRATION_CONTRACT.md` détenu par le lead est la référence normative absolue.
4. **Protocole de Fusion :**
   - Chaque worktree committe ses fichiers de périmètre sans rebase ni merge non coordonné.
   - Le lead examine les branches et effectue l'intégration sur `main`.
   - La suite de smoke tests (`scripts/smoke_test.py`) valide l'ensemble intégré contre le backend FastAPI réel en fonctionnement.
