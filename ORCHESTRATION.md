# Current state — 2026-11-15

Only the lead root checkout and `.worktrees/wt-codex2` remain. Only codex2 (GPT-6.1 Sol medium, pane wS:p4) is delegated work. No Gemini or codex1 agents are used. Useful prior branches were merged. The real capture/audio/studio rewrite and research milestones are integrated on main. All 447 inventoried lexical videos were retrieved; 440 motion assets are available and seven require tracking review. Three measured text-model experiments failed to provide useful general translation. See README.md and research/IMPLEMENTATION_STATUS.md for current results.

The following log is historical and describes earlier development stages.

# SignaCulture AI — exécution par worktrees

Le plan source est conservé dans `SignaCulture_AI_Plan_Developpement.md`.
L'implémentation utilise quatre tâches isolées par Git, coordonnées dans Herdr.

Workspace créé : **Agents 2**, ID `wS` (Agents 1 existait déjà).

| Tâche | Tab | Pane ID | Commande | Modèle effectif | Worktree | Branche |
|---|---|---|---|---|---|---|
| Frontend et lecteur 3D | Gemini 3.8 Flash high | `wS:p1` | `agy1` | Gemini 3.8 Flash high | `.worktrees/wt-gemini-1` | `feat/frontend` |
| QA, outillage, documentation | Gemini 3.8 Flash high | `wS:p2` | `agy2` | Gemini 3.8 Flash high | `.worktrees/wt-gemini-2` | `feat/qa` |
| Architecture, schémas, données, extraction | GPT 6.1 Sol low | `wS:p3` | `codex1` | GPT 6.1 Sol low | `.worktrees/wt-luna-1` | `feat/architecture` |
| API, lookup, algorithmes audio, tests | GPT 6.1 Sol low | `wS:p4` | `codex2` | GPT 6.1 Sol low | `.worktrees/wt-luna-2` | `feat/backend` |

Les noms historiques `wt-luna-*` sont conservés ; les deux agents Codex ont été changés en cours de session vers **GPT 6.1 Sol low**, à la demande de l'utilisateur. Les IDs Herdr sont propres à cette session.

Contrat d'intégration : `docs/INTEGRATION_CONTRACT.md`. Chaque tâche possède des chemins distincts. Le lead examine les diffs, exécute les contrôles, intègre les commits approuvés et redistribue le travail si un agent atteint son quota. Aucun workspace préexistant n'est fermé.

Le dépôt est initialisé dans ce dossier pour ne pas modifier le dépôt parent `/home/jmal/Projects` ni ses changements préexistants. Les worktrees et fichiers de session locaux sont ignorés.

## Évaluation des modèles

Les sources officielles décrivent Gemini 3.8 Flash comme un modèle rapide pour la programmation, les tâches agentiques et les entrées multimodales : https://deepmind.google/models/gemini/flash/ et https://deepmind.google/models/model-cards/gemini-3-8-flash/.
OpenAI décrit GPT-6 Luna comme une option économique pour la programmation et l'itération : https://openai.com/index/introducing-gpt-6-sol-and-luna/.
Ces descriptions ne constituent pas une comparaison directe sur ce projet. Le mapping initial demandé a été conservé, puis les deux rôles Codex ont été affectés à Sol low suivant la correction utilisateur.

## Limites de validation

Un mouvement synthétique original peut vérifier le lecteur et les contrats, mais ne démontre pas une traduction LSF correcte. Le corpus Dicta-Sign et les modèles optionnels nécessitent une installation et des ressources appropriées. L'approbation linguistique et la validation d'animation restent des étapes humaines ; leur statut ne doit pas être inventé.

## Intégration et vérification du lead

Les quatre branches ont été intégrées sur `main` après revue. Les erreurs d'interface identifiées (routage des assets, format de réponse de rendu, mise à jour des revues, lecture limitée au premier clip et remplacement silencieux lors d'erreurs API) ont été corrigées. Les worktrees sont conservés pour reprise ; leurs branches représentent leurs contributions, pas nécessairement le dernier état intégré.

Vérifications locales : 54 tests Python, dont inférence spaCy française réelle ; lint TypeScript/React ; build de production ; 7 smoke tests contre FastAPI réel ; parcours navigateur réel sur desktop et mobile. Un avertissement de dépréciation Starlette/httpx et un avertissement de taille de bundle Vite restent présents. Aucun quota épuisé n'a été observé.

Restent à effectuer : collecte autorisée de vrais signes, validation linguistique et cinématique avec des signants compétents, articulation complète des doigts/visage, exécution du modèle emotion2vec, retargeting Animics testé et export vidéo signé côté application. Aucun de ces résultats n'est revendiqué par les fixtures.
