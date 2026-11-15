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
