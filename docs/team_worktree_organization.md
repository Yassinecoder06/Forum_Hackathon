# Organisation actuelle

Deux worktrees Git sont conservés, conformément à la dernière demande utilisateur :

| Responsable | Worktree | Branche | Périmètre |
|---|---|---|---|
| Lead | racine du dépôt | main | API, capture, audio, avatar, interface, ingestion, intégration |
| codex2 — GPT-6.1 Sol medium | .worktrees/wt-codex2 | feat/research-training | training/, research/, data/catalog/, data/text_to_gloss/ |

Aucun agent Gemini ni codex1 n’est utilisé. Les anciens travaux utiles ont été fusionnés, leurs quatre worktrees supprimés. Le seul agent délégué est dans le panneau Herdr `wS:p4`, espace Agents 2. Les commandes ciblent cet identifiant, sans changer le focus.

Chaque propriétaire travaille dans son périmètre ; le lead inspecte les commits, exécute les vérifications et fusionne sur main. Les environnements ML restent isolés du runtime API. Les processus de test ne terminent que leurs propres processus, jamais un service tiers par port ou un pkill global.
