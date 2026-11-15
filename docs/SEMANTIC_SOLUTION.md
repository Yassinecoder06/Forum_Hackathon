> Current evidence update: the supplied seven-page `research_paper.pdf` has now been read. See [verified comparison](../research/PAPER_B_VERIFIED_REVIEW.md); no GAN is currently implemented.

# Solution en cours : séparer compréhension française et production LSF

Les essais mT5 ont échoué avec 61 phrases d’entraînement. L’étape suivante réutilise la compréhension d’un modèle préentraîné pour proposer des concepts existants, puis laisse l’utilisateur corriger leur ordre et leur sélection avant une prévisualisation. Cela améliore potentiellement l’accès à la bibliothèque, sans prétendre apprendre une langue signée à partir de 61 exemples.

Le studio possède un éditeur de concepts réels : aucune fixture synthétique ni identifiant inconnu ne peut être envoyé à cette composition. L’ordre et les répétitions sont préservés. Les propositions sont des candidats, jamais des approbations LSF. Négation, questions, nombres et informations non représentées doivent rester visibles. Une sélection manuelle ne prouve ni équivalence sémantique ni correction grammaticale.

Le téléchargement de Qwen2.5-1.5B-Instruct a été interrompu après un blocage réseau. Le modèle Qwen3.5-9B GGUF déjà présent localement est évalué avec llama-server et offload partiel GPU ; aucun modèle cloud n’est invoqué. L’expérience locale et ses limites sont conservées dans research/. Les résultats de sélection lexicale ne se comparent pas aux scores du benchmark de traduction STK.

Nouvel audit : eglantinefonrose/lsf-fr-gloss, révision aadc033abb9da95a921458181c87f902d311a756, annonce 408 exemples train et 45 validation entièrement synthétiques, non validés par un locuteur LSF. Carte récupérée via hf le 15 novembre 2026. Cette source n’est pas ajoutée aux annotations expertes et ne résout pas le manque de vérité terrain : https://huggingface.co/datasets/eglantinefonrose/lsf-fr-gloss.

Les ressources LSFB (Belgique) et LSF-CH (Suisse) trouvées lors de la recherche restent des langues/variantes distinctes. Leur disponibilité ne justifie pas de les relabeller comme France-LSF. Pour une traduction générale, un corpus France-LSF de phrases alignées et une évaluation par des locuteurs restent nécessaires.

## Utilisation locale

Après intégration de l’expérience mesurée, démarrer le serveur dédié avec `python3 scripts/start-concept-server.py`. La configuration dans `research/concept_planner_runtime.json` contient les chemins et paramètres réellement testés sur cette machine. Ce script ne télécharge pas de poids et n’arrête aucun autre service. Le modèle reste séparé des runtimes de capture/audio.

Dans Studio : ouvrir « Préparer et corriger les concepts », demander une proposition, vérifier chaque concept et le sens signalé comme non représenté, corriger l’ordre ou ajouter des signes par recherche, puis prévisualiser. Télécharger la fiche de revue permet de conserver la phrase française, les concepts sélectionnés et les points non représentés. Ces fichiers ne sont pas versés automatiquement dans un corpus d’entraînement. Une personne qualifiée doit vérifier la traduction et les marqueurs non manuels avant qu’un brouillon devienne une annotation experte.

## Premier résultat mesuré

Sur 24 cas logiciels rédigés avant inférence (pas des traductions LSF de référence), le candidat préentraîné obtient 13 séquences de concepts exactes, contre 12 pour le lookup littéral existant. F1 micro des concepts : 0,760 contre 0,708. Latence moyenne : 7,8 s, premier appel 23,7 s ; environ 5,2 Go de VRAM après chargement. Le même corpus avec les mêmes gardes est également rapporté pour ne pas attribuer au modèle les avertissements issus de règles déterministes.

Ce gain est faible et ne démontre pas une solution de traduction générale. La détection de noms a supprimé à tort certains concepts ordinaires, dont Bonjour. Le rapport et le code évalué restent figés ; une correction séparée du runtime ne reçoit pas rétroactivement ces scores. Le lookup reste le comportement par défaut. Le modèle n’est qu’une aide facultative pour proposer un brouillon éditable.
