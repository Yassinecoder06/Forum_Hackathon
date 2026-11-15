# Solution en cours : séparer compréhension française et production LSF

Les essais mT5 ont échoué avec 61 phrases d’entraînement. L’étape suivante réutilise la compréhension d’un modèle préentraîné pour proposer des concepts existants, puis laisse l’utilisateur corriger leur ordre et leur sélection avant une prévisualisation. Cela améliore potentiellement l’accès à la bibliothèque, sans prétendre apprendre une langue signée à partir de 61 exemples.

Le studio possède un éditeur de concepts réels : aucune fixture synthétique ni identifiant inconnu ne peut être envoyé à cette composition. L’ordre et les répétitions sont préservés. Les propositions sont des candidats, jamais des approbations LSF. Négation, questions, nombres et informations non représentées doivent rester visibles. Une sélection manuelle ne prouve ni équivalence sémantique ni correction grammaticale.

Le téléchargement de Qwen2.5-1.5B-Instruct a été interrompu après un blocage réseau. Le modèle Qwen3.5-9B GGUF déjà présent localement est évalué avec llama-server et offload partiel GPU ; aucun modèle cloud n’est invoqué. L’expérience locale et ses limites sont conservées dans research/. Les résultats de sélection lexicale ne se comparent pas aux scores du benchmark de traduction STK.

Nouvel audit : eglantinefonrose/lsf-fr-gloss, révision aadc033abb9da95a921458181c87f902d311a756, annonce 408 exemples train et 45 validation entièrement synthétiques, non validés par un locuteur LSF. Carte récupérée via hf le 15 novembre 2026. Cette source n’est pas ajoutée aux annotations expertes et ne résout pas le manque de vérité terrain : https://huggingface.co/datasets/eglantinefonrose/lsf-fr-gloss.

Les ressources LSFB (Belgique) et LSF-CH (Suisse) trouvées lors de la recherche restent des langues/variantes distinctes. Leur disponibilité ne justifie pas de les relabeller comme France-LSF. Pour une traduction générale, un corpus France-LSF de phrases alignées et une évaluation par des locuteurs restent nécessaires.
