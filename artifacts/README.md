# Livrables locaux

Les captures `browser/studio.png`, `audio.png`, `library.png`, `mobile.png` et `research.png` montrent le studio actuel. `browser/preview.webm` est un vrai export de l’avatar, produit par le navigateur pendant le test automatisé. Les crédits des mouvements restent affichés dans l’interface et l’export. Les mouvements reconstruits demandent une revue LSF ; ce ne sont pas des traductions validées.

La bibliothèque de cette session contient 440 captures réelles, issues de 447 concepts récupérés. Sept captures nécessitent une revue du suivi des mains et ne sont pas enregistrées. Les données volumineuses sont locales et ignorées par Git ; voir `data/source_inventory/ingestion_report.json`.

Les expériences textuelles sont dans `research/runs/` : T5-small CPU et deux mT5-small RTX 4060. Les résultats sont négatifs pour la traduction, avec prédictions et limites conservées.

`SignaCulture_Presentation.pdf`, `SignaCulture_Demo_FR.mp4` et `demo_narration.txt` sont des **livrables historiques du premier prototype**, antérieurs au studio actuel et à l’ingestion réelle. Ils ne décrivent pas les capacités actuelles ; consulter le README racine et les nouvelles captures.

Vérification actuelle : 58 tests API/schémas/mouvement, 12 tests de recherche, lint et build frontend, tests navigateur contre les vrais services. Aucun résultat état de l’art ni validation par des signants n’est revendiqué.
