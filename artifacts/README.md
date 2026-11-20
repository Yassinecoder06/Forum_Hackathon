# Livrables locaux

Les captures `browser/studio.png`, `audio.png`, `library.png`, `mobile.png` et `research.png` montrent le studio actuel. `browser/preview.webm` est un vrai export de l’avatar, produit par le navigateur pendant le test automatisé. Les crédits des mouvements restent affichés dans l’interface et l’export. Les mouvements reconstruits demandent une revue LSF ; ce ne sont pas des traductions validées.

La bibliothèque de cette session contient 440 captures réelles, issues de 447 concepts récupérés. Sept captures nécessitent une revue du suivi des mains et ne sont pas enregistrées. Les données volumineuses sont locales et ignorées par Git ; voir `data/source_inventory/ingestion_report.json`.

Les expériences textuelles sont dans `research/runs/` : T5-small CPU et deux mT5-small RTX 4060. Les résultats sont négatifs pour la traduction, avec prédictions et limites conservées.

`SignaCulture_Presentation.pdf`, `SignaCulture_Demo_FR.mp4` et `demo_narration.txt` sont des **livrables historiques du premier prototype**, antérieurs au studio actuel et à l’ingestion réelle. Ils ne décrivent pas les capacités actuelles ; consulter le README racine et les nouvelles captures.

`browser/concepts.png`, `semantic.png` et `concept-review.json` documentent la sélection manuelle, la proposition locale réelle et la fiche de revue non validée.

Vérification actuelle : 65 tests API/schémas/mouvement, 18 tests de recherche, un défaut de détection de prénom conservé en xfail strict, lint et build frontend, tests navigateur contre les vrais services. Aucun résultat état de l’art ni validation par des signants n’est revendiqué.

## Repaired player

`SignaCulture_Repaired_Preview.mp4` and `browser/repaired-preview.webm` are actual recordings of two real captured signs with the new articulated Noa character and dedicated clip transitions. `browser/repaired-studio.png` shows the repaired rendering at a captured signing pose. Avatar provenance and pending publication/redistribution rights are in `docs/avatar/AVATAR_PROVENANCE.md`.

Current verification: 67 API/schema/motion tests, 24 research tests, one documented strict xfail for name detection, frontend motion/IK checks, lint/build, seven live API checks and the real browser workflow. The research tab separately shows the trained CGAN's intra-sign experiment; the player does not claim validated learned co-articulation.
