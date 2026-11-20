# Gouvernance des Données — SignaCulture AI

> **Statut du document :** Spécification opérationnelle et registre de conformité v1.0  
> **Date :** 15 novembre 2026  
> **Cadre :** Hackathon IA Éthique 360  

---

## 1. Principes Fondamentaux de Gouvernance

SignaCulture AI traite des données linguistiques en langue des signes française (LSF), des signaux audio vocaux et des données géométriques de capture de mouvement (mocap / landmarks 3D).

Conformément au [Contrat d'intégration](../architecture/INTEGRATION_CONTRACT.md), la gouvernance des données repose sur quatre piliers stricts :

1. **Traçabilité absolue des identifiants :** Chaque entité de gloss (`gloss_id`), de mouvement (`motion_id`) et de ressource d'origine (`source_id`) est immuable et référencée dans les registres du dépôt.
2. **Cloisonnement des données personnelles :** Les vidéos brutes montrant des visages ou des personnes identifiables ne sont jamais stockées sur des dépôts publics sans consentement explicite et archivage séparé.
3. **Interdiction de scraping massif :** Aucun téléchargement automatique non autorisé de portails de recherche (ex. Dicta-Sign, MEDIAPI-SKEL) n'est exécuté.
4. **Distinction formelle entre données synthétiques et données validées :** Tout asset de test est explicitement labellisé `synthetic: true` et `review_status: pending`.

---

## 2. Registre des Sources et Droits

| Identifiant Source | Nom de la Ressource | Nature | Licence / Droits | Usage Autorisé | Statut d'Intégration |
|---|---|---|---|---|---|
| `SRC_DICTASIGN_CONCEPTS` | Dicta-Sign French Concepts | Lexique conceptuel LSF | CC-BY-NC-SA 4.0 / Consultation recherche | Définition des concepts et glosses textuels | Référencé (non téléchargé en masse) |
| `SRC_DICTASIGN_LSF_V2` | Dicta-Sign-LSF-v2 (Ortolang) | Corpus vidéo annoté | Licence académique recherche | Étude des structures de transition | Soumis à autorisation préalable |
| `SRC_ORIGINAL_FIXTURES` | Fixtures Synthétiques SignaCulture | JSON cinématique généré (rig-v1) | CC0-1.0 (Public Domain) | Tests techniques et pipeline CI/CD | **Actuel (Intégré dans le dépôt)** |
| `SRC_FIELD_CONTRIBUTIONS` | Enregistrements signants consentis | Vidéos mocap LSF natives | Accord éclairé spécifique | Enrichissement de la bibliothèque | **En attente (Protocole défini)** |

---

## 3. Schéma et Traçabilité des Métadonnées

Les métadonnées respectent la structure unifiée définie dans `data/glossary.json` et `data/motions_manifest.json` :

```json
{
  "gloss_id": "LSF_CONCEPT_001",
  "concept_fr": "bonjour",
  "gloss": "BONJOUR",
  "language": "LSF",
  "variant": "concept français non validé LSF",
  "source_id": "SRC_ORIGINAL_FIXTURES",
  "motion_id": "SYNTHETIC_001",
  "aliases": ["bonjour"],
  "review_status": "pending"
}
```

```json
{
  "motion_id": "SYNTHETIC_001",
  "gloss_id": "LSF_CONCEPT_001",
  "path": "data/approved_motion_assets/SYNTHETIC_001.json",
  "rig_version": "signaculture-v1",
  "fps": 10,
  "duration_ms": 1000,
  "license": "CC0-1.0",
  "quality_status": "synthetic_technical_fixture_not_validated_LSF",
  "rights_status": "cleared",
  "animation_review": "pending",
  "synthetic": true
}
```

---

## 4. Matrice de Statut : Réel (Actuel) vs En Attente (Pending)

| Élément de Données | Statut Actuel (MVP) | Statut Cible / En Attente | Condition de Passage |
|---|---|---|---|
| **Glossaire LSF** | 6 concepts contrôlés (`bonjour`, `merci`, `culture`, etc.) labellisés non validés | Lexique étendu (50+ concepts) issu de corpus autorisés | Revue linguistique par experts sourds natifs |
| **Assets de mouvement** | Fixtures cinématiques synthétiques minimales (rig `signaculture-v1`) | Enregistrements mocap retargetés haute fidélité | Autorisations de droit à l'image et session mocap |
| **Gating de publication** | Blocage strict (`render_allowed_for_publication: false`) sur tout plan | Déblocage conditionnel après double approbation | Approbation humaine enregistrée + droits vérifiés |
| **Analyse vocale** | Traitement déterministe DSP (RMS, pitch F0, silence) | Expérimentation d'un modèle local emotion2vec+ | Validation du modèle sans surcharge VRAM |
| **Persistance des revues** | Base SQLite locale (`reviews`) avec traçabilité | Base centralisée sécurisée | Audit de sécurité et migration base de production |

### Ressources candidates vérifiées par le lead

Le [lexique Dicta-Sign](https://www.sign-lang.uni-hamburg.de/lr/compendium/lex/dictasignlex.html) mentionne un accord individuel pour les chercheurs. La fiche [Dicta-Sign-LSF-v2](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/dictasignlsfv2.html) mentionne CC BY-NC-SA 4.0 et un accès via inscription Ortolang. Ces ressources sont distinctes ; aucune autorisation de ce projet n'est présumée. Le registre les conserve comme candidates avec `rights_status: pending`, sans asset importé.
