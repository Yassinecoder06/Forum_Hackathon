# SignaCulture AI — Présentation du Projet

> **Support :** Diapositives de présentation du projet (Markdown)  
> **Date :** 15 novembre 2026  
> **Équipe :** SignaCulture AI  

---

## Slide 1 — Titre & Vision
# SignaCulture AI
### Synthèse 3D en Langue des Signes Française (LSF) assistée par l'analyse vocale et régie par l'éthique

*Projet Hackathon IA Éthique 360*

> **Vision du projet :** Concevoir un outil d'assistance à la pré-production d'animations 3D en LSF, fondé sur des mouvements documentés, sous contrôle humain permanent, sans jamais feindre une validation linguistique non éprouvée.

---

## Slide 2 — Le Problème Réel
### L'accès aux contenus pour les personnes signantes

- **Constat :** Les sous-titres écrits en français ne remplacent pas la LSF, qui dispose d'une grammaire spatiale et visuelle autonome.
- **Enjeu de la synthèse gestuelle :** La génération automatique directe sans validation pose des défis critiques d'intelligibilité, de configurations manuelles inexactes et de respect grammatical.
- **Le besoin identifié :** Les médiateurs et créateurs de contenu recherchent des outils pour préparer des ébauches d'animation tout en garantissant un cadre de relecture humaine qualifiée avant toute diffusion.

---

## Slide 3 — Notre Démarche Scientifique
### Rendre la voix visuelle sans dénaturer la langue

- **Hypothèse de recherche :** L'intégration de caractéristiques prosodiques (rythme vocal, intensité, pauses) peut-elle enrichir la naturalité perçue d'un avatar 3D sans altérer la compréhension des signes ?
- **Principe fondamental :** La grammaire spatiale et les marqueurs non manuels (sourcils, regard, tête) **priment toujours** sur toute modulation issue de la voix.
- **Positionnement :** Une chaîne traçable reliant des concepts contrôlés à un rig squelettique 3D normalisé (`signaculture-v1`), avec séparation stricte entre le signal acoustique et les règles linguistiques.

---

## Slide 4 — L'Architecture Technique
### Une chaîne modulaire et vérifiable

```text
Texte FR ──> spaCy / Matcher ──> Glosses LSF ──> Lookup Mouvements
                                                        │
Audio WAV ──> DSP CPU (F0 estimé / RMS / Pauses) ───────┼──> Plan d'Animation 3D
                                                        │         │
                                                        ▼         ▼
                                                Revue Humaine ──> Rendu Gated (403)
```

1. **Text-to-Gloss :** Lemmatisation et matcher d'expressions avec détection explicite des termes inconnus (`unknowns`).
2. **Traitement Acoustique Réel (CPU) :** Estimation de la fréquence fondamentale (F0), calcul de l'énergie RMS et détection des pauses.
3. **Moteur 3D :** Interpolation cinématique sur squelette WebGL Three.js.

---

## Slide 5 — Le Prototype MVP Actuel
### Des fonctionnalités réelles et transparentes

- **Lexique contrôlé :** Résolution sur 6 concepts de démonstration (`bonjour`, `merci`, `culture`, etc.).
- **Fixtures cinématiques :** **Tous les 6 mouvements actuels sont des fixtures techniques synthétiques non validées en LSF**.
- **Analyse du signal audio :** Traitement testé sur signaux de référence démontrant l'estimation de F0 (~220 Hz / ~440 Hz) et la détection effective des silences.
- **Gating de publication :** Prévisualisation locale autorisée (`preview`), mais **publication strictement bloquée** (`HTTP 403`) tant que les validations et les droits ne sont pas confirmés.

---

## Slide 6 — Démarche Éthique : Gouvernance & Données
### Créateurs, Droits et Protection des Données

- **Créateurs (Humain dans la boucle) :** L'outil assiste la pré-production ; l'approbation humaine formelle via `POST /review` est requise.
- **Droits & Traçabilité :** Tout asset est associé à un identifiant de source et à une licence répertoriée (`data/sources.csv`). Aucun scraping massif non autorisé.
- **Données Personnelles :** Aucune vidéo brute de signants identifiables n'est diffusée sur le dépôt public ; les données biométriques sont strictement isolées.

---

## Slide 7 — Démarche Éthique : Culture LSF & Frugalité
### Respect linguistique, Vérité et Sobriété Numérique

- **Cultures et Langues :** Refus du calque mot-à-mot. Les limites de la lemmatisation comme approximation de lookup sont explicitement documentées.
- **Transparence :** Étiquetage `synthetic: true` et `review_status: pending` sur chaque asset du prototype.
- **Sobriété Numérique :** **Prototype testé et exécuté sur CPU standard** via un traitement du signal déterministe léger, évitant l'empreinte énergétique de modèles lourds non justifiés à ce stade.

---

## Slide 8 — Modèle et Hypothèse de Valeur Économique
### Une hypothèse à mesurer sur le terrain

- **Hypothèse de valeur :** Évaluer auprès des institutions culturelles et des créateurs si une ébauche animée semi-automatique réduit le temps global de réalisation de contenus adaptés par rapport à une modélisation manuelle intégrale.
- **Modèle envisagé :**
  - Outil de pré-composition pour structures de médiation culturelle.
  - Rémunération de professionnels sourds pour la constitution, la relecture et la validation certifiée des lexiques gestuels.

---

## Slide 9 — Matrice d'Avancement Réel (Actual vs Pending)
### Une feuille de route rigoureuse

| Composant | Statut Réel Actuel (Prototype) | Étape en Attente (Feuille de Route) |
|---|---|---|
| **API & Contrats** | Validé par tests d'intégration contre le backend réel | Déploiement et tests d'infrastructure |
| **Gating de Rendu** | Blocage HTTP 403 actif sur toute tentative de publication | Déverrouillage conditionné par signature qualifiée |
| **Analyse Audio** | DSP réel sur CPU (F0 estimé, RMS, pauses) | Expérimentation d'un modèle d'émotion local |
| **Mouvements 3D** | 6 fixtures techniques synthétiques (non validées LSF) | Intégration de mouvements mocap autorisés |
| **Validation Humaine** | Protocole d'évaluation méthodologique rédigé | Sessions d'évaluation avec des locuteurs sourds natifs |
| **Vidéo Démo & Dépôt** | Walkthrough local de 75 s avec voix synthétique ; code sous Git local | Relecture de la vidéo et mise en ligne publique |

---

## Slide 10 — Conclusion & Prochaines Étapes
### Vers une technologie inclusive et humble

- **Notre apport clé :** Une architecture vérifiable qui place les contraintes linguistiques de la langue des signes au-dessus des automatismes algorithmiques.
- **Prochaines étapes :** Engager des sessions d'évaluation perceptuelle auprès de signants compétents en LSF et affiner la coarticulation cinématique des doigts et du visage.

> *« Rendre la culture accessible commence par le respect rigoureux de la langue et de ses locuteurs. »*
