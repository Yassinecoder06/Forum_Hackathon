# Script de Narration — Démonstration SignaCulture AI (1–2 minutes)

> **Format :** Vidéo de pitch et démonstration technique (durée cible : 90 secondes / 1 minute 30)  
> **Langue :** Français  
> **Avertissement Formel :** Ce document constitue le script officiel préparé pour l'enregistrement de la démonstration du Hackathon IA Éthique 360. Aucune vidéo pré-enregistrée ou publiée n'est prétendue comme existante avant l'étape de captation finale.

---

## Scénario Minuté de Démonstration

### [00:00 - 00:18] Introduction : Le Problème Réel et Notre Posture
- **Visuel à l'écran :** Écran d'accueil de SignaCulture AI avec la mention éthique bien visible : *« Prototype expérimental — Respect de la LSF et validation humaine requise »*.
- **Voix off (Français) :**  
  *« Rendre la culture accessible aux personnes sourdes est un impératif. Mais en matière de langue des signes française, automatiser sans discernement revient souvent à trahir la langue. La LSF n'est pas un calque gestuel du français : c'est une langue visuelle et spatiale riche. Avec SignaCulture AI, nous avons choisi une voie responsable : assister la création d'animations 3D en LSF à partir de mouvements réels, enrichis par l'intonation de la voix, sous le contrôle absolu de la communauté sourde. »*

---

### [00:18 - 00:42] Démonstration 1 : Du Texte au Mouvement Contrôlé
- **Visuel à l'écran :** L'utilisateur saisit la phrase « Bonjour, bienvenue à la découverte de notre culture ». L'interface décompose les tokens, extrait les concepts autorisés (`bonjour`, `culture`), affiche les glosses associés et signale clairement les termes hors lexique avec le statut `translation_validated: false`. L'avatar 3D articulé sur le rig `signaculture-v1` joue la séquence prévisualisée.
- **Voix off (Français) :**  
  *« Voyez notre pipeline en action : chaque mot saisi est analysé. Si une expression appartient à notre bibliothèque contrôlée, son mouvement cinématique est chargé. Si un terme est inconnu ou ambigu, le système refuse formellement d'inventer : il l'isole et exige une revue linguistique. Aucun faux sentiment de traduction parfaite n'est communiqué à l'utilisateur. »*

---

### [00:42 - 01:05] Démonstration 2 : Analyse Acoustique Réelle et Prosodie
- **Visuel à l'écran :** Chargement d'un signal audio WAV. Le module de traitement du signal décompose en temps réel la courbe de hauteur (F0), l'énergie sonore et les pauses respiratoires avec un horodatage précis en millisecondes.
- **Voix off (Français) :**  
  *« Nous intégrons également la voix, non pour remplacer les règles de la LSF, mais pour capturer son rythme. Notre traitement du signal déterministe extrait l'énergie et la fréquence fondamentale pour synchroniser les pauses et l'intensité du geste. L'hypothèse émotionnelle reste strictement encadrée : jamais une intonation vocale ne viendra altérer les expressions faciales grammaticales d'un signe. »*

---

### [01:05 - 01:30] Démonstration 3 : Gating Éthique et Revue Humaine
- **Visuel à l'écran :** Clic sur le bouton « Publier la vidéo ». Une alerte rouge s'affiche : *« Publication bloquée (HTTP 403) : validations linguistique et juridique en attente »*. L'utilisateur accède au panneau de revue où un expert peut annoter et valider chaque segment gestuel.
- **Voix off (Français) :**  
  *« C'est ici que réside notre engagement éthique le plus fort : notre système intègre un verrouillage logiciel inviolable. Même après un aperçu technique, l'exportation finale pour publication est strictement interdite tant qu'un relecteur sourd n'a pas validé les mouvements et que les droits d'auteur ne sont pas certifiés. SignaCulture AI, c'est l'intelligence artificielle au service de l'inclusion, avec humilité et rigueur. »*
