# Livrables locaux — SignaCulture AI

- `SignaCulture_Presentation.pdf` : dix diapositives françaises, exportées depuis `docs/slides.md`.
- `SignaCulture_Demo_FR.mp4` : walkthrough de 75 secondes de l'application locale, avec narration française de synthèse (eSpeak NG) et indications à l'écran.
- `demo_narration.txt` : texte effectivement utilisé pour cette voix de synthèse.
- `browser/studio.png`, `browser/audio.png`, `browser/mobile.png` : captures du parcours navigateur automatisé.

La vidéo montre le logiciel et six mouvements synthétiques de test. Elle ne représente pas une traduction LSF approuvée ni une démonstration d'emotion2vec. Aucun corpus vidéo de signants n'a été téléchargé ou redistribué. La vidéo est une capture de présentation ; `/render` reste un endpoint d'aperçu sans export vidéo applicatif.

Vérifications réalisées : 54 tests Python (modèle spaCy français installé et réellement exécuté), lint frontend, build de production, 7 smoke tests HTTP contre FastAPI réel, parcours Playwright desktop/mobile et gestion des erreurs réseau. Le contrôle des worktrees et les limites restantes sont documentés dans `ORCHESTRATION.md`.
