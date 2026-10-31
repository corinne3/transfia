# ADR-004 — L'évaluation avant l'optimisation

- **Contexte** : sans mesure, impossible de savoir si un changement (chunking, modèle, prompt) améliore ou dégrade.
- **Décision** : banc d'éval dès le départ ; chaque rapport enregistre l'empreinte de configuration et les versions des prompts ; les comparaisons passent par un test statistique ; un échantillon est validé par des humains.
- **Conséquences** : un peu plus lent au début, mais chaque choix technique est justifié par des chiffres.
