---
name: quiz-qcm
description: Créer un QCM pédagogique sourcé à partir de passages du corpus. À utiliser quand l'utilisateur demande un quiz, un QCM ou des questions d'entraînement.
---
# Créer un QCM

1. Une question = UNE notion clé des passages. Pas de question sur un détail anecdotique.
2. Exactement 4 propositions, une seule bonne réponse.
3. Les distracteurs (mauvaises réponses) sont PLAUSIBLES : erreurs fréquentes, confusions classiques, notions voisines. Jamais absurdes ni drôles.
4. Évite les formulations négatives (« laquelle n'est PAS… ») et « toutes les réponses ».
5. Varie la position de la bonne réponse.
6. Explication courte (1 à 2 phrases) qui dit POURQUOI, avec le numéro du passage dans `source`.
   Écris l'explication AVANT de fixer `bonne_reponse` : l'index doit désigner la proposition qu'elle justifie
   (0 = A, 1 = B, 2 = C, 3 = D). Relis : si l'explication valide deux propositions, la question est à refaire.
7. Adapte le vocabulaire au niveau (voir grille-qualite.md).
