# ADR-003 — PyMuPDF par défaut, Docling en option

- **Contexte** : Docling extrait mieux les tableaux et la mise en page, mais charge plusieurs modèles (lourd pour 8 Go de RAM).
- **Décision** : PyMuPDF par défaut (texte + taille de police pour détecter les titres, tableaux simples) ; Docling activable par `parser: docling`.
- **Conséquences** : ingestion rapide et légère ; qualité moindre sur les tableaux complexes. À comparer sur un échantillon avec le banc d'éval.
