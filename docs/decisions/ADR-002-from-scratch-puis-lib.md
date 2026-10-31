# ADR-002 — From scratch d'abord, bibliothèque ensuite

- **Contexte** : le but est de comprendre en profondeur, pas seulement d'assembler.
- **Décision** : chaque mécanisme central (BM25, cosinus, RRF, boucle d'agent, métriques, tracing, machine à états multi-agents) est codé à la main ; la version bibliothèque (Qdrant, cross-encoder, LangGraph, Ragas, Phoenix) est ajoutée à côté pour comparer.
- **Conséquences** : plus de code à maintenir, mais chaque concept est explicable ligne par ligne.
