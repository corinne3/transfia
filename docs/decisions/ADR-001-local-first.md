# ADR-001 — Local d'abord, cloud en option

- **Contexte** : PC sans GPU, 8 Go de RAM, pas de crédits API ; objectif d'apprentissage et de publication open source.
- **Décision** : tout tourne en local (Ollama + modèles quantifiés + Qdrant local). Le cloud passe par un adaptateur (LiteLLM) et un fichier de configuration.
- **Conséquences** : coût nul, souveraineté, reproductibilité ; en contrepartie, petits modèles (qualité limitée, lenteur). La couche d'abstraction permet de mesurer l'écart avec un modèle cloud.
