# TransfIA (version française)

> English version: [README.md](README.md)

**De la recherche en élevage aux contenus pédagogiques sourcés.**
Un projet open source pour implémenter, et comprendre, **tous les concepts clés de l'IA actuelle** : du machine learning classique aux agents, en passant par le RAG, l'évaluation et la frugalité. Tout tourne **sur un PC sans GPU**.

> Auteure : Corinne Bergès. Licence du code : MIT.

---

## Ce que fait l'application

| Fonctionnalité | Ce que voit l'utilisateur | Concepts IA |
|---|---|---|
| F1 Collecte | Un catalogue d'articles réutilisables (CC BY) | Data engineering, licences, idempotence |
| F2 Ingestion | Des PDF transformés en documents structurés | Parsing, layout, OCR, VLM |
| F3 Enrichissement | Des filtres par espèce et thème, une carte des thèmes | Chunking, NER, ML classique, clustering |
| F4 Recherche | Les passages les plus pertinents | Embeddings, BM25, hybride, RRF, reranking, ANN |
| F5 Questions sourcées | Des réponses avec citations vérifiées | RAG, rewriting, multi-query, HyDE, vérification |
| F6 Graphe | La carte des liens entre concepts | Knowledge graph, communautés, GraphRAG |
| F7 Générateur | QCM, fiches, cas pratiques, glossaires par niveau | Sorties structurées, few-shot, skills |
| F8 Assistant | Un agent qui cherche, planifie, rédige | Tool calling, ReAct, plan-and-execute, MCP, mémoire |
| F9 Atelier éditorial | Rédacteur → relecteur → vulgarisateur | Multi-agents, reflection |
| F10 Modèle spécialisé | Un petit modèle fine-tuné | SFT, LoRA, QLoRA, quantification, distillation |
| F11 Évaluation | Des chiffres pour comparer les configurations | Métriques, LLM-as-judge, statistiques, accord humain |
| F12 Garde-fous et traces | Refus des injections, arbre des étapes | Guardrails, prompt injection, tracing |
| F13 Interface web | Chat en streaming, recherche, agent en direct | FastAPI, SSE, React |
| F14 Sobriété | Qualité vs coût, front de Pareto | Dimensionnement, énergie, routage, cache |
| Fondations | Démos « from scratch » | IA symbolique, ML, réseaux de neurones, transformer, RL, LoRA, word2vec |

Le détail de chaque concept, fichier par fichier : **[docs/CONCEPTS.md](docs/CONCEPTS.md)**.
Le dossier fonctionnel complet : **[docs/dossier-fonctionnel-transfia.md](docs/dossier-fonctionnel-transfia.md)**.

---

## Démarrage rapide (Windows, 8 Go de RAM)

Le guide pas à pas, brique par brique, en 4 h : **[docs/GUIDE_WEEKEND.md](docs/GUIDE_WEEKEND.md)**.

```powershell
# 1. Prérequis : Python 3.12, Git, Ollama (https://ollama.com)
ollama pull qwen2.5:1.5b

# 2. Installation
git clone https://github.com/corinne3/transfia.git
cd transfia
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[recommended]"

# 3. Vérification
transfia doctor
pytest

# 4. Premiers pas (corpus d'exemple, hors ligne)
transfia ingest --samples
transfia search "facteurs de risque des mammites"
transfia ask "Pourquoi le colostrum est-il indispensable au veau ?" --trace
```

Le vrai corpus (revue INRAE Productions Animales, CC BY 4.0) :

```powershell
transfia collect --max 30
transfia ingest
```

Interface web :

```powershell
transfia serve                     # API sur http://127.0.0.1:8000 (doc : /docs)
cd frontend; npm install; npm run dev   # interface sur http://localhost:5173
```

---

## Principes du projet

- **From scratch d'abord, la bibliothèque ensuite.** BM25, cosinus, RRF, boucle d'agent, métriques, tracing : codés à la main, puis comparés aux outils standard.
- **Frugal par défaut.** Petit modèle quantifié, CPU, cache, routage. Chaque choix est mesuré (F14).
- **Sourcé et vérifiable.** Chaque contenu généré cite ses sources et signale ce qui n'est pas vérifié.
- **Portable.** Architecture hexagonale (ports et adaptateurs) : local, GCP ou Azure en changeant un fichier de configuration (`config/*.yaml`).
- **Testé.** Près de 150 tests automatisés, qui tournent sans réseau et sans modèle (LLM simulé).

## Configurations

| Fichier | Usage |
|---|---|
| `config/local.yaml` | Par défaut : Ollama + e5 + Qdrant local + cross-encoder |
| `config/light.yaml` | Ultra-léger : aucun modèle à part le LLM (embeddings et reranker from scratch) |
| `config/test.yaml` | Tests automatisés : LLM factice, aucun réseau |
| `config/gcp.yaml`, `config/azure.yaml` | Cloud via LiteLLM (chapitres 12 à 14 du dossier) |

Utilisation : `transfia --config config/light.yaml ask "..."` (l'option `--config` se place **avant** la commande).

## Structure

```
src/transfia/
  core/            modèles de données, ports (interfaces), texte, prompts versionnés
  llm/             clients Ollama / LiteLLM / factice, sorties structurées, cache, routeur
  ingestion/       collecte, parsing, chunking, NER, thèmes, pipeline
  embeddings/      hashing (from scratch), sentence-transformers, Ollama
  retrieval/       cosinus, BM25, bases vectorielles, RRF, reranking, moteur
  rag/             transformations de requête, pipeline, vérification
  graph/           graphe de connaissances, communautés, GraphRAG
  generation/      schémas pédagogiques, générateur, lisibilité
  skills_runtime/  chargement progressif des skills
  agents/          tools, ReAct, plan-and-execute, multi-agents, mémoire, client MCP, LangGraph
  mcp_server/      serveur MCP
  guardrails/      injection, hors périmètre, données personnelles, transparence
  evaluation/      jeux de test, métriques, juges, banc d'éval, annotation humaine
  sobriety/        dimensionnement, front de Pareto
  ml/              classification et clustering
  foundations/     toute l'IA « from scratch » en NumPy
  api/             FastAPI + streaming SSE
prompts/           templates Jinja2 (versionnés et testés)
skills/            une skill = un dossier avec SKILL.md
frontend/          React + Vite + TypeScript
finetuning/        notebook Colab LoRA, jeu de données, Modelfile Ollama
data/samples/      corpus d'exemple et jeu de test (les vraies données ne sont jamais commitées)
docs/              guide, carte des concepts, model card, datasheet, décisions d'architecture
```

## Données et licences

- **Code** : MIT.
- **Corpus** : articles de la revue *INRAE Productions Animales*, licence **CC BY 4.0**, téléchargés par le script de collecte et **jamais redistribués** dans ce dépôt. Chaque contenu généré cite les auteurs, le titre et le DOI.
- **Corpus d'exemple** (`data/samples`) : textes courts rédigés pour le projet (CC0), clairement marqués « [Exemple] ».
- **Modèles** : open-weights, téléchargés depuis Ollama ou Hugging Face, chacun sous sa propre licence (voir `docs/MODEL_CARD.md`).

> Contenus générés par IA à partir des sources citées : à vérifier avant toute diffusion à des apprenants.
