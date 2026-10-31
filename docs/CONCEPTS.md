# Carte des concepts : où chaque notion est implémentée

> Pour chaque concept : le fichier qui l'implémente, le test qui le vérifie, et la commande pour le voir tourner.
> « FS » = codé **from scratch** ; « Lib » = version avec une bibliothèque standard, pour comparer.

## 1. Fondations de l'IA (`src/transfia/foundations/`, test : `tests/test_foundations.py`)

| Concept | Fichier | Commande |
|---|---|---|
| Recherche dans un espace d'états : BFS, A*, heuristique admissible | `search_symbolic.py` | `transfia foundations --only search_symbolic` |
| Système expert, chaînage avant, explicabilité | `search_symbolic.py` | idem |
| Train/test, k-fold, précision/rappel/F1, MSE, R², standardisation | `ml_scratch.py` | `transfia foundations --only ml_scratch` |
| Régression linéaire par descente de gradient vs équation normale | `ml_scratch.py` | idem |
| Régression logistique, entropie croisée, régularisation L2 | `ml_scratch.py` | idem |
| kNN, Naive Bayes (log-probabilités, lissage de Laplace) | `ml_scratch.py` | idem |
| Arbre de décision (Gini), KMeans (k-means++), PCA (SVD) | `ml_scratch.py` | idem |
| Perceptron multicouche, ReLU, softmax, rétropropagation | `neural_net.py` | `transfia foundations --only neural_net` |
| SGD vs Adam, mini-batchs, initialisation de He, vérification du gradient | `neural_net.py` | idem |
| Tokenisation BPE | `transformer.py` | `transfia foundations --only transformer` |
| Encodage de position, attention (Q, K, V, √d_k), masque causal, multi-head | `transformer.py` | idem |
| Bloc transformer (LayerNorm, résiduel), weight tying | `transformer.py` | idem |
| Échantillonnage : glouton, température, top-k, top-p ; perplexité | `transformer.py` | idem |
| Embeddings appris : word2vec skip-gram, échantillonnage négatif | `word2vec.py` | `transfia foundations --only word2vec` |
| RL : bandit ε-greedy, Q-learning ; alignement : perte DPO (lien RLHF) | `rl.py` | `transfia foundations --only rl` |
| LoRA (B·A, rang, fusion), quantification 8/4 bits, par groupes | `lora_quant.py` | `transfia foundations --only lora_quant` |

## 2. Données et ingestion (F1 à F3)

| Concept | Fichier | Test | Commande |
|---|---|---|---|
| API OpenAlex, pagination par curseur, licences article par article | `ingestion/collect.py` | `test_ingestion.py` | `transfia collect` |
| OAI-PMH / Dublin Core (alternative) | `ingestion/collect.py` | `test_parse_oai_dublin_core` | — |
| Catalogue SQLite, idempotence, ingestion incrémentale, manifeste | `ingestion/collect.py`, `ingestion/pipeline.py` | `test_pipeline_is_incremental` | `transfia ingest` |
| Parsing PDF structurel (titres par taille de police) FS / Docling Lib | `ingestion/parse.py` | `test_pdf_parsing_end_to_end` | `transfia ingest` |
| OCR (Tesseract), VLM pour décrire les figures | `ingestion/parse.py` | — | config `ocr`, `describe_figures` |
| Chunking : taille fixe + overlap, par phrases, structurel, sémantique | `ingestion/chunking.py` | `test_chunking_strategies` | config `chunking` |
| NER : dictionnaire FS / GLiNER zero-shot / LLM structuré | `ingestion/ner.py` | `test_gazetteer_ner_longest_match_and_plurals` | config `ner` |
| Étiquetage faible (weak supervision) | `ingestion/themes.py` | `test_weak_label` | — |

## 3. Machine learning appliqué (F3)

| Concept | Fichier | Commande |
|---|---|---|
| TF-IDF + régression logistique, Pipeline (pas de data leakage) | `ml/classic.py` | `transfia classify` |
| Validation croisée stratifiée, F1 macro, matrice de confusion, classes déséquilibrées | `ml/classic.py` | idem |
| Embeddings + classifieur ; LLM zero-shot ; interprétabilité (coefficients) | `ml/classic.py` | `transfia classify --llm` |
| Réduction de dimension (PCA, UMAP), KMeans + silhouette, HDBSCAN, c-TF-IDF | `ml/clustering.py` | `transfia themes` |

## 4. Recherche d'information (F4, `tests/test_retrieval.py`)

| Concept | Fichier |
|---|---|
| Embeddings : hashing trick FS, sentence-transformers (préfixes e5), Ollama | `embeddings/embedders.py` |
| Similarité cosinus FS, top-k par `argpartition` | `retrieval/similarity.py` |
| BM25 FS (IDF, saturation k1, normalisation b), persistance JSON (pas de pickle) | `retrieval/bm25.py` |
| Base vectorielle exacte FS / Qdrant local (HNSW, ANN), filtres de métadonnées | `retrieval/vector_store.py` |
| Fusion RRF FS, fusion pondérée min-max | `retrieval/hybrid.py` |
| Reranking : lexical FS / cross-encoder Lib (bi- vs cross-encoder) | `retrieval/rerank.py` |
| Moteur 2 étages (rappel puis précision), modes comparables | `retrieval/search.py` |

## 5. LLM et RAG (F5, `tests/test_rag.py`, `tests/test_llm.py`)

| Concept | Fichier |
|---|---|
| Client Ollama écrit à la main (chat, tools, JSON schema, streaming NDJSON, usage) | `llm/ollama.py` |
| Adaptateur cloud (LiteLLM), architecture hexagonale (ports/adaptateurs) | `llm/litellm_client.py`, `core/ports.py` |
| Sorties structurées : JSON schema, décodage contraint, validation, retry avec l'erreur | `llm/base.py` |
| Cache exact (SQLite) et cache sémantique | `llm/cache.py` |
| Routage a priori et cascade a posteriori | `llm/router.py` |
| Query rewriting, multi-query, HyDE | `rag/query.py` |
| Prompt RAG, lost in the middle, citations, droit de ne pas savoir, streaming | `rag/pipeline.py` |
| Vérification phrase par phrase : lexicale FS, NLI, LLM ; chiffres inventés | `rag/verify.py` |
| Prompts versionnés, `StrictUndefined`, tests de non-régression | `core/prompts.py`, `tests/test_prompts.py` |

## 6. Connaissances, génération, agents (F6 à F9, `tests/test_agents.py`)

| Concept | Fichier |
|---|---|
| Graphe de co-occurrence FS, triplets par LLM, entity resolution | `graph/kg.py` |
| Communautés (Louvain), recherche locale, recherche globale map-reduce (GraphRAG) | `graph/kg.py` |
| Schémas Pydantic + validateurs métier, lisibilité (Flesch–Kandel–Moles) | `generation/schemas.py`, `generation/readability.py` |
| Few-shot, templates, export Markdown avec mention IA | `generation/generator.py` |
| Skills : SKILL.md, progressive disclosure (3 niveaux), sécurité des chemins | `skills_runtime/registry.py`, `skills/` |
| Tools : schéma généré depuis les types, validation, timeout, erreurs renvoyées | `agents/tools.py` |
| Agent ReAct FS : tool calling natif et ReAct textuel, garde-fous de boucle | `agents/react.py` |
| Plan-and-execute, human-in-the-loop | `agents/planner.py` |
| Mémoire court terme (fenêtre + résumé) et long terme | `agents/memory.py` |
| Multi-agents : machine à états, reflection/critique, juge d'une autre famille | `agents/multi_agent.py` |
| Les mêmes agents avec LangGraph (checkpoints, interruption) | `agents/langgraph_impl.py` |
| MCP : serveur (tools, resources, prompts) et client (pont vers l'agent) | `mcp_server/server.py`, `agents/mcp_client.py` |

## 7. Qualité, sécurité, sobriété (F11, F12, F14, `tests/test_eval_ml_graph.py`)

| Concept | Fichier |
|---|---|
| Recall@k, Precision@k, MRR, nDCG, token F1, pertinence | `evaluation/metrics.py` |
| Intervalle de Wilson, bootstrap, test de permutation apparié | `evaluation/metrics.py` |
| Kappa de Cohen, corrélation de Spearman | `evaluation/metrics.py` |
| Jeux de test : humain, synthétique (filtré), pseudo-requêtes ; dev/test | `evaluation/dataset.py` |
| LLM-as-judge (raisonnement avant la note), jury, biais de position | `evaluation/judge.py` |
| Banc d'éval multi-configurations, empreinte de config, versions des prompts | `evaluation/runner.py` |
| Annotation humaine (CSV), calibration juge ↔ humain | `evaluation/human.py` |
| Injection directe et indirecte, hors périmètre (classifieur), PII, transparence | `guardrails/guard.py` |
| Tracing (spans, arbre), pont OpenTelemetry (Phoenix) | `observability/tracing.py` |
| Énergie et carbone (estimation, CodeCarbon), percentiles de latence | `observability/metrics.py` |
| Dimensionnement : poids, KV cache, tokens/s, index, coût API | `sobriety/sizing.py` |
| Front de Pareto, configuration la moins coûteuse qui atteint la qualité | `sobriety/pareto.py` |
| Fine-tuning SFT + LoRA/QLoRA, distillation, export GGUF | `finetuning/` |

## 8. Ingénierie et produit (F13)

| Concept | Fichier |
|---|---|
| Configuration centralisée, empreinte, `.env` | `config.py`, `config/*.yaml` |
| Injection de dépendances, chargement paresseux (RAM limitée) | `container.py` |
| API REST, SSE, OpenAPI, feedback utilisateur (éval online) | `api/app.py` |
| Front React : streaming par `fetch`, layout de graphe "force-directed" FS | `frontend/src/` |
| Ligne de commande par brique | `cli.py` |
| Tests sans réseau (LLM simulé, HTTP simulé), saut des dépendances optionnelles | `tests/` |

## Ce qui n'est pas couvert (honnêtement)

- **Génération d'images** (diffusion, GAN), **audio** (reconnaissance et synthèse vocale), **vidéo**.
- **Vision par ordinateur entraînée** (réseaux convolutifs) : seule l'utilisation d'un VLM est prévue.
- **Séries temporelles**, **systèmes de recommandation**, **apprentissage fédéré**.
- **Entraînement distribué** et vrai **RLHF** (seule la perte DPO est implémentée, pour comprendre).
- **Pré-entraînement** d'un vrai LLM (le mini-transformer montre les calculs, avec des poids aléatoires).
