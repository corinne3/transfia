# Parcours du code : quelle commande lance quels fichiers

> Pour chaque commande : les fichiers `.py` appelés, **dans l'ordre**, et ce que chacun fait.
> Tous les chemins partent de `src/transfia/`.

## Le point d'entrée commun à toutes les commandes

```
transfia <commande>
  └─ cli.py                 main() lit la commande et appelle la fonction cmd_<commande>()
       └─ config.py         load_settings() : lit config/local.yaml (+ .env)
       └─ container.py      AppContext : la "boîte à outils" qui fabrique chaque brique
                            (LLM, embedder, base vectorielle, moteur de recherche...)
                            SEULEMENT quand on en a besoin (chargement paresseux)
```

---

## Où sont les données ?

| Quoi | Où (sur ton PC) | Produit par |
|---|---|---|
| Corpus d'**exemple** (8 textes courts) | `data\samples\sample_corpus.jsonl` | Fourni avec le projet |
| Catalogue des **vrais** articles | `data\catalog.sqlite` | `transfia collect` |
| PDF des vrais articles | `data\raw\*.pdf` | `transfia collect` |
| Index vectoriel (les embeddings) | `data\index\qdrant\` | `transfia ingest` |
| Index lexical (BM25) | `data\index\bm25.json` | `transfia ingest` |
| Liste des articles déjà indexés | `data\index\manifest.json` | `transfia ingest` |
| Modèles d'embedding et de reranking | `models\` | `transfia models download` |
| Traces (étapes de chaque requête) | `data\traces\traces.jsonl` | Toutes les commandes |
| Rapports d'évaluation | `data\eval\` | `transfia eval ...` |

**La recherche se fait toujours sur l'INDEX** (`data\index\`), jamais directement sur les PDF.
L'index est construit une fois par `ingest`, puis interrogé autant de fois qu'on veut.

---

## 1. `transfia ingest --samples` — indexer le corpus d'exemple (hors ligne)

```
cli.py                       cmd_ingest()
└─ container.py              index_samples()
   ├─ ingestion/collect.py   load_sample_articles()   lit data/samples/sample_corpus.jsonl
   ├─ ingestion/parse.py     TextParser               transforme chaque texte en ParsedDocument (sections)
   └─ ingestion/pipeline.py  Indexer.run()            pour CHAQUE article :
      ├─ ingestion/chunking.py   chunk_document()     découpe en morceaux (~220 mots), garde titre/auteurs/page
      ├─ ingestion/ner.py        GazetteerNER          repère espèces, pathologies, nutriments…
      ├─ ingestion/themes.py     weak_label()          devine le thème (nutrition, santé…)
      ├─ embeddings/embedders.py embed_documents()     transforme chaque chunk en vecteur (modèle e5)
      ├─ retrieval/vector_store.py upsert()            range les vecteurs dans Qdrant (data/index/qdrant)
      └─ retrieval/bm25.py       BM25.fit().save()     construit l'index lexical (data/index/bm25.json)
```

## 2. `transfia collect --max 20` — télécharger les VRAIS articles (Internet)

```
cli.py                       cmd_collect()
└─ ingestion/collect.py      collect()
   ├─ iter_openalex()        interroge l'API OpenAlex (catalogue mondial de publications)
   ├─ article_from_openalex() convertit chaque réponse en Article (titre, auteurs, DOI, licence, résumé)
   ├─ is_reusable()          garde seulement les licences CC BY
   ├─ Catalog.upsert()       enregistre dans data/catalog.sqlite
   └─ download_pdf()         télécharge le PDF dans data/raw/ (s'il est disponible)
```

## 3. `transfia ingest` — indexer les vrais articles

Même chaîne que l'étape 1, sauf le début :
```
cli.py                       cmd_ingest()
├─ ingestion/collect.py      Catalog.all()            lit data/catalog.sqlite
└─ ingestion/pipeline.py     Indexer.run()
   └─ ingestion/parse.py     PyMuPDFParser.parse()    lit chaque PDF, détecte les titres (taille de police), les tableaux
   └─ … puis chunking → NER → thèmes → embeddings → Qdrant → BM25 (comme l'étape 1)
```
Un article sans PDF est indexé sur son **résumé** seulement.

## 4. `transfia search "..."` — chercher des passages

```
cli.py                       cmd_search()
└─ retrieval/search.py       SearchEngine.search()
   ├─ dense()                (modes dense, hybrid)
   │  ├─ embeddings/embedders.py  embed_query()       la question → un vecteur
   │  └─ retrieval/vector_store.py search()           les vecteurs les plus proches dans Qdrant
   ├─ lexical()              (modes bm25, hybrid)
   │  └─ retrieval/bm25.py        search()            les passages qui partagent les mots rares
   ├─ retrieval/hybrid.py    rrf()                    fusionne les deux classements (mode hybrid)
   └─ retrieval/rerank.py    CrossEncoderReranker     relit question + passage ensemble, reclasse
                             (sauf avec --no-rerank)
```

## 5. `transfia ask "..."` — répondre avec des sources (RAG)

```
cli.py                       cmd_ask()
└─ rag/pipeline.py           RAGPipeline.answer()
   ├─ guardrails/guard.py    check_input()            bloque injection et hors-sujet
   ├─ rag/query.py           rewrite_query()          le LLM reformule la question
   ├─ retrieval/search.py    search()                 l'étape 4 complète
   ├─ rag/pipeline.py        build_messages()         prompt = consignes + passages numérotés [1]…[5]
   │  └─ core/prompts.py     render()                 lit prompts/rag_system.j2 et rag_user.j2
   ├─ llm/ollama.py          chat()                   envoie le prompt au LLM local (Ollama)
   ├─ rag/verify.py          verify_answer()          chaque phrase est-elle soutenue par son passage ?
   └─ guardrails/guard.py    check_output()           masque les données perso, ajoute la mention IA
```
Avec `--trace`, `observability/tracing.py` affiche l'arbre de ces étapes et leur durée.

## 6. `transfia generate quiz "..."` — créer un contenu pédagogique

```
cli.py                          cmd_generate()
└─ generation/generator.py      ContentGenerator.generate()
   ├─ retrieval/search.py       search()              passages sur le thème
   ├─ skills_runtime/registry.py load("quiz-qcm")     lit skills/quiz-qcm/SKILL.md (le savoir-faire)
   ├─ core/prompts.py           render("generate")    prompt = niveau + skill + exemple + passages
   ├─ llm/base.py               chat_structured()     le LLM doit renvoyer un JSON conforme au schéma
   │  └─ generation/schemas.py  Quiz                  le schéma (4 choix, 1 bonne réponse, une source…)
   └─ generation/readability.py flesch_kandel_moles() le texte est-il lisible pour ce niveau ?
```

## 7. `transfia agent "..."` — l'assistant qui décide seul de ses étapes

```
cli.py                        cmd_agent()
└─ agents/react.py            ReActAgent.stream()     BOUCLE :
   ├─ llm/ollama.py           chat(tools=…)            le LLM répond par du texte OU demande un outil
   ├─ agents/tools.py         ToolRegistry.call()      exécute l'outil demandé, renvoie le résultat
   │  └─ agents/corpus_tools.py search_corpus, read_article, load_skill, generate_content…
   └─ … recommence jusqu'à la réponse finale (8 étapes maximum)
```
Avec `--mcp`, les outils viennent de `agents/mcp_client.py`, qui lance `mcp_server/server.py` dans un autre processus.

## 8. `transfia eval rag` — mesurer la qualité

```
cli.py                        cmd_eval()
├─ evaluation/dataset.py      load_dataset()          les questions de test (data/samples/golden_sample.jsonl)
└─ evaluation/runner.py       eval_rag()              pour chaque question : l'étape 5 complète
   ├─ evaluation/metrics.py   token_f1, intervalles de confiance…
   ├─ evaluation/judge.py     (option --judges)       d'autres LLM notent les réponses
   └─ observability/metrics.py EnergyMeter            temps et énergie consommés
```
