# TransfIA

**From livestock research to sourced teaching content — an open-source, measured, CPU-only AI system.**

TransfIA turns scientific articles from the INRAE journal *Productions Animales* (CC BY 4.0) into **cited answers** and **teaching material** (quizzes, fact sheets, glossaries, case studies) for agricultural education. It implements, end to end, the key concepts of today's AI — classic ML, retrieval, RAG, structured generation, agents, MCP, multi-agent workflows, fine-tuning, evaluation, guardrails, frugality — and **measures every choice**.

> Author: **Corinne Bergès** — AI solution architect. Designed and measured by me, coded with AI assistance.
> 🇫🇷 Documentation is mostly in French: [README.fr.md](README.fr.md), [docs/](docs/). The UI is in French.

---

## Why it is interesting

- **Runs on a laptop with no GPU and 8 GB of RAM** (Ollama, Qwen 2.5 1.5B, quantized), fully offline.
- **Same code on Google Cloud**: switch to Gemini with one config file (`config/gcp.yaml`, via LiteLLM). Hexagonal architecture (ports & adapters).
- **The guardrails are the product**: input filtering, subject anchoring (refuse when the corpus has no source), sentence-level verification of every answer, blocking of unsupported answers, citations built by code.
- **Everything is benchmarked** with the same command (`transfia bench`) across models.

## Results (same benchmark, three models)

| Metric | Qwen 2.5 1.5B (local) | Qwen 2.5 3B (local) | Gemini 3.1 Flash-Lite (cloud) |
|---|---|---|---|
| Answer faithfulness | **1.00** | 0.94 | 0.96 |
| Hallucinations on out-of-corpus questions | **0** | **0** | **0** |
| Wrongful refusals | 0 | 0 | 0 |
| Agent tasks completed without fallback | 0 / 3 | **3 / 3** | — |
| Correct quiz questions kept | 3 / 9 | **7 / 9** | — |
| Median answer latency (same 10 questions) | 23.8 s | — | **8.6 s** |

- **Small samples** (10–17 questions including out-of-corpus ones): read them as trends, not statistical proof.
- **Qwen3 4B** did not fit: on 8 GB of RAM a single call took more than 15 minutes (memory spilled to disk).
- Full test report with every failure, cause and fix: [docs/RAPPORT_TESTS.md](docs/RAPPORT_TESTS.md) (French).

## Five lessons

1. **A bigger model is not a more faithful one.** Size bought capability (tool use, quizzes), not truthfulness.
2. **Valid is not correct.** A schema-perfect JSON quiz still marked the wrong answer → content checks in code.
3. **An instruction is not a control.** The small agent answered from memory despite the prompt → grounding guard + code fallback.
4. **Re-evaluate on every model change.** Gemini paraphrases more; the word-level verifier flags correct sentences.
5. **Hardware and licences decide.** The 3B model has a non-commercial licence, the 4B one does not fit in RAM.

## Features

| Feature | What the user sees | AI concepts |
|---|---|---|
| Collection & ingestion | a catalogue of reusable articles, structured documents | licences, parsing, chunking, cleaning |
| Search | the most relevant passages | BM25, embeddings, hybrid (RRF), cross-encoder reranking |
| Sourced Q&A | answers with verified citations | RAG, query cleaning, subject anchoring, verification, blocking |
| Knowledge graph | concept map | entities, communities, GraphRAG (local and global) |
| Generator | quizzes, fact sheets, case studies, glossaries by level | structured outputs, consistency checks, repair loop, skills |
| Assistant | an agent that searches, plans, writes | tool calling, ReAct, plan-and-execute, MCP server, memory, fallback |
| Editorial workshop | writer → reviewer → simplifier | multi-agent state machine, rule-based reviewer, final check |
| Specialised model | a fine-tuned 1.5B quiz model | QLoRA (1.18 % of parameters), distillation, GGUF, prompt contract |
| Evaluation | comparable numbers between configurations | metrics, refusal confusion matrix, LLM judges, human agreement, benchmark |
| Guardrails & traces | refusals, step tree | prompt injection, out-of-scope, PII, OpenTelemetry tracing |
| Frugality | quality vs cost, energy per request | sizing, routing, cache, energy estimate |

## Quick start (Windows, 8 GB of RAM)

```powershell
# Prerequisites: Python 3.12, Git, Ollama (https://ollama.com)
ollama pull qwen2.5:1.5b
git clone https://github.com/corinne3/transfia.git
cd transfia
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[recommended]"
transfia doctor
pytest                                   # ~150 tests, no network, no model needed

transfia ingest --samples                # small offline sample corpus
transfia ask "Pourquoi le colostrum est-il indispensable au veau ?"
transfia bench --label small             # reproduce the benchmark
```

Real corpus: `transfia collect --max 30` then `transfia ingest`.
Web UI: `transfia serve` and `cd frontend; npm install; npm run dev` → http://localhost:5173.
Google Cloud (Gemini, free tier, no credit card): put `GEMINI_API_KEY` in `.env`, then `transfia --config config/gcp.yaml ask "..."`.

## Data and licences

- **Code**: MIT.
- **Corpus**: *INRAE Productions Animales* articles, **CC BY 4.0**, downloaded by the collection script and **never redistributed** here. Every generated content cites authors, title and DOI.
- **Models**: open-weights, each under its own licence (note: Qwen 2.5 **3B** is under a non-commercial research licence; 1.5B and Qwen3 4B are Apache 2.0). See `docs/MODEL_CARD.md`.

> AI-generated content from cited sources: always review before using it with learners.
