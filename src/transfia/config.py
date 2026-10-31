"""Configuration centralisée (11.6 du dossier : reproductibilité).

- Une seule source de vérité : un fichier YAML (config/local.yaml, config/gcp.yaml...).
- Changer de fournisseur (local, GCP, Azure) = changer de fichier, pas le code (chapitre 12).
- Chaque résultat d'évaluation enregistre la configuration utilisée (``Settings.fingerprint``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class LLMSettings(BaseModel):
    provider: Literal["ollama", "litellm", "fake"] = "ollama"
    model: str = "qwen2.5:1.5b"           # petit modèle par défaut (8 Go de RAM)
    big_model: str | None = "qwen2.5:3b"   # pour le routage en cascade (F14)
    routing: bool = False                  # utiliser big_model (routage + cascade) ? option, coûteuse en RAM
    # Modèle pour la GÉNÉRATION de contenus (F7). None = même modèle que les questions.
    # Routage PAR TÂCHE : une réponse doit être rapide (on attend), un quiz se prépare à l'avance
    # (on peut attendre 2 min de plus pour un meilleur modèle).
    generation_model: str | None = None
    vlm_model: str | None = "moondream"    # description des figures (F2)
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 4096                    # fenêtre de contexte : pas plus que nécessaire (KV cache)
    max_tokens: int = 700
    timeout_s: float = 300.0
    cache: bool = True


class EmbedderSettings(BaseModel):
    kind: Literal["hashing", "sentence-transformers", "ollama"] = "sentence-transformers"
    model: str = "intfloat/multilingual-e5-small"
    dim: int = 512                         # pour l'embedder "hashing" (from scratch)
    query_prefix: str = "query: "          # e5 attend ces préfixes !
    passage_prefix: str = "passage: "


class RetrievalSettings(BaseModel):
    vector_store: Literal["memory", "qdrant"] = "qdrant"
    k_dense: int = 20
    k_bm25: int = 20
    k_final: int = 5
    rrf_k: int = 60
    reranker: Literal["none", "lexical", "cross-encoder"] = "cross-encoder"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_top: int = 20
    rerank_max_length: int = 256          # tokens lus par le cross-encoder par paire (512 = 2x plus lent)
    min_score: float | None = None         # seuil de pertinence du 1er passage (à CALIBRER avec le banc d'éval)


class IngestionSettings(BaseModel):
    parser: Literal["pymupdf", "pypdf", "docling", "text"] = "pymupdf"
    chunking: Literal["fixed", "structural", "semantic"] = "structural"
    chunk_words: int = 220
    overlap_words: int = 40
    ner: Literal["gazetteer", "gliner", "llm"] = "gazetteer"
    describe_figures: bool = False         # VLM sur CPU = lent : désactivé par défaut
    ocr: bool = False


class RAGSettings(BaseModel):
    rewrite: bool = True
    rewrite_only_with_history: bool = True   # reformuler seulement les questions de suite de conversation
    multi_query: bool = False
    hyde: bool = False
    verify: Literal["none", "lexical", "nli", "llm"] = "lexical"
    reorder_lost_in_middle: bool = True
    max_tokens: int = 350                 # ~6 phrases : borne la réponse (temps, coût, dérive vers des listes)
    # Que faire d'une réponse dont AUCUNE phrase n'est soutenue par les sources (hallucination probable) ?
    on_unsupported: Literal["refuse", "warn"] = "refuse"
    # ANCRAGE SUR LE SUJET : mots de la question absents de TOUT le corpus (« grippe », « autruche »...).
    # Refus si le meilleur passage n'est pas franchement pertinent (score < anchor_max_score), et phrases
    # contenant ces mots déclarées non soutenues. Vu en vrai : 2 hallucinations hors corpus sur 8.
    anchor_unknown_terms: bool = True
    anchor_max_score: float = 0.0


class GuardSettings(BaseModel):
    enabled: bool = True
    min_citation_ratio: float = 0.5


class EvalSettings(BaseModel):
    judges: list[str] = Field(default_factory=lambda: ["qwen2.5:1.5b", "llama3.2:1b"])


class ObservabilitySettings(BaseModel):
    traces: bool = True
    otel: bool = False                     # True pour envoyer vers Phoenix / Cloud Trace
    otel_endpoint: str = "http://localhost:6006/v1/traces"
    energy: Literal["none", "estimate", "codecarbon"] = "estimate"
    cpu_watts: float = 25.0                # puissance moyenne estimée du CPU en charge
    grid_gco2_per_kwh: float = 50.0        # ordre de grandeur pour la France (très variable)


class Settings(BaseModel):
    name: str = "local"
    data_dir: str = "data"
    prompts_dir: str = "prompts"
    skills_dir: str = "skills"
    models_dir: str = "models"          # modèles Hugging Face copiés DANS le projet (transfia models download)
    contact_email: str | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    guardrails: GuardSettings = Field(default_factory=GuardSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    # --- chemins résolus -------------------------------------------------
    def path(self, *parts: str) -> Path:
        base = Path(self.data_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
        p = base.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    def model_path(self, name: str) -> str:
        """Si le modèle a été copié dans models/ (``transfia models download``), on renvoie ce dossier :
        la bibliothèque le charge alors depuis le disque, sans AUCUN appel à Hugging Face.
        Sinon, on renvoie le nom (téléchargement automatique dans le cache au premier usage)."""
        local = self.resolve(self.models_dir) / name.replace("/", "__")
        return str(local) if (local / "config.json").exists() else name

    @property
    def fingerprint(self) -> str:
        """Empreinte courte de la configuration : à stocker avec chaque résultat."""
        raw = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _load_dotenv(path: Path) -> None:
    """Mini-lecteur de .env (évite une dépendance) : KEY=VALUE, sans écraser l'existant."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip())


def load_settings(path: str | Path | None = None, **overrides) -> Settings:
    _load_dotenv(REPO_ROOT / ".env")
    path = path or os.environ.get("TRANSFIA_CONFIG") or REPO_ROOT / "config" / "local.yaml"
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update(overrides)
    settings = Settings(**data)
    if os.environ.get("TRANSFIA_OLLAMA_URL"):         # ex. http://ollama:11434 dans Docker Compose
        settings.llm.base_url = os.environ["TRANSFIA_OLLAMA_URL"]
    if not settings.contact_email:
        settings.contact_email = os.environ.get("TRANSFIA_CONTACT_EMAIL") or None
    return settings
