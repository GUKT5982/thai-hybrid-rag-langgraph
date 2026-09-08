"""Configuration for the Hybrid Thai RAG pipeline.

Every tunable knob (k values, thresholds, model names, DB/data locations)
lives here as a dataclass field, loaded from environment variables (see
.env.example). Nothing downstream (indexing.py, graph.py, nodes/, eval/)
hardcodes these values — they all take an AppConfig instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; real env vars still work without it


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class DataConfig:
    """Where the two HF datasets live and how much of the corpus to use.

    corpus_repo/corpus_lang -> miracl/miracl-corpus, the thing we INDEX.
    nomiracl_repo/nomiracl_lang -> miracl/nomiracl, the thing we EVALUATE with.
    See README.md "Two datasets, two jobs" for why these must not be confused.
    """

    corpus_repo: str = field(default_factory=lambda: _env("CORPUS_REPO", "miracl/miracl-corpus"))
    corpus_lang: str = field(default_factory=lambda: _env("CORPUS_LANG", "th"))
    nomiracl_repo: str = field(default_factory=lambda: _env("NOMIRACL_REPO", "miracl/nomiracl"))
    nomiracl_lang: str = field(default_factory=lambda: _env("NOMIRACL_LANG", "thai"))

    # "sample" -> first `sample_size` corpus passages, fast dev loop.
    # "full"   -> all 542,166 passages, real evaluation.
    data_mode: str = field(default_factory=lambda: _env("DATA_MODE", "sample"))
    sample_size: int = field(default_factory=lambda: _env_int("DATA_SAMPLE_SIZE", 5000))

    cache_dir: Path = field(default_factory=lambda: Path(_env("DATA_CACHE_DIR", "./data_cache")))

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def nomiracl_local_dir(self) -> Path:
        """Local snapshot dir for the (Thai-filtered) nomiracl repo.

        MUST stay filtered with allow_patterns: an unfiltered
        snapshot_download(repo_id=...) pulls all 166 files across every
        language in the repo, and on Windows without Developer Mode enabled
        that hard-crashes partway through (OSError: WinError 1314, missing
        symlink privilege) instead of degrading gracefully — confirmed by
        hitting it directly. Keep this pattern list identical to
        data.download.download_nomiracl's.
        """
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=self.nomiracl_repo,
            repo_type="dataset",
            allow_patterns=["*thai*", "*/th/*", "*-th-*", "*th.*"],
        )
        return Path(local_dir)


@dataclass
class DBConfig:
    """Postgres + pgvector connection and schema settings."""

    host: str = field(default_factory=lambda: _env("PGHOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("PGPORT", 5432))
    dbname: str = field(default_factory=lambda: _env("PGDATABASE", "thai_rag"))
    user: str = field(default_factory=lambda: _env("PGUSER", "postgres"))
    password: str = field(default_factory=lambda: _env("PGPASSWORD", ""))
    table_name: str = field(default_factory=lambda: _env("PGVECTOR_TABLE", "passages"))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))

    @property
    def dsn(self) -> str:
        return f"host={self.host} port={self.port} dbname={self.dbname} user={self.user} password={self.password}"


@dataclass
class RetrievalConfig:
    """Retrieval/fusion/rerank knobs. See README for what each one does to
    the pipeline's recall/precision tradeoff."""

    use_newmm: bool = field(default_factory=lambda: _env_bool("USE_NEWMM", True))
    bm25_k: int = field(default_factory=lambda: _env_int("BM25_K", 20))
    vector_k: int = field(default_factory=lambda: _env_int("VECTOR_K", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    fusion_top_k: int = field(default_factory=lambda: _env_int("FUSION_TOP_K", 20))
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 5))
    max_query_variants: int = field(default_factory=lambda: _env_int("MAX_QUERY_VARIANTS", 3))


@dataclass
class ModelConfig:
    """Which LLM / embedder / reranker to use. Swap via env vars, not code."""

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-sonnet-5"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    # Ollama unloads a model from RAM after this much idle time (default 5m)
    # — confirmed by hitting `ollama ps` mid-session and finding it empty
    # between calls, meaning every call was paying a ~minutes-long cold
    # reload. Setting OLLAMA_KEEP_ALIVE as a plain env var does NOT fix this
    # — that only affects an `ollama serve` process started fresh with it
    # set, not an already-running Ollama app/service. This has to be sent
    # per-request instead, which is what ChatOllama's `keep_alive` field is
    # for (confirmed present via ChatOllama.model_fields).
    ollama_keep_alive: str = field(default_factory=lambda: _env("OLLAMA_KEEP_ALIVE", "30m"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    )
    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    )


@dataclass
class GraphConfig:
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 2))


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    db: DBConfig = field(default_factory=DBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    # Lazily-built, cached client handles. Deliberately NOT dataclass fields
    # (no type annotation) so `dataclass` leaves them as plain class
    # attributes instead of constructor params / __eq__ members; each
    # instance shadows them with its own cache on first use.
    _llm = None
    _embedder = None
    _reranker = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls()

    def get_llm(self):
        """Chat model used by nodes/rewrite.py, grade.py, generate.py."""
        if self._llm is None:
            if self.model.llm_provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                self._llm = ChatAnthropic(model=self.model.llm_model, temperature=self.model.llm_temperature)
            elif self.model.llm_provider == "openai":
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(model=self.model.llm_model, temperature=self.model.llm_temperature)
            elif self.model.llm_provider == "ollama":
                from langchain_ollama import ChatOllama

                self._llm = ChatOllama(
                    model=self.model.llm_model,
                    temperature=self.model.llm_temperature,
                    base_url=self.model.llm_base_url,
                    keep_alive=self.model.ollama_keep_alive,
                )
            else:
                raise ValueError(f"unknown llm_provider {self.model.llm_provider!r}")
        return self._llm

    def get_embedder(self):
        """Sentence embedder used by indexing.py (index-time) and
        nodes/retrieval.py's vector_search (query-time) — same model both
        times, or cosine similarity is meaningless."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model.embedding_model)
        return self._embedder

    def get_reranker(self):
        """Cross-encoder used by nodes/rerank.py."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.model.rerank_model)
        return self._reranker
