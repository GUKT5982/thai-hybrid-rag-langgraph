"""LocalVectorStore — an in-memory, NumPy-based stand-in for
indexing.PgVectorStore, implementing the exact same interface
(`.build(passages)`, `.similarity_search(query, k)`) that
nodes/retrieval.py's vector_search node and eval/mode_b.py depend on.

Why this exists: this project's spec calls for Postgres+pgvector as the
vector store, and PgVectorStore (indexing.py) is still the documented
production path — nothing about that changed. But no Postgres instance is
available in this dev environment (confirmed: port 5432 closed, no
service, no Docker installed either), which blocked Mode B and the
bm25/vector/hybrid ablation entirely. LocalVectorStore unblocks both
without any server: same embedding model (config.get_embedder()), same
cosine-similarity ranking, just held in a NumPy array in this process's
memory instead of a Postgres table — a drop-in replacement anywhere a
`vector_store` object is passed in, and safe to swap for PgVectorStore
later with no other code changes.

Not persisted, not concurrent-safe, and rebuilds from scratch on every
process start — fine for dev/eval on the sample corpus (a few thousand
passages), not a substitute for pgvector at full-corpus (542k) scale or in
a real multi-request server.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from config import AppConfig


class LocalVectorStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = config.get_embedder()
        self.doc_ids: list[str] = []
        self.titles: list[str] = []
        self.contents: list[str] = []
        self.embeddings: np.ndarray | None = None  # (n_docs, dim), L2-normalized rows

    def build(self, passages: Iterable[dict], batch_size: int = 256) -> int:
        self.doc_ids, self.titles, self.contents = [], [], []
        vectors: list[np.ndarray] = []

        batch: list[dict] = []

        def flush(batch: list[dict]) -> None:
            texts = [p["text"] for p in batch]
            emb = self.embedder.encode(texts, normalize_embeddings=True)
            vectors.append(np.asarray(emb, dtype=np.float32))
            for p in batch:
                self.doc_ids.append(p["doc_id"])
                self.titles.append(p.get("title", ""))
                self.contents.append(p["text"])

        for p in passages:
            batch.append(p)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

        self.embeddings = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
        print(f"[LocalVectorStore] indexed {len(self.doc_ids)} passages in memory "
              f"(embedding dim={self.embeddings.shape[1] if self.embeddings.size else 0})")
        return len(self.doc_ids)

    def save(self, path: Path) -> None:
        """Persist doc_ids/titles/contents/embeddings so rebuilding doesn't
        mean re-embedding every passage again — confirmed ~161ms/passage on
        this machine's CPU (intfloat/multilingual-e5-base), i.e. ~40 min for
        the ~14.7k-passage smart sample. Embed once, reuse across runs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "titles": self.titles,
                    "contents": self.contents,
                    "embeddings": self.embeddings,
                },
                fh,
            )
        print(f"[LocalVectorStore] saved {len(self.doc_ids)} passages -> {path}")

    @classmethod
    def load(cls, config: AppConfig, path: Path) -> "LocalVectorStore":
        store = cls(config)
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        store.doc_ids = blob["doc_ids"]
        store.titles = blob["titles"]
        store.contents = blob["contents"]
        store.embeddings = blob["embeddings"]
        print(f"[LocalVectorStore] loaded {len(store.doc_ids)} passages <- {path}")
        return store

    def similarity_search(self, query: str, k: int) -> list[dict]:
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []
        [q_emb] = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        # Both sides are L2-normalized, so dot product == cosine similarity
        # — same score definition PgVectorStore uses (1 - cosine_distance).
        scores = self.embeddings @ q_emb
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            {
                "doc_id": self.doc_ids[i],
                "title": self.titles[i],
                "content": self.contents[i],
                "score": float(scores[i]),
            }
            for i in top_idx
        ]
