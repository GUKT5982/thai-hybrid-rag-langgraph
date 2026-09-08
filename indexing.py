"""Builds both indexes over the MIRACL Thai corpus: BM25 (in-memory,
rank_bm25) and a pgvector table. Kept separate from graph.py / nodes/ —
indexing is a batch job; the graph only *queries* what this file builds.

Usage:
    python indexing.py --mode sample                # smart sample, fast dev loop (see below)
    python indexing.py --mode sample --skip-vector   # BM25 only, no Postgres needed
    python indexing.py --mode full                   # all 542,166 passages
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Iterator

from rank_bm25 import BM25Okapi

from config import AppConfig
from data.download import download_corpus
from data.loaders import find_passage_files, find_qrels_files, load_passages, load_qrels
from text_utils import thai_tokenize


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _collect_gold_doc_ids(config: AppConfig) -> set[str]:
    """Every doc_id NoMIRACL's dev-split qrels reference (relevant +
    non_relevant) — the passages any dev eval could actually need. Reading
    these TSVs is cheap; it's used to build a sample corpus that's
    guaranteed to contain real answers (see iter_corpus's "sample" branch)."""
    root = config.data.nomiracl_local_dir
    ids: set[str] = set()
    for subset in ("relevant", "non_relevant"):
        for path in find_qrels_files(root, config.data.nomiracl_lang, "dev", subset):
            ids.update(load_qrels(path)["docid"])
    return ids


def iter_corpus(config: AppConfig) -> Iterator[dict]:
    """Streams the local corpus shards. `full` reads every shard to the end,
    542k passages never sitting in memory at once.

    `sample` is a SMART sample, not a naive prefix slice — confirmed by
    running the real pipeline (see README "Verified results"): a plain
    "first N rows" sample almost never contains the passage that actually
    answers a given NoMIRACL query (a real dev.relevant question about
    สามก๊ก / Three Kingdoms retrieved a totally unrelated top hit, because
    its gold passage — doc_id far into the corpus — simply wasn't in the
    first 5,000 rows), which makes any eval run against it structurally
    meaningless rather than just "less thorough." So "sample" mode always
    includes every doc_id NoMIRACL's dev qrels reference (~9,700 passages,
    confirmed by actually counting them), then backfills
    `config.data.sample_size` additional sequential passages on top as BM25
    distractor volume. `sample_size` is therefore "how many extra
    distractors," not a hard total cap — if you need the old truncate-at-N
    behavior for a specific memory/speed constraint, use `full` and read
    fewer rows from iter_corpus yourself.
    """
    corpus_dir = download_corpus(config)  # no-op re-download if already cached
    shard_files = find_passage_files(corpus_dir, config.data.corpus_lang)

    if config.data.data_mode != "sample":
        for shard in shard_files:
            for row in load_passages(shard):
                yield {"doc_id": row["docid"], "title": row.get("title", ""), "text": row["text"]}
        return

    gold_ids = _collect_gold_doc_ids(config)
    print(f"[indexing] smart sample: guaranteeing {len(gold_ids)} NoMIRACL dev gold doc_ids are present "
          f"(+ up to {config.data.sample_size} backfill distractors)")

    seen: set[str] = set()
    backfilled = 0
    for shard in shard_files:
        for row in load_passages(shard):
            doc_id = row["docid"]
            if doc_id in seen:
                continue
            is_gold = doc_id in gold_ids
            if not is_gold and backfilled >= config.data.sample_size:
                continue
            seen.add(doc_id)
            if not is_gold:
                backfilled += 1
            yield {"doc_id": doc_id, "title": row.get("title", ""), "text": row["text"]}

    print(f"[indexing] smart sample: {len(seen)} passages total ({len(seen) - backfilled} gold + {backfilled} backfill)")


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(passages: Iterable[dict], config: AppConfig) -> tuple[BM25Okapi, list[str], dict[str, dict]]:
    """Tokenizes every passage with the SAME newmm setting nodes/retrieval.py
    uses at query time (config.retrieval.use_newmm) — mismatched tokenization
    between index-time and query-time silently breaks BM25 recall (see
    eval/ablation.py's newmm_ablation, which exists specifically to catch that).

    Returns (bm25_index, doc_ids_in_index_order, corpus_lookup).
    """
    doc_ids: list[str] = []
    corpus: dict[str, dict] = {}
    tokenized: list[list[str]] = []

    for p in passages:
        doc_ids.append(p["doc_id"])
        corpus[p["doc_id"]] = p
        tokenized.append(thai_tokenize(p["text"], use_newmm=config.retrieval.use_newmm))

    if not tokenized:
        raise ValueError("no passages to index")

    bm25 = BM25Okapi(tokenized)
    return bm25, doc_ids, corpus


def save_bm25_index(bm25: BM25Okapi, doc_ids: list[str], corpus: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids, "corpus": corpus}, fh)
    print(f"[indexing] BM25 index saved -> {path} ({len(doc_ids)} passages)")


def load_bm25_index(path: Path) -> tuple[BM25Okapi, list[str], dict[str, dict]]:
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    return blob["bm25"], blob["doc_ids"], blob["corpus"]


# ---------------------------------------------------------------------------
# pgvector index
# ---------------------------------------------------------------------------

class PgVectorStore:
    """Thin wrapper: connects to Postgres+pgvector and exposes the interface
    nodes/retrieval.py needs (`similarity_search`). Embeds with
    config.get_embedder() so index-time and query-time use the same model.

    rank_bm25's in-memory index has no "production" story — the docstring
    on RetrievalConfig.use_newmm and the README note the alternative
    (PostgreSQL tsvector) for BM25 at scale; this class only covers the
    vector half, which needs a real DB regardless of scale.
    """

    def __init__(self, config: AppConfig):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self.config = config
        self.embedder = config.get_embedder()
        self.conn = psycopg2.connect(config.db.dsn)
        register_vector(self.conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.config.db.table_name} (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    embedding VECTOR({self.config.db.embedding_dim})
                );
                """
            )
        self.conn.commit()

    def build(self, passages: Iterable[dict], batch_size: int = 256) -> int:
        total = 0
        batch: list[dict] = []
        for p in passages:
            batch.append(p)
            if len(batch) >= batch_size:
                total += self._upsert_batch(batch)
                batch = []
        if batch:
            total += self._upsert_batch(batch)

        with self.conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.config.db.table_name}_embedding_idx "
                f"ON {self.config.db.table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            )
        self.conn.commit()
        print(f"[indexing] pgvector: upserted {total} passages into {self.config.db.table_name}")
        return total

    def _upsert_batch(self, batch: list[dict]) -> int:
        texts = [p["text"] for p in batch]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        with self.conn.cursor() as cur:
            for p, emb in zip(batch, embeddings):
                cur.execute(
                    f"""
                    INSERT INTO {self.config.db.table_name} (doc_id, title, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title, content = EXCLUDED.content, embedding = EXCLUDED.embedding;
                    """,
                    (p["doc_id"], p.get("title", ""), p["text"], emb.tolist()),
                )
        self.conn.commit()
        return len(batch)

    def similarity_search(self, query: str, k: int) -> list[dict]:
        [emb] = self.embedder.encode([query], normalize_embeddings=True)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT doc_id, title, content, 1 - (embedding <=> %s) AS score
                FROM {self.config.db.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (emb.tolist(), emb.tolist(), k),
            )
            rows = cur.fetchall()
        return [{"doc_id": r[0], "title": r[1], "content": r[2], "score": float(r[3])} for r in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "full"], default=None)
    parser.add_argument("--skip-vector", action="store_true", help="build BM25 only, skip the vector store entirely")
    parser.add_argument("--local-vector", action="store_true",
                         help="build+save a LocalVectorStore (in-memory NumPy, no Postgres needed) instead of "
                              "PgVectorStore — see local_vector_store.py. Ignored if --skip-vector is set.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.mode:
        config.data.data_mode = args.mode

    print(f"[indexing] data_mode={config.data.data_mode} corpus={config.data.corpus_repo}[{config.data.corpus_lang}]")
    passages = list(iter_corpus(config))
    print(f"[indexing] loaded {len(passages)} passages")

    bm25, doc_ids, corpus = build_bm25_index(passages, config)
    save_bm25_index(bm25, doc_ids, corpus, config.data.cache_dir / f"bm25_{config.data.data_mode}.pkl")

    if args.skip_vector:
        print("[indexing] --skip-vector: no vector store built")
    elif args.local_vector:
        from local_vector_store import LocalVectorStore

        store = LocalVectorStore(config)
        store.build(({"doc_id": d, **corpus[d]} for d in doc_ids))
        store.save(config.data.cache_dir / f"local_vectors_{config.data.data_mode}.pkl")
    else:
        store = PgVectorStore(config)
        store.build(passages)


if __name__ == "__main__":
    main()
