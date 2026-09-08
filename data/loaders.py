"""Readers for the raw files data/download.py pulls down — no `datasets`
library involved for either repo (see download.py docstring for why).

Confirmed against the live repos on 2026-09-03:

  miracl/nomiracl (thai), one snapshot dir containing:
    data/thai/corpus.jsonl.gz          — ALL annotated passages for every
                                          split/subset, in one file. Filter
                                          by docid via qrels, don't assume
                                          per-split passage files.
    data/thai/topics/dev.relevant.tsv          \\ filename dot-tokens are
    data/thai/topics/dev.non_relevant.tsv      / exactly {split}.{subset}.tsv
    data/thai/topics/test.relevant.tsv
    data/thai/topics/test.non_relevant.tsv
    data/thai/qrels/{same four}.tsv

  miracl/miracl-corpus, one snapshot dir containing (per language):
    miracl-corpus-v1.0-th/docs-0.jsonl.gz
    miracl-corpus-v1.0-th/docs-1.jsonl.gz    (2 shards, ~110MB compressed
                                               total for Thai, 542,166 rows)

Both are gzipped JSONL of {"docid", "title", "text"} — same shape, so one
`load_passages()` reads either. We still glob rather than hardcode exact
paths, and match filename tokens by exact dot-separated equality (not
substring — "relevant" is a substring of "non_relevant", so a naive
`"relevant" in name` check silently pulls in the wrong subset).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def _path_contains(path: Path, token: str) -> bool:
    token = token.lower()
    return any(token in part.lower() for part in path.parts)


def _filename_tokens(path: Path) -> list[str]:
    """"dev.relevant.tsv" -> ["dev", "relevant", "tsv"]."""
    return path.name.lower().split(".")


def find_passage_files(root: Path, lang: str) -> list[Path]:
    hits = sorted(p for p in root.rglob("*.jsonl.gz") if _path_contains(p, lang))
    if not hits:
        raise FileNotFoundError(
            f"no *.jsonl.gz under a path containing {lang!r} in {root} — "
            "re-run data/download.py and check the printed file tree; the repo layout may have changed"
        )
    return hits


def find_topic_files(root: Path, lang: str, split: Optional[str] = None, subset: Optional[str] = None) -> list[Path]:
    # "topic" lives in the parent DIRECTORY name (".../topics/dev.relevant.tsv"),
    # not in the filename itself — check the whole path, not p.name.
    hits = []
    for p in root.rglob("*.tsv"):
        if not _path_contains(p, "topic") or not _path_contains(p, lang):
            continue
        tokens = _filename_tokens(p)
        if split and split not in tokens:
            continue
        if subset and subset not in tokens:
            continue
        hits.append(p)
    if not hits:
        raise FileNotFoundError(f"no topics tsv for lang={lang} split={split} subset={subset} under {root}")
    return sorted(hits)


def find_qrels_files(root: Path, lang: str, split: Optional[str] = None, subset: Optional[str] = None) -> list[Path]:
    # Same deal: "qrel" lives in the parent directory name (".../qrels/...").
    hits = []
    for p in root.rglob("*.tsv"):
        if not _path_contains(p, "qrel") or not _path_contains(p, lang):
            continue
        tokens = _filename_tokens(p)
        if split and split not in tokens:
            continue
        if subset and subset not in tokens:
            continue
        hits.append(p)
    if not hits:
        raise FileNotFoundError(f"no qrels tsv for lang={lang} split={split} subset={subset} under {root}")
    return sorted(hits)


def load_passages(path: Path) -> Iterator[dict]:
    """Yields {"docid", "title", "text"} rows from one *.jsonl.gz shard.
    Always opens with explicit utf-8 — on Windows the console codepage
    (often cp874) will *display* Thai text as mojibake even when the bytes
    and the resulting str are correct; don't "fix" that by touching the
    encoding here, it's a terminal rendering issue, not a data issue.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_topics(path: Path) -> pd.DataFrame:
    """TREC-style topics TSV: query_id \\t query_text, no header."""
    return pd.read_csv(path, sep="\t", header=None, names=["query_id", "query"], dtype=str)


def load_qrels(path: Path) -> pd.DataFrame:
    """TREC qrels TSV: query_id, iteration, docid, relevance — no header."""
    df = pd.read_csv(path, sep="\t", header=None, names=["query_id", "iteration", "docid", "relevance"], dtype=str)
    df["relevance"] = df["relevance"].astype(int)
    return df


def load_nomiracl_subset(root: Path, lang: str, split: str, subset: str) -> list[dict]:
    """Assemble one of the four (dev|test) x (relevant|non_relevant) subsets.

    NoMIRACL ships topics + qrels + the (already-annotated, <=10-per-query)
    passages needed to judge each query — this is what eval/mode_a.py
    consumes directly, with no retrieval step involved.

    Returns: [{"query_id", "query", "passages": [{"docid","title","text","relevance"}]}]
    """
    topic_files = find_topic_files(root, lang, split, subset)
    qrels_files = find_qrels_files(root, lang, split, subset)
    passage_files = find_passage_files(root, lang)  # one shared corpus.jsonl.gz, not per-subset

    topics = pd.concat([load_topics(p) for p in topic_files], ignore_index=True)
    qrels = pd.concat([load_qrels(p) for p in qrels_files], ignore_index=True)

    # Only hydrate the passages this subset actually needs — corpus.jsonl.gz
    # holds annotated passages for every split/subset combined.
    needed_ids = set(qrels["docid"])
    passages_by_id: dict[str, dict] = {}
    for p in passage_files:
        for row in load_passages(p):
            if row["docid"] in needed_ids:
                passages_by_id[row["docid"]] = row

    records: list[dict] = []
    for _, topic in topics.iterrows():
        rows = qrels[qrels["query_id"] == topic["query_id"]]
        passages = []
        for _, r in rows.iterrows():
            passage = passages_by_id.get(r["docid"])
            if passage is None:
                continue
            passages.append({**passage, "relevance": int(r["relevance"])})
        records.append({"query_id": topic["query_id"], "query": topic["query"], "passages": passages})

    return records
