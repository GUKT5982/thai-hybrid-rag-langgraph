"""Step 1 of the pipeline: pull both datasets down and confirm what we got.

Neither dataset is loaded through the `datasets` library — both go through
`snapshot_download` and are parsed directly with data/loaders.py. This was
NOT the original plan (miracl-corpus is nominally "just parquet, load with
`datasets`"), but running it against the live repo on 2026-09-03 gave:

    Dataset scripts are no longer supported, but found miracl-corpus.py

...i.e. miracl/miracl-corpus also still ships a legacy loading script, and
datasets>=3.0 (we have 5.0.1) refuses to run it, same failure mode as
nomiracl. Confirmed structure instead: miracl-corpus-v1.0-{lang}/docs-N.jsonl.gz
shards, same {"docid","title","text"} shape as nomiracl's passages — so one
code path (data/loaders.load_passages) reads both.

Thai corpus = 2 shards, ~110MB compressed, 542,166 passages. Small enough
to just download in full even for `DATA_MODE=sample` dev loops; sampling
happens by truncating how many rows indexing.py *reads*, not by fetching
less.

Run directly to sanity-check what actually got downloaded:
    python -m data.download
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from config import AppConfig
from data.loaders import find_passage_files, load_passages


def download_nomiracl(config: AppConfig) -> Path:
    """Downloads (or reuses the cached copy of) the nomiracl repo, filtered
    to Thai-ish files. The globs are intentionally broad — re-check the
    printed file list to confirm nothing unexpected slipped in.
    """
    local_dir = snapshot_download(
        repo_id=config.data.nomiracl_repo,
        repo_type="dataset",
        allow_patterns=["*thai*", "*/th/*", "*-th-*", "*th.*"],
    )
    path = Path(local_dir)
    print(f"[download] nomiracl -> {path}")
    _print_tree(path)
    return path


def download_corpus(config: AppConfig) -> Path:
    """Downloads just the shard(s) for one language out of miracl-corpus
    (the repo also holds ar/bn/de/en/... — dozens of languages we don't want).
    """
    local_dir = snapshot_download(
        repo_id=config.data.corpus_repo,
        repo_type="dataset",
        allow_patterns=[f"miracl-corpus-v1.0-{config.data.corpus_lang}/*"],
    )
    path = Path(local_dir)
    print(f"[download] miracl-corpus[{config.data.corpus_lang}] -> {path}")
    _print_tree(path)
    return path


def _print_tree(root: Path, max_files: int = 200) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"[download] {len(files)} files under {root}")
    for p in files[:max_files]:
        rel = p.relative_to(root)
        print(f"  {rel}  ({p.stat().st_size:,} bytes)")
    if len(files) > max_files:
        print(f"  ... and {len(files) - max_files} more")


def peek_corpus(config: AppConfig, corpus_dir: Path, n: int = 3) -> None:
    """Reads the first few corpus rows straight off disk to confirm schema
    — no `datasets`, no loading 542k rows into memory."""
    shard = find_passage_files(corpus_dir, config.data.corpus_lang)[0]
    print(f"[download] miracl-corpus[{config.data.corpus_lang}] sample from {shard.name}:")
    for i, row in enumerate(load_passages(shard)):
        if i >= n:
            break
        preview = row["text"][:120].replace("\n", " ")
        print(f"  docid={row['docid']!r} title={row['title']!r} text={preview!r}...")


def main() -> None:
    config = AppConfig.from_env()
    nomiracl_dir = download_nomiracl(config)
    print()
    corpus_dir = download_corpus(config)
    print()
    peek_corpus(config, corpus_dir)
    print(f"\n[download] nomiracl local dir: {nomiracl_dir}")
    print(f"[download] miracl-corpus local dir: {corpus_dir}")


if __name__ == "__main__":
    main()
