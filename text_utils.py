"""Shared Thai tokenization.

Used identically by indexing.py (index-time, building the BM25 index) and
nodes/retrieval.py (query-time, bm25_search). Keeping this in exactly one
place is what makes the newmm on/off ablation (eval/ablation.py) meaningful
— both arms of that comparison must tokenize the same way at both stages,
or the "ablation" is just measuring a bug.
"""
from __future__ import annotations


def thai_tokenize(text: str, use_newmm: bool = True) -> list[str]:
    """Tokenize `text` for BM25 indexing/querying.

    use_newmm=True  -> pythainlp's newmm word segmenter (the real thing).
    use_newmm=False -> naive whitespace split. Thai has no spaces between
        words, so this collapses most Thai text into a handful of giant
        "tokens" — that's the point: it's the negative arm of the newmm
        ablation in eval/ablation.py, not meant to be a usable tokenizer.
    """
    if not use_newmm:
        return text.split()

    from pythainlp.tokenize import word_tokenize

    tokens = word_tokenize(text, engine="newmm")
    return [t for t in tokens if t.strip()]
