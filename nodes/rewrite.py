"""query_rewrite node.

Reads:  state["question"]
Writes: state["rewritten_queries"] (list[str], element 0 is always the
        original question — rewriting is a recall booster, never a
        replacement for the literal question)

Does NOT touch retry_count — see graph.py's bump_retry node.
"""
from __future__ import annotations

import re

from config import AppConfig
from state import RAGState

REWRITE_SYSTEM_PROMPT = """\
คุณเป็นผู้ช่วยขยายคำถามเพื่อค้นหาเอกสาร
จากคำถามที่ให้มา ให้เขียนคำถามใหม่ 2 แบบที่สื่อความหมายเดียวกันแต่ใช้คำ/มุมมองต่างกัน
เพื่อเพิ่มโอกาสค้นเจอเอกสารที่เกี่ยวข้อง (paraphrase, ขยายศัพท์เฉพาะ, หรือถามในมุมกลับ)
ห้ามแปลคำถามเป็นภาษาอื่นเด็ดขาด คำถามใหม่ต้องเป็นภาษาเดียวกับคำถามต้นฉบับทุกคำถาม
ตอบเป็นบรรทัดละหนึ่งคำถามเท่านั้น ไม่ต้องมีลำดับเลขหรือคำอธิบายอื่น
"""

_THAI_RANGE = re.compile(r"[฀-๿]")


def _same_language(original: str, candidate: str) -> bool:
    """Corpus is Thai-only, so a rewritten query in a different language
    doesn't just fail to help BM25 — it actively hurts. Confirmed by
    actually running scripts/ask.py: llama3 (via Ollama) translated two of
    three rewrites into English despite the system prompt forbidding it
    ("What is the study of celestial bodies..." for a Thai astronomy
    question), and BM25's per-doc max-score-across-variants aggregation
    (nodes/retrieval.py) let common English words that happened to appear
    in unrelated Thai articles' parenthetical glosses outscore the real
    Thai-language match — the actual astronomy passage didn't place in the
    top 20 BM25 hits at all for that run. The instruction alone isn't
    reliable enough on a small local model; this is the backstop."""
    original_is_thai = bool(_THAI_RANGE.search(original))
    if not original_is_thai:
        return True  # non-Thai questions aren't this project's concern here
    return bool(_THAI_RANGE.search(candidate))


def make_query_rewrite_node(config: AppConfig, llm=None):
    """Factory: bind an LLM client + config to a LangGraph node function."""

    def query_rewrite(state: RAGState) -> dict:
        question = state["question"]
        variants: list[str] = [question]

        client = llm or config.get_llm()
        try:
            response = client.invoke(
                [
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
            )
            text = getattr(response, "content", str(response))
            for line in text.strip().splitlines():
                # lstrip only: strip a leading "1. "/"- " enumeration marker,
                # but a trailing digit can be real content (e.g. a question
                # ending in a year) — .strip() on both ends silently ate that.
                line = line.lstrip("-*•0123456789. \t").rstrip()
                if not line or line == question:
                    continue
                # Smaller/local models (confirmed with llama3 via Ollama)
                # often prefix the list with meta-commentary instead of just
                # the questions, e.g. "Here are two paraphrased questions:" —
                # that line has no question content and would otherwise go
                # straight into bm25_search/vector_search as a bogus query.
                # A line ending in ":" with no "?" is preamble, not a variant.
                if line.endswith(":") and "?" not in line:
                    continue
                if not _same_language(question, line):
                    continue
                variants.append(line)
        except Exception:
            # Rewrite is a recall booster, not a hard dependency: on failure
            # fall back to searching with the original question alone.
            pass

        return {"rewritten_queries": variants[: config.retrieval.max_query_variants]}

    return query_rewrite
