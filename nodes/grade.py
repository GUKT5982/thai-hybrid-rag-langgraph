"""grade_documents node — LLM-as-judge on whether reranked_docs can answer
the question.

Reads:  state["question"], state["reranked_docs"]
Writes: state["documents_sufficient"] (bool), state["grade_reason"] (str)

Does NOT touch retry_count — the conditional edge in graph.py owns that,
because whether a retry is *allowed* (retry_count < max_retries) is a
graph-routing concern, not a grading concern.
"""
from __future__ import annotations

from config import AppConfig
from state import RAGState

GRADE_SYSTEM_PROMPT = """\
คุณเป็นผู้ตรวจสอบว่าเอกสารที่ให้มาเพียงพอต่อการตอบคำถามหรือไม่
ตอบด้วยคำเดียวขึ้นต้นประโยคเท่านั้น: "พอ" หรือ "ไม่พอ"
"""


def make_grade_documents_node(config: AppConfig, llm=None):
    client = llm or config.get_llm()

    def grade_documents(state: RAGState) -> dict:
        docs = state.get("reranked_docs") or []
        if not docs:
            return {"documents_sufficient": False, "grade_reason": "ไม่มีเอกสารที่ค้นเจอ"}

        context = "\n\n".join(f"[{d['doc_id']}] {d['content']}" for d in docs)
        user_msg = f"คำถาม: {state['question']}\n\nเอกสาร:\n{context}"
        try:
            response = client.invoke(
                [
                    {"role": "system", "content": GRADE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )
            text = getattr(response, "content", str(response)).strip()
            sufficient = text.startswith("พอ")  # "ไม่พอ" does NOT start with "พอ", no extra check needed
        except Exception as exc:
            # Fail open: if the judge itself errors out, let generate() decide
            # rather than looping on an infra failure until max_retries.
            return {"documents_sufficient": True, "grade_reason": f"grade error: {exc}"}

        return {"documents_sufficient": sufficient, "grade_reason": text}

    return grade_documents
