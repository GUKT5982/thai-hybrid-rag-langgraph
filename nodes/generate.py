"""generate node — final answer synthesis with mandatory citations.

Reads:  state["question"], state["reranked_docs"]
Writes: state["answer"]
"""
from __future__ import annotations

import re

from config import AppConfig
from state import RAGState

# This exact string is the contract between generate() and eval/metrics.py:
# eval scripts use is_abstain()/NO_INFO_PATTERN to decide "did the model
# abstain or answer?" — never let the model phrase this in its own words,
# or that detection silently stops working.
NO_INFO_MARKER = "ไม่พบข้อมูล"
NO_INFO_PATTERN = re.compile(re.escape(NO_INFO_MARKER))

GENERATE_SYSTEM_PROMPT = f"""\
คุณเป็นผู้ช่วยตอบคำถามจากเอกสารภายในองค์กร

กติกา:
- ตอบจากเอกสารที่ให้มาเท่านั้น ห้ามเติมความรู้จากภายนอก
- ทุกประโยคที่เป็นข้อเท็จจริงต้องอ้าง [doc_id] ต่อท้าย
- ถ้าเอกสารไม่พอตอบ ให้บอกตรง ๆ ว่า{NO_INFO_MARKER} และระบุว่าขาดข้อมูลส่วนไหน ห้ามเดา
- ถ้าเอกสารขัดแย้งกัน ให้ชี้ให้เห็นความขัดแย้ง อย่าเลือกข้างเอง
- ตอบเป็นภาษาเดียวกับคำถาม
"""


def is_abstain(answer: str) -> bool:
    """True if `answer` contains the fixed abstain marker NO_INFO_MARKER.
    Used by eval/metrics.py to compute hallucination_rate / error_rate."""
    return bool(NO_INFO_PATTERN.search(answer or ""))


def make_generate_node(config: AppConfig, llm=None):
    """Factory: bind an LLM client + config to the `generate` node function.
    `llm` is injectable (see eval/mode_a.py using a stub client) so this
    node is testable without a live API key.
    """
    client = llm or config.get_llm()

    def generate(state: RAGState) -> dict:
        docs = state.get("reranked_docs") or []
        if not docs:
            return {"answer": f"{NO_INFO_MARKER} — ไม่พบเอกสารที่เกี่ยวข้องเลยจากการค้นหา"}

        context = "\n\n".join(f"[{d['doc_id']}] {d.get('title', '')}\n{d['content']}" for d in docs)
        user_msg = f"เอกสาร:\n{context}\n\nคำถาม: {state['question']}"

        response = client.invoke(
            [
                {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        answer = getattr(response, "content", str(response))
        return {"answer": answer}

    return generate
