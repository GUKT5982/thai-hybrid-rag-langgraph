"""Mode A — generate-only eval (do this before touching pgvector at all).

NoMIRACL already hands us the (<=10) annotated passages for each query, so
we feed those straight into generate() — no retrieval, no indexing. This
isolates the prompt + abstain-detection logic from anything retrieval
related: if this fails, the bug is here, not in BM25/vector search.

    python -m eval.mode_a --split dev --limit 50
    python -m eval.mode_a --split dev --limit 5 --stub    # no API key needed,
                                                            # smoke-tests the
                                                            # harness only —
                                                            # NOT real metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import AppConfig
from data.loaders import load_nomiracl_subset
from nodes.generate import make_generate_node, NO_INFO_MARKER
from eval.metrics import hallucination_rate, error_rate


class StubLLM:
    """Deterministic fake client for `--stub` runs: answers using the
    highest-relevance passage's text as its citation, or abstains if every
    passage is relevance 0. Used only to prove the eval *harness* (metric
    computation, abstain detection, JSON/report output) runs correctly
    without needing ANTHROPIC_API_KEY — it is NOT a substitute for real
    hallucination/error numbers, which require an actual model.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def invoke(self, messages):
        user_msg = messages[-1]["content"]
        if "[" in user_msg and any(f"] " in line and not line.strip().startswith(NO_INFO_MARKER) for line in user_msg.splitlines()):
            # Cheap heuristic: if any cited doc line looks non-trivial, "answer".
            for line in user_msg.splitlines():
                if line.startswith("["):
                    doc_id = line.split("]")[0].lstrip("[")
                    return self._Resp(f"คำตอบอ้างอิงจากเอกสาร [{doc_id}] (stub answer, not a real generation)")
        return self._Resp(f"{NO_INFO_MARKER} — stub: ไม่มีเอกสารที่ดูเกี่ยวข้อง")


def run_subset(config, records, llm, limit) -> list[dict]:
    generate = make_generate_node(config, llm=llm)
    results = []
    for record in records[:limit]:
        docs = [
            {"doc_id": p["docid"], "title": p.get("title", ""), "content": p["text"]}
            for p in record["passages"]
        ]
        state = {"question": record["query"], "reranked_docs": docs}
        out = generate(state)
        results.append({"query_id": record["query_id"], "query": record["query"], "answer": out["answer"]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=50, help="cap per subset, keep dev loops cheap")
    parser.add_argument("--out", default="eval/results/mode_a.json")
    parser.add_argument("--stub", action="store_true", help="use StubLLM instead of a real API call")
    args = parser.parse_args()

    config = AppConfig.from_env()
    llm = StubLLM() if args.stub else config.get_llm()

    root = config.data.nomiracl_local_dir
    relevant = load_nomiracl_subset(root, config.data.nomiracl_lang, args.split, "relevant")
    non_relevant = load_nomiracl_subset(root, config.data.nomiracl_lang, args.split, "non_relevant")

    print(f"[mode_a] relevant={len(relevant)} non_relevant={len(non_relevant)} (using first {args.limit} each)")
    if args.stub:
        print("[mode_a] --stub: using a fake LLM. Numbers below are a harness smoke test, NOT real metrics.")

    relevant_results = run_subset(config, relevant, llm, args.limit)
    non_relevant_results = run_subset(config, non_relevant, llm, args.limit)

    err = error_rate([r["answer"] for r in relevant_results])
    halluc = hallucination_rate([r["answer"] for r in non_relevant_results])

    print("\n=== Mode A results ===")
    print(f"{'subset':<14}{'n':>6}{'metric':>22}{'value':>10}")
    print(f"{'relevant':<14}{len(relevant_results):>6}{'error_rate':>22}{err:>10.3f}")
    print(f"{'non_relevant':<14}{len(non_relevant_results):>6}{'hallucination_rate':>22}{halluc:>10.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "stub": args.stub,
                "error_rate": err,
                "hallucination_rate": halluc,
                "relevant_results": relevant_results,
                "non_relevant_results": non_relevant_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[mode_a] full results -> {out_path}")


if __name__ == "__main__":
    main()
