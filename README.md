# Thai Hybrid RAG (LangGraph)

A Thai + English RAG pipeline: BM25 + pgvector hybrid retrieval, orchestrated
with LangGraph, measured against real ground truth (NoMIRACL) rather than
guesswork about whether the system is any good.

## Two datasets, two jobs — read this first

| dataset | job | goes into |
|---|---|---|
| [`miracl/miracl-corpus`](https://huggingface.co/datasets/miracl/miracl-corpus) (config `th`) | the thing you **index** | BM25 + pgvector |
| [`miracl/nomiracl`](https://huggingface.co/datasets/miracl/nomiracl) (config `thai`) | the thing you **evaluate with** | `eval/mode_a.py`, `eval/mode_b.py` |

**`miracl-corpus`**: 542,166 Wikipedia-Thai passages, pre-chunked — index
them as-is, don't re-chunk. Schema: `docid` (`"X#Y"`), `title`, `text`.

**`nomiracl`**: topics (questions) + qrels (answer key) + up to 10
annotated passages per question, split into:
- **`relevant`** — the question has >=1 relevant passage → the system
  **must answer**, with citations.
- **`non_relevant`** — every passage is irrelevant → the system **must say
  it found no information**.

Both `relevant` and `non_relevant` further split into `dev`/`test`. All
topics were written by native speakers, not translated from English.

Confusing these two is the single easiest way to break this project: if
you evaluate against `miracl-corpus` or index `nomiracl`, nothing will
error, the numbers will just be meaningless.

## Neither dataset loads through `datasets` — a real finding, not the original plan

The original plan was: pull `nomiracl` via `snapshot_download` (it ships a
legacy loading script `datasets>=3.0` can't run), but load `miracl-corpus`
normally via `datasets.load_dataset(..., streaming=True)` since it's "just
parquet."

Running that against the live repo on 2026-09-03 (`datasets==5.0.1`) gave:

```
Dataset scripts are no longer supported, but found miracl-corpus.py
```

`miracl-corpus` *also* still ships a legacy script. So both datasets are
pulled with `snapshot_download` and parsed directly — `data/loaders.py`
reads both with the same `load_passages()` (gzip + json), since both turn
out to be sharded `*.jsonl.gz` files of `{"docid","title","text"}`, not
parquet. Confirmed shapes:

```
nomiracl (thai), one snapshot:
  data/thai/corpus.jsonl.gz            <- ALL annotated passages, every split/subset in one file
  data/thai/topics/{dev,test}.{relevant,non_relevant}.tsv
  data/thai/qrels/{dev,test}.{relevant,non_relevant}.tsv

miracl-corpus (th), one snapshot:
  miracl-corpus-v1.0-th/docs-0.jsonl.gz   (~102MB)
  miracl-corpus-v1.0-th/docs-1.jsonl.gz   (~8MB)     <- 542,166 rows total
```

Two consequences worth knowing about:
- `data/loaders.py` matches files by **directory** name (`topics/`,
  `qrels/`) and by exact dot-separated filename tokens
  (`"dev.relevant.tsv".split(".") == ["dev","relevant","tsv"]`), not
  substring matching — `"relevant" in "dev.non_relevant.tsv"` is `True`,
  which would silently merge the two subsets if you used substring checks.
- `nomiracl`'s passages are **not** split per subset — `corpus.jsonl.gz`
  holds everything; `load_nomiracl_subset()` filters it down via qrels'
  docids before loading, so it only ever materializes the passages a given
  subset actually needs.

Windows note: if Thai text prints as mojibake in your terminal, that's a
console codepage issue (this machine's `sys.stdout.encoding` defaulted to
`cp874`), not a data bug — set `PYTHONIOENCODING=utf-8` before running.

## Pipeline (graph.py)

```
        START
          │
    query_rewrite  (question -> question + 2-3 paraphrases)
        /    \
  bm25_search  vector_search      <- run concurrently (LangGraph fans out
        \    /                      from one predecessor, back in at fusion)
       fusion                     (RRF, see below)
          │
        rerank                    (cross-encoder, top rerank_top_k)
          │
   grade_documents                (LLM judge: "พอ" / "ไม่พอ")
        /    \
  sufficient  insufficient & retry_count < max_retries
      │             │
      │        bump_retry -> back to query_rewrite
      │
   generate -> END
```

### RRF (Reciprocal Rank Fusion) — `nodes/fusion.py`

For each doc, sum `1 / (k + rank)` over every ranked list it appears in
(rank is 1-indexed). `k` defaults to 60 (the original RRF paper's value) —
tune it via `RRF_K` in `.env`. We fuse by **rank**, not by raw score,
because BM25 scores and cosine similarities live on unrelated scales
(unbounded vs. `[-1, 1]`); summing them directly just lets whichever
happens to have the larger magnitude dominate. A doc ranked #1 by one
signal and absent from the other still outranks a doc that's merely
decent in both — usually the right call.

### The abstain contract

`generate()` is instructed to say the literal string `ไม่พบข้อมูล`
(`nodes/generate.py`'s `NO_INFO_MARKER`) when it can't answer. Every eval
script detects abstention by searching for that exact string
(`is_abstain()`) — it is a hardcoded constant, not something the model is
free to phrase itself, because the eval math breaks silently if it drifts.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (or OPENAI_API_KEY + LLM_PROVIDER=openai)
                        # and Postgres credentials for Mode B / full indexing
```

No Anthropic/OpenAI key? Set `LLM_PROVIDER=ollama` + `LLM_MODEL=<name from
"ollama list">` and run `ollama serve` locally — no key needed, but expect
it to be slower and less instruction-following-reliable than a hosted
frontier model (see the real Mode A run below for what that looks like in
practice).

A cloud-routed Ollama model (e.g. `LLM_MODEL=gemma4:31b-cloud`) goes through
the same local Ollama daemon and still needs no key in `.env` — just a
signed-in Ollama install and outbound HTTPS to ollama.com (where that path
is blocked it fails with a TLS handshake error — that's what stopped it in
the environment the 2026-09-03 results came from, see `.env.example`).
Verified 2026-09-29: ~1s per LLM call instead of 10-45s, and far more
reliable on the Thai-only / citation / `ไม่พบข้อมูล` instructions — see
"Mode A re-run with a stronger model" below. Make sure the Ollama app (or
`ollama serve`) is actually running first: `curl
http://127.0.0.1:11434/api/version`.

Mode A and the newmm ablation need **no Postgres** — start there.

## Running things, in the order that actually worked

1. **`python -m data.download`** — pulls both datasets, prints exactly what
   it got. Verified 2026-09-03: 9 nomiracl files (~750KB), 2 corpus shards
   (~110MB, 542,166 Thai passages).

2. **`python -m eval.mode_a --split dev --limit 50`** — generate-only eval,
   real passages straight from NoMIRACL into `generate()`, no indexing at
   all. Add `--stub` to smoke-test the harness (JSON output, metric math)
   with a fake deterministic LLM and no API key — that run is **not** a
   real hallucination/error measurement, it just proves the plumbing works.

3. **`python indexing.py --mode sample --skip-vector`** — builds the BM25
   index only (no Postgres needed). `sample` mode is a **smart sample**,
   not a naive prefix slice — see "sample mode had to be redesigned" below
   for why. Verified: query `"ดาราศาสตร์คือ"` (what is astronomy) top-ranks
   three genuinely on-topic passages from the ดาราศาสตร์ Wikipedia article.

4. **`python -m eval.ablation --mode sample --skip-retriever-ablation`** —
   the newmm on/off check, no Postgres needed. **Run this in CI on every
   change to `text_utils.py` or `indexing.py`** — see results below.

5. No Postgres instance was available in this environment (confirmed: port
   5432 closed, no service, no Docker either) — every command that
   originally needed pgvector (`indexing.py`, `eval/mode_b.py`,
   `eval.ablation`'s retriever-ablation half, `scripts/smoke_graph.py`'s
   vector search) now has a `--local-vector` flag (or, for
   `scripts/smoke_graph.py`, uses it by default) that swaps in
   `local_vector_store.py`'s `LocalVectorStore` — an in-memory NumPy cosine
   store behind the exact same interface as `PgVectorStore`, so it's a
   drop-in either way. `PgVectorStore` stays the documented production
   path; `LocalVectorStore` is explicitly a dev/small-corpus stand-in (see
   its own docstring for the "why" and its limits). Embedding is the slow
   part (~161ms/passage on this machine's CPU, confirmed) — `indexing.py
   --local-vector` builds it once and saves it to `data_cache/`, and
   `mode_b.py`/`ablation.py`/`smoke_graph.py` all load that cache instead
   of re-embedding on every run.

6. Scale up: `DATA_MODE=full` (or `--mode full`) re-runs the same commands
   over all 542,166 passages instead of the sample.

### `sample` mode had to be redesigned — a real bug caught by actually running it

The original design: `sample` = the first `DATA_SAMPLE_SIZE` (5,000) rows
of the corpus shards, for a fast dev loop. Running the full graph against
it (`scripts/smoke_graph.py`) surfaced why that's broken: a real
`dev.relevant` question about สามก๊ก (Three Kingdoms) retrieved a
top result about ประเทศญี่ปุ่น (Japan) — completely unrelated — because its
gold passage's `doc_id` simply wasn't in the first 5,000 rows. Every
eval number computed against a plain-prefix sample is measuring "did we
find an answer that was never present," not retrieval quality — not
*less accurate*, structurally meaningless.

Fix (`indexing.iter_corpus`): `sample` now always includes every doc_id
NoMIRACL's `dev` qrels reference — confirmed by counting them: 9,732
unique across `dev.relevant` + `dev.non_relevant` — then backfills
`DATA_SAMPLE_SIZE` (default 5,000) additional sequential passages on top
as BM25/vector distractor volume. `DATA_SAMPLE_SIZE` is therefore "how
many extra distractors," not a hard cap. Re-verified after the fix: the
same สามก๊ก query now ranks its real gold passage (`9800#4`) **#2 by BM25
alone**, and **#1 by vector search** (LocalVectorStore, cosine 0.829).

## Verified results (this session, `DATA_MODE=sample`, 2026-09-03)

### newmm ablation — real, not a stub

Re-run after the smart-sample fix, all 733 real `dev.relevant` NoMIRACL
queries (every one — smart sample guarantees every gold passage is
present, see below):

| tokenization | BM25 recall@10 |
|---|---|
| no newmm (whitespace split) | 0.334 |
| newmm (pythainlp) | **0.750** |
| gap | **+0.416** (>> the 0.05 CI failure threshold) |

This is the exact failure mode the ablation exists to catch: without word
segmentation, a Thai paragraph collapses into ~1 BM25 token and retrieval
is far worse than with it. `PASS`. (Earlier run, before the sample fix:
49 queries, 0.153 vs 0.718, +0.565 gap — same conclusion, 15x less
statistical power. Note the absolute recall numbers here run higher than a
full-corpus number would: the smart sample is ~66% gold passages by
construction, far denser than the real corpus's needle-in-542k-haystack —
trust the *comparison*, not the absolute recall, until a `DATA_MODE=full`
run.)

### Mode A (generate-only) — real run, `LLM_PROVIDER=ollama LLM_MODEL=llama3`

No Anthropic/OpenAI key was available in this environment; `llama3:8b`
via a local `ollama serve` was used instead (`--stub`'s fake LLM only
smoke-tests the harness plumbing — this run is the real thing). n=8/subset
(kept small: ~10-45s per generate() call on this machine's CPU).

| subset | n | metric | value |
|---|---|---|---|
| relevant | 8 | error_rate | **0.000** |
| non_relevant | 8 | hallucination_rate | **0.750** |

Reading the actual non_relevant answers (`eval/results/mode_a_llama3.json`)
before trusting that number, per the "report both metrics, always" rule —
two things stood out:

1. llama3 rarely emits the literal `ไม่พบข้อมูล` marker even when clearly
   hedging — e.g. one answer opens with "ข้อมูลที่มีอยู่ในเอกสารไม่ระบุ...
   อย่างชัดเจน" (the documents don't clearly specify this) and then keeps
   answering anyway instead of stopping there. `is_abstain()` correctly
   counts that as *not* abstaining per the exact-string contract this
   project deliberately requires (see "The abstain contract" above) — but
   it means part of this 0.75 is llama3's weak instruction-following, not
   only invented facts.
2. It also frequently ignored "answer in the same language as the
   question" and answered in English for Thai questions — same
   instruction-following gap, different symptom.
3. At least one answer looks like a genuine fabrication rather than a
   phrasing miss: asked "บริษัท ซูบารุ ก่อตั้งขึ้นเมื่อใด?" (when was
   Subaru founded) — a question NoMIRACL's annotators marked as
   unanswerable from the given passages — llama3 confidently answered
   "1958" anyway.

Net: this is a real measurement, not a placeholder, but a small
CPU-bound local model is a weak stand-in for judging the *prompt design* —
a lot of the 0.75 is instruction-following failure on top of whatever
true hallucination rate exists underneath it. Re-run with a stronger
model (Claude/GPT-4-class, or a larger local model) before drawing
conclusions about the prompt itself. (NoMIRACL paper reference points, not
this project's numbers: LLAMA-2 and Orca-2 hallucination_rate > 88%;
Mistral/LLAMA-3 hallucinate less but error_rate up to 74.9% — the two
metrics trade off against each other, which is exactly why both are
reported here even though error_rate landed at a suspiciously clean 0.)

### Mode A re-run with a stronger model — `gemma4:31b-cloud` via Ollama (2026-09-29)

The re-run the paragraph above asks for. It couldn't happen in the original
environment — cloud-routed Ollama models failed there with a TLS handshake
error — but works on a machine with normal outbound HTTPS, through the same
local Ollama app and `ChatOllama` path, no code changes. 100 `generate()`
calls took 1m57s (~1.2s each, vs. 10-45s for llama3 on CPU).
`python -m eval.mode_a --split dev --limit 50`:

| subset | n | metric | llama3 (n=8, above) | gemma4:31b-cloud (n=50) |
|---|---|---|---|---|
| relevant | 50 | error_rate | 0.000 | **0.060** (3/50) |
| non_relevant | 50 | hallucination_rate | 0.750 | **0.520** (26/50) |

Read all 100 answers before reading anything into the 0.520:

1. **No invented facts found.** 0/100 answers cite a doc_id that wasn't in
   the context, all 100 are in Thai (llama3 often switched to English), and
   in all 26 non_relevant answers that didn't abstain, every number (years,
   dates, counts) appears in the passages they cite. The Subaru question
   llama3 answered "1958" gets a correct `ไม่พบข้อมูล` here.
2. **Most of those 26 are label misses, not model errors.** 19 give a direct
   answer; spot-checking 9 of them, the cited passage states the answer
   outright every time, even though NoMIRACL marks every passage for these
   queries relevance 0 — e.g. q2301 asks how many provinces อำเภองาว borders
   and `35128#0` says "ชายแดนติดต่อกันถึงสามจังหวัด"; q2884 correctly chains
   `218388#0` (คุโรซากิ โอริฮิเมะ = อิโนะอุเอะ โอริฮิเมะ) with `218388#1`
   ("มีผมสีน้ำตาล"). A few answer a neighboring question instead — q1502 gives
   the year of บาดาลใจ's first TV adaptation when asked when it was first
   published — those are the soft errors worth fixing.
3. **The other 7 hedge without the marker**: "เอกสารไม่ได้ระบุ...อย่างชัดเจน
   แต่..." followed by related facts, or (q1559) pointing out that two
   passages contradict each other — which is exactly what the prompt asks
   for on conflicts. Under the exact-string contract `is_abstain()`
   correctly counts all 7 as answering.
4. **The same contract cuts the other way on error_rate.** q65 ("is
   chlorophyll found only in green leafy vegetables?") gets a correct
   answer — also in bacteria and algae, `[142149#0]` — but it opens with
   "ไม่พบข้อมูลว่า...เฉพาะ...", so it's counted as a refusal. Of the other two
   refusals, q144's relevant-labeled passage never names the first king of
   Wessex, so refusing there is defensible.

Net: with a model that actually follows instructions, what's left is the
prompt contract, not fabrication. `GENERATE_SYSTEM_PROMPT` never says what to
do when the passages hold *related but not answering* information, and "say
ไม่พบข้อมูล" is satisfied by using the phrase anywhere, in either direction.
Tightening it (e.g. "start the answer with ไม่พบข้อมูล when the documents
don't answer the question directly", plus a start-of-answer check in
`is_abstain()`) is the next prompt change to try — it changes the eval
contract, so it needs its own before/after run, not a silent edit.

### Retriever ablation (bm25/vector/hybrid) — real, via `--local-vector`

733 real `dev.relevant` queries, smart-sample corpus, `LocalVectorStore`
(so: no Postgres needed — `python -m eval.ablation --mode sample --local-vector`):

| retriever | recall@10 |
|---|---|
| BM25 only | 0.750 |
| **Vector only** | **0.920** |
| Hybrid (RRF) | 0.905 |

Not the result the spec expected to find ("prove hybrid beats either alone")
— **hybrid is slightly *worse* than vector-only here.** Reporting it
straight rather than only the result that confirms the design, because
that's what this ablation is for. Plausible reasons, none of them a code
bug:

1. Vector search has an easy target here — the smart sample is ~66% gold
   passages by construction (see above), a much lower-noise environment
   than the real 542k-passage corpus. BM25's relative disadvantage may
   shrink at full scale.
2. RRF weights both signals equally by rank. When one signal (vector) is
   much stronger than the other (BM25) on a given query, naive RRF can let
   a doc BM25 likes but vector doesn't displace one vector already ranked
   highly — a known RRF failure mode when signals are this unbalanced,
   not specific to this implementation.
3. This ablation script pulls top-10 per signal before fusing; `graph.py`'s
   real pipeline uses top-20 each (`BM25_K`/`VECTOR_K`) before RRF — hybrid
   behavior in the actual graph may differ from this specific ablation
   setup.

Conclusion to draw from this: **don't assume hybrid wins — verify at full
corpus scale before deciding fusion is worth its complexity here.** That
uncertainty is the correct output of this ablation, not a failure of it.

### Real bug found via `scripts/ask.py`: `query_rewrite` translating into English poisoned BM25

Asking `scripts/ask.py` a plain Thai question ("ดาราศาสตร์คืออะไร" — what is
astronomy) and reading every stage's output (not just the final answer)
surfaced this: despite `REWRITE_SYSTEM_PROMPT` being written entirely in
Thai, llama3 (via Ollama) rewrote 2 of the 3 query variants into English
("What is the study of celestial bodies and their movements called?").
`bm25_search` takes the max score per doc **across** all rewritten-query
variants — so common English words that happened to appear in unrelated
Thai articles' parenthetical glosses (e.g. "Jurisprudence" quoted inside a
นิติศาสตร์ article) out-scored the real Thai-language match under the
English variant. Result: the actual ดาราศาสตร์ passage didn't even place
in the top 20 BM25 hits for that run — BM25 came back with นิติศาสตร์,
ประเทศออสเตรเลีย, แฟร็กทัล, none of them on-topic.

**The system self-corrected anyway** — `vector_search` (embeddings don't
care about script/language the way lexical BM25 does) still ranked every
astronomy passage correctly, RRF fusion pulled them into the candidate
set, and the cross-encoder reranker put 4 of 5 final `reranked_docs` back
on-topic — so the final answer was still correct. That's real evidence for
why hybrid + rerank earns its complexity: one signal being actively wrong
didn't sink the pipeline. But it's still bandwidth wasted on garbage
candidates, and a harder query might not recover.

Fixed in `nodes/rewrite.py`: the prompt now explicitly forbids translating
the question, and — since a prompt instruction alone clearly isn't
reliable enough on a small local model — a `_same_language()` backstop
drops any rewritten variant that doesn't share Thai-script characters with
a Thai original, regardless of what the LLM does. Re-verified after the
fix: same question, rewritten_queries stayed Thai-only.

### Mode B (full pipeline) — first end-to-end run, `gemma4:31b-cloud` (2026-09-29)

Blocked in the original environment by the only available LLM being a slow
local one; with `gemma4:31b-cloud` it's ~5s per query.
`python -m eval.mode_b --mode sample --split dev --limit 30 --local-vector`
— 30 `relevant` + 30 `non_relevant` queries through the whole graph (the
non_relevant 30 are the first 30 of Mode A's 50 above), 7 minutes including
the BM25 rebuild. The vector cache it loads came from `python indexing.py
--mode sample --local-vector`: 14,732 passages embedded in ~28 min on an
8-core desktop CPU.

| metric | value |
|---|---|
| recall@10 (relevant, n=30) | 0.854 |
| nDCG@10 (relevant, n=30) | 0.824 |
| error_rate (relevant, n=30) | 0.200 (6/30) |
| hallucination_rate (non_relevant, n=30) | 0.533 (16/30) |

`recall@10` / `nDCG@10` are computed over `reranked_docs` — at most
`RERANK_TOP_K=5` docs, the ones `generate()` actually sees — so they're
effectively @5, not comparable to the retriever ablation's recall@10 over
top-10 lists.

The first end-to-end run surfaced a bug: `nDCG@10` printed `nan` on every
run. A `non_relevant` query's qrels are all relevance 0, so `ndcg_at_k`
returns NaN for it (ideal DCG is 0), and that NaN went into the mean. Fixed
by scoring nDCG only for queries that have relevant passages — the same gate
recall already used. `eval/mode_b.py` also now prints error_rate /
hallucination_rate, which its docstring talks about but it never computed.

Reading every row that counts against the model:

- **error_rate 0.200 is mostly retrieval — and the model handling it
  correctly.** Of the 6 relevant queries that abstained, 3 (q77, q136, q144)
  had recall 0: no gold passage reached `generate()`, so `ไม่พบข้อมูล` was
  the right answer and the fix is retrieval, not the prompt. 2 (q63, q65)
  are the marker problem from Mode A: correct content opened with
  "ไม่พบข้อมูลว่า...". The model never answered a relevant query without a
  gold passage in its context.
- **hallucination_rate 0.533 — again no fabrication found.** 0/16 cite a
  doc outside the retrieved set; the only number missing from its cited
  passage is q1324's "2503", which is the year from the question. 13 of the
  16 are questions Mode A also answered from NoMIRACL's own passages (see
  above). The other 3 flipped from abstaining in Mode A to answering here:
  - q240 and q1324 are exactly the caveat below — the retriever found a real
    answer NoMIRACL's passages didn't have: `266749#3` defines สหศึกษา
    ("โรงเรียนที่เรียนกันระหว่างนักเรียนชายและนักเรียนหญิง"), `80055#0` gives
    จอมพลสฤษดิ์'s term as 2502–2506 for "who was PM in 2503".
  - q333 ("is the Mekong in Bangkok?") had the same passage, `13797#0`, both
    times and gave the same facts both times; Mode A opened with
    "ไม่พบข้อมูลว่า..." (scored as abstaining), Mode B with "ไม่ใช่" (scored as
    a hallucination). Same contract problem as Mode A's point 4.

### Mode B interpretation caveat

`non_relevant` queries come from cases where **MIRACL's original
retriever** found nothing relevant — not proof Thai Wikipedia has no
answer. Mode A sidesteps this (fixed passage set); Mode B can't — a better
retriever may legitimately surface a real answer for a query labeled
`non_relevant`. That's why `eval/mode_b.py` always dumps 20-30 individual
cases to `eval/results/mode_b_examples.md` alongside the aggregate numbers
— read the `non_relevant` rows that didn't abstain before calling them
hallucinations.

## Config

Every k/threshold/model name lives in `config.py` as dataclass fields,
overridable via `.env` (see `.env.example`) — nothing is hardcoded in
`indexing.py`, `graph.py`, `nodes/`, or `eval/`.

## License

Both `miracl/miracl-corpus` and `miracl/nomiracl` are released under
Apache-2.0. However, `miracl-corpus`'s underlying text is Wikipedia
content, which is CC BY-SA — if you redistribute the indexed corpus text
itself (not just your code), CC BY-SA's share-alike/attribution terms
apply to that content regardless of the dataset wrapper's license.

## Layout

```
config.py              # AppConfig — every tunable value, loaded from .env
text_utils.py           # shared Thai tokenizer (index-time AND query-time)
state.py                # RAGState TypedDict
indexing.py              # BM25 (rank_bm25) + pgvector index builders, smart sample logic
local_vector_store.py    # in-memory NumPy vector store — no-Postgres stand-in for PgVectorStore
graph.py                 # StateGraph wiring (nodes/ + retry loop)
data/
  download.py            # snapshot_download for both datasets, prints what it got
  loaders.py              # jsonl.gz / topics.tsv / qrels.tsv readers, no `datasets` lib
nodes/
  rewrite.py               # query_rewrite
  retrieval.py              # bm25_search, vector_search (run concurrently)
  fusion.py                 # RRF
  rerank.py                 # cross-encoder
  grade.py                  # LLM-as-judge sufficiency check
  generate.py                # citation-constrained answer + NO_INFO_MARKER
eval/
  metrics.py               # hallucination_rate, error_rate, recall@k, nDCG@k
  mode_a.py                 # generate-only eval (+ --stub harness smoke test)
  mode_b.py                 # full-pipeline eval, dumps manual-review examples
  ablation.py                # newmm on/off (CI-able, no DB) + bm25/vector/hybrid
scripts/
  smoke_graph.py            # runs the FULL graph on real NoMIRACL questions with
                             # real BM25 + LocalVectorStore + LLM + reranker — for
                             # eyeballing the whole thing before eval/mode_b.py
  ask.py                     # interactive: type your OWN question, see every
                             # stage's real doc_ids/scores/snippets, not just the
                             # final answer — this is how the query_rewrite
                             # language bug above was actually found
```
