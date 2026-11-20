# Knowledge Copilot

Question answering over a company's documents **and** its database, built the way an enterprise deployment needs it:
answers cite their sources, users only see what their role allows, the model's SQL is checked before it runs, and every
change to retrieval is measured on a golden question set.

![Architecture](docs/images/architecture.svg)

## 1. Try it in 2 minutes (no API key, no database)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                   # 66 pass, 10 skip (those need Postgres, see section 3)
python -m copilot.cli ask "How often are backups taken?" --role MEMBER
python -m copilot.cli ask "What is the L4 engineer salary band?" --role MEMBER   # -> not found (restricted)
python -m copilot.cli ask "What is the L4 engineer salary band?" --role ADMIN
```
Offline mode uses word-hash embeddings and an extractive "model". It proves the plumbing, **not** answer quality.

## 2. Run it for real (Gemini + Postgres/pgvector + a real corpus)
```bash
cp .env.example .env                 # fill in GEMINI_API_KEY (https://aistudio.google.com/apikey) and JWT_SECRET
set -a; source .env; set +a
docker run -d --name pg -p 5432:5432 -e POSTGRES_USER=copilot -e POSTGRES_PASSWORD=copilot -e POSTGRES_DB=copilot pgvector/pgvector:pg16
export DATABASE_URL=postgresql://copilot:copilot@localhost:5432/copilot

python -m copilot.cli check          # tells you exactly what is misconfigured (model names, key, database)

# a real corpus, e.g. the Kubernetes docs (any folder of markdown works)
git clone --depth 1 --filter=blob:none --sparse https://github.com/kubernetes/website ../website
(cd ../website && git sparse-checkout set content/en/docs/concepts)
python -m copilot.cli prepare --src ../website/content/en/docs/concepts --dst data/corpus \
       --internal cluster-administration configuration --restricted security
python -m copilot.cli ingest --docs data/corpus            # embeds ~3,000 chunks with Gemini, stores them in pgvector
python -m copilot.cli ask "How do I encrypt Secret data at rest?" --role ADMIN

# golden questions: let Gemini DRAFT them, then you review/delete weak ones (aim for 50-100 good ones)
python -m copilot.cli draft-golden --docs data/corpus --n 120 --out eval/golden.draft.jsonl
cp eval/golden.draft.jsonl eval/golden.jsonl               # after editing
python eval/run_eval.py --docs data/corpus --answers       # ablation table with REAL numbers

DOCS_ROOT=data/corpus uvicorn copilot.main:build_app --factory --port 8000
```
`docker compose up --build` runs the API and Postgres together (put your settings in `.env`).

**Deploy to Cloud Run:** `docs/DEPLOY.md` (one script; free Postgres with pgvector; spending and abuse protection built in).

### Web UI (good for demos and screenshots)
```bash
DEMO_MODE=1 uvicorn copilot.main:build_app --factory --port 8000     # open http://localhost:8000
```
Switch between Member / Manager / Admin and ask the same question: the answer, the cited sources (with their access
labels), latency, tokens and cost change with the role. **`DEMO_MODE` lets anyone who opens the page pick a role.**
Use it locally only, or for a public demo set `DEMO_ROLES=MEMBER` so nobody can become Admin.
Capture the images for a post with `python scripts/screenshots.py` (see `docs/PUBLISH.md`).

## 3. What has actually been verified
| Verified | How |
|---|---|
| Chunking, BM25, RRF, rerank, ACL, SQL guard, router, JWT, API, web UI, rate limiting, demo safety guards | 66 unit/API tests; ACL tests proven to fail when access control is deliberately broken |
| **PgVectorStore on real PostgreSQL 16 + pgvector 0.6** | 9 integration tests (`TEST_DATABASE_URL=... pytest`), incl. parity with the in-memory store on the golden set. They found a real bug: Postgres full-text search ANDed every word, so natural questions returned nothing (keyword hit@4 = 0.0). Fixed with OR semantics. |
| **Real corpus (Kubernetes docs, 189 files, 3,018 chunks)** | 450 role-based queries: **0** restricted/internal chunks leaked to lower roles; retrieval p50 25 ms / p95 40 ms on local Postgres (offline embeddings); filtered HNSW recall@10 = 0.999 vs exact search |
| Idle-connection drops (serverless Postgres) | store reconnects transparently; tested by killing the live connection on real Postgres |
| Image layout | the Docker layout was simulated (fresh non-editable install, run as the container would). The `docker build` and the Cloud Run deploy themselves were not run |
| Gemini adapters | tested against the SDK's own response types (field names, batching, normalisation, token usage) |
| Relevance cutoff (`MIN_SCORE`) | on that corpus, `MIN_SCORE=1.0` kept 98% of answerable questions and refused 5 of 6 questions whose only sources were restricted. **Small sample, lexically easy questions: tune on your own golden set.** |

**Not verified (no network access to Google from where this was built):** live calls to Gemini/Vertex AI
(`copilot check` does this for you), the Docker image build / docker-compose, and the Cloud Run deploy.
Model names are configurable (`EMBED_MODEL`, `CHAT_MODEL`) because they change over time.

## Security design (what to say in an interview)
- **Access is enforced inside the retrieval query**, for both vector and keyword search, not by filtering afterwards:
  a restricted chunk can't be returned and can't influence ranking. Folder = label (`public/`, `internal/`, `restricted/`).
- **SQL defence in depth:** parse → one statement → SELECT only → table allow-list → forced LIMIT → read-only connection.
  A test proves the database rejects writes even if the guard were bypassed. SQL answers are built from the result rows.
- JWT with expiry and a role allow-list; weak or default secrets are refused outside offline mode.
- Integration tests use their own table (`copilot_test_chunks`) and never touch a real `chunks` table.

## Known limits
- The reranker is a lexical baseline; swap in a cross-encoder or Vertex AI ranking for quality.
- Without a cutoff, vector search returns the nearest chunks even when nothing is relevant (see `MIN_SCORE`).
- Router is rule-based. pgvector's filtered HNSW search can return fewer rows on very selective filters; measured fine here
  (recall 0.999), but re-check on your data (pgvector 0.8+ has iterative scans).
- Faithfulness here is a lexical proxy; use an LLM judge (the evaluation-toolkit project) for a real score.

## Filling the resume placeholders
| Placeholder | Measure it with |
|---|---|
| `[N]` pages, `[N]`-question golden set | your corpus size; lines in `eval/golden.jsonl` |
| best hit@5 / faithfulness vs naive vector | `eval/run_eval.py` rows `vector` vs `hybrid+rerank` (`--k 5`), run with `PROVIDER=gemini` |
| p95 latency / cost per request | `run_eval.py --answers` (set `PRICE_IN_PER_M`/`PRICE_OUT_PER_M` from Google's price list) |
| `[N]` strategies compared | rows in the ablation table |
