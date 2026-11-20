# Publish to GitHub and make the LinkedIn images

## 1. Run it for real first (README section 2)
Screenshots must come from YOUR real run (Gemini + Postgres), not the offline demo. Offline answers are
extractive quotes, and posting them as AI output would be misleading.

## 2. Capture the images
```bash
pip install playwright && playwright install chromium
DEMO_MODE=1 PROVIDER=gemini DATABASE_URL=... JWT_SECRET=$(openssl rand -hex 32) \
  uvicorn copilot.main:build_app --factory --port 8000 &
python scripts/screenshots.py --url http://localhost:8000 --out docs/images
python scripts/make_architecture.py
```
You get: `01-member-cannot-see-restricted`, `02-admin-can-see-restricted` (the same question, two roles),
`03-public-answer-with-citation`, `04-sql-answer-with-guard`, and `architecture.png`.
The SQL screenshot needs a real model (the offline one cannot write SQL). Also screenshot your terminal:
`pytest` (all passing) and the `eval/run_eval.py` table with your real numbers.

## 3. Push
Create an EMPTY repository on github.com (no README, no .gitignore, no license), then:
```bash
cd knowledge-copilot
git init -b main
git add .
git status                      # check: no .env, no data/ folder, no API keys
git commit -m "Knowledge Copilot: hybrid RAG + guarded text-to-SQL with role-based access"
git remote add origin https://github.com/<your-username>/knowledge-copilot.git
git push -u origin main
```
Then on the repo page: add the topics `rag`, `pgvector`, `gemini`, `fastapi`, `text-to-sql`; check that the
Actions tab shows a green run (CI starts a real pgvector database); add the screenshots to the README
under a "Screenshots" heading. NEVER commit `.env` or an API key (it is in `.gitignore`; if one leaks, revoke it).

## 4. LinkedIn
Post 3-5 images: architecture, the Member vs Admin pair, the SQL guard, the eval table. First comment: the repo link.
Draft (fill every [bracket] with YOUR measured result; delete any line you cannot back with a number):

> I built a question-answering system for company documents and data, and the hard part wasn't the AI.
>
> It was making sure people only get answers from what they're allowed to read.
>
> What it does:
> • Hybrid search (keyword + vector) over [N] pages in Postgres/pgvector, answers with citations
> • Role-based access enforced inside the search query itself, so restricted text can never reach the model or the user
> • Text-to-SQL that must pass a safety check (one SELECT, allow-listed tables, forced LIMIT) and runs on a read-only connection
> • Evaluated on [N] golden questions: hit@5 [X]%, p95 latency [X] ms, [$X] per request
>
> A bug the tests caught: Postgres full-text search required every word to match, so natural questions returned nothing and "hybrid" quietly became vector-only. Fixed, and now covered by an integration test against a real database.
>
> Built with Python, FastAPI, Gemini, PostgreSQL + pgvector. Code and architecture: [repo link]
>
> #RAG #GenAI #Python #PostgreSQL #GoogleCloud

Alt text for each image (LinkedIn asks): describe what it shows in one sentence, e.g. "Same question asked as a Member and as an Admin:
the Member gets 'could not find', the Admin gets the answer with a restricted source cited."
