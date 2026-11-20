# Deploy Knowledge Copilot (Google Cloud Run + a free Postgres)

Result: a public URL where anyone can try the copilot, with real Gemini answers, citations, and the
Member / Manager / Admin switch. Cloud Run scales to zero when idle. Check Google's and your database provider's
CURRENT free-tier limits before you start; they change.

**Cost safety first (5 minutes, do this before deploying):**
1. In Google Cloud Console → Billing → Budgets & alerts, create a small budget with email alerts.
   (Cloud Run needs a project with billing enabled, even if your usage stays inside the free tier.)
2. The deploy script limits the service to 2 instances. The public demo also has a per-IP rate limit (20/min)
   and a daily cap (300 requests per instance) that protect your Gemini quota. Change with `RATE_LIMIT_PER_MIN` / `DAILY_CAP`.

## Step 1: a Postgres with pgvector
Easiest free route: create a project on **Neon** (neon.tech) or **Supabase**, both offer pgvector. Copy the connection string
(it ends in `?sslmode=require`). The app creates the extension, table and indexes itself on first start.
(Cloud SQL for PostgreSQL also works but is an always-on paid instance.)

## Step 2: check everything from your laptop
```bash
cp .env.example .env     # fill GEMINI_API_KEY, JWT_SECRET (openssl rand -hex 32), DATABASE_URL
set -a; source .env; set +a
python -m copilot.cli check                            # must print OK for embeddings, generation and postgres
python -m copilot.cli ingest --docs sample_data/docs   # loads the synthetic sample documents
python -m copilot.cli ask "How often are backups taken?" --role MEMBER
```
If `check` fails it says what to fix. If Gemini reports an unknown model, set `EMBED_MODEL` / `CHAT_MODEL` to names your key can use.

## Step 3: deploy
```bash
gcloud auth login
PROJECT_ID=<your-project-id> DATABASE_URL="$DATABASE_URL" GEMINI_API_KEY="$GEMINI_API_KEY" \
  bash scripts/deploy_cloud_run.sh
```
It stores the secrets in Secret Manager, builds the image with Cloud Build, deploys, and prints the URL after a health check.

## Step 4: use it, then capture the LinkedIn images from the LIVE site
```bash
pip install playwright && playwright install chromium
python scripts/screenshots.py --url https://<your-service-url> --out docs/images
```
Open the URL, switch between Member and Admin, ask the same question. (See `docs/PUBLISH.md` for the GitHub and LinkedIn steps.)

## Safe-by-default rules the app enforces
- `DEMO_MODE` (login-free role switching) is **only allowed with the bundled synthetic documents**. With your own documents
  the app refuses to start unless `DEMO_ROLES=MEMBER`. Anyone who can open a demo page can pick the roles you allow.
- Demo logins cannot call `/ingest`. JWT secrets shorter than 32 characters are refused.
- To use your OWN documents: ingest them locally with `python -m copilot.cli ingest --docs data/corpus` (same `DATABASE_URL`),
  deploy with `DEMO_MODE=0`, and call the API with tokens you issue.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Deploy ends with "health check failed" | `gcloud run services logs read knowledge-copilot --region asia-south1 --limit 50` |
| Log says `JWT_SECRET` or `synthetic sample data` | the safety checks above; read the message, it says what to change |
| Log says connection refused / SSL error | the `DATABASE_URL` must be reachable from the internet and end with `?sslmode=require` |
| First request after idle is slow | Cloud Run cold start (scale to zero). Set `--min-instances 1` if you want it always warm (costs money) |
| Answers say "could not find" for everything | the table is empty or embeddings changed: re-run `ingest` with the same `EMBED_MODEL` you deploy with |
| Remove everything | `gcloud run services delete knowledge-copilot --region asia-south1` and delete the three `copilot-*` secrets |
