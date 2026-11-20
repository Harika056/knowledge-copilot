#!/usr/bin/env bash
# Deploys Knowledge Copilot to Google Cloud Run (built by Cloud Build from the Dockerfile in this folder).
#
#   PROJECT_ID=my-project \
#   DATABASE_URL='postgresql://user:pass@host/db?sslmode=require' \
#   GEMINI_API_KEY=... \
#   bash scripts/deploy_cloud_run.sh
#
# Optional: REGION (asia-south1) SERVICE (knowledge-copilot) MAX_INSTANCES (2) DEMO_MODE (1)
#           DEMO_ROLES (MEMBER,MANAGER,ADMIN) EMBED_MODEL CHAT_MODEL DAILY_CAP RATE_LIMIT_PER_MIN YES=1
# Secrets go to Secret Manager (never into the image or the command line). See docs/DEPLOY.md.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id}"
: "${DATABASE_URL:?Set DATABASE_URL to a Postgres with pgvector (see docs/DEPLOY.md, step 1)}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY (https://aistudio.google.com/apikey)}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-knowledge-copilot}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"
DEMO_MODE="${DEMO_MODE:-1}"
DEMO_ROLES="${DEMO_ROLES:-MEMBER,MANAGER,ADMIN}"
JWT_SECRET="${JWT_SECRET:-$(openssl rand -hex 32)}"

[ -f Dockerfile ] || { echo "Run this from the project root (the folder with the Dockerfile)."; exit 1; }
command -v gcloud >/dev/null || { echo "Install the gcloud CLI first: https://cloud.google.com/sdk/docs/install"; exit 1; }

if [ "${YES:-0}" != "1" ]; then
  echo "This creates a PUBLIC (unauthenticated) Cloud Run service '$SERVICE' in project '$PROJECT_ID' ($REGION)."
  echo "Demo logins: $([ "$DEMO_MODE" = 1 ] && echo "ON for roles $DEMO_ROLES (use ONLY with the bundled synthetic documents)" || echo OFF)."
  read -r -p "Continue? [y/N] " ok; [ "$ok" = "y" ] || exit 1
fi

gcloud config set project "$PROJECT_ID" >/dev/null
echo "Enabling APIs (first time takes a minute)..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com

put_secret() {  # name, value  -> create the secret or add a new version (value goes via stdin, not argv)
  if gcloud secrets describe "$1" >/dev/null 2>&1; then
    printf %s "$2" | gcloud secrets versions add "$1" --data-file=- >/dev/null
  else
    printf %s "$2" | gcloud secrets create "$1" --replication-policy=automatic --data-file=- >/dev/null
  fi
}
put_secret copilot-database-url "$DATABASE_URL"
put_secret copilot-gemini-key   "$GEMINI_API_KEY"
put_secret copilot-jwt-secret   "$JWT_SECRET"

# Cloud Run runs as the default compute service account unless you changed it; it must be able to read the secrets.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"
for s in copilot-database-url copilot-gemini-key copilot-jwt-secret; do
  gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:${RUNTIME_SA}" \
    --role=roles/secretmanager.secretAccessor >/dev/null
done

# '^@^' makes '@' the separator so DEMO_ROLES may contain commas.
ENV="^@^PROVIDER=gemini@DEMO_MODE=${DEMO_MODE}@DEMO_ROLES=${DEMO_ROLES}@GOOGLE_CLOUD_LOCATION=${REGION}"
for v in EMBED_MODEL CHAT_MODEL DAILY_CAP RATE_LIMIT_PER_MIN PRICE_IN_PER_M PRICE_OUT_PER_M; do
  [ -n "${!v:-}" ] && ENV="${ENV}@${v}=${!v}"
done

echo "Building and deploying (Cloud Build; 3-6 minutes)..."
gcloud run deploy "$SERVICE" --source . --region "$REGION" --allow-unauthenticated \
  --max-instances "$MAX_INSTANCES" --min-instances 0 --memory 512Mi --timeout 60 \
  --set-env-vars "$ENV" \
  --set-secrets "DATABASE_URL=copilot-database-url:latest,GEMINI_API_KEY=copilot-gemini-key:latest,JWT_SECRET=copilot-jwt-secret:latest"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo; echo "Deployed: $URL"
for i in 1 2 3 4 5; do
  if curl -fsS "$URL/health"; then echo; echo "Healthy. Open $URL"; exit 0; fi
  sleep 5
done
echo "Health check failed. Read the logs:  gcloud run services logs read $SERVICE --region $REGION --limit 50"; exit 1
