"""Run:  uvicorn copilot.main:build_app --factory --port 8000"""
import os
from pathlib import Path

from .api import create_app
from .config import Settings
from .factory import build_components
from .pipeline import Copilot
from .sample import build_sample_db

DOCS_ROOT = "sample_data/docs"


def build_app():
    s = Settings()
    if s.provider != "offline" and (s.jwt_secret == "dev-only-change-me" or len(s.jwt_secret) < 32):
        raise RuntimeError("Set JWT_SECRET to a random string of at least 32 characters.")
    sample = Path(DOCS_ROOT).resolve()
    if s.demo_mode and set(s.demo_roles) - {"MEMBER"} and Path(s.docs_root).resolve() != sample \
            and os.environ.get("ALLOW_DEMO_ADMIN") != "1":
        raise RuntimeError("DEMO_MODE with roles above MEMBER lets anyone open the page read restricted documents. "
                           "It is only allowed with the bundled synthetic sample data. For a deployment with your own "
                           "documents use DEMO_ROLES=MEMBER, or turn DEMO_MODE off and log in with real tokens.")
    embedder, llm, store = build_components(s)
    if not Path(s.sql_db_path).exists():
        build_sample_db(s.sql_db_path)
    copilot = Copilot(store, embedder, llm, s)
    docs = Path(s.docs_root)
    if store.count() == 0 and docs.exists():
        copilot.ingest_folder(docs)
    per_min = s.rate_limit_per_min or (20 if s.demo_mode else 0)
    cap = s.daily_cap or (300 if s.demo_mode else 0)
    return create_app(copilot, s.jwt_secret, str(docs), s.demo_roles if s.demo_mode else (), per_min, cap)
