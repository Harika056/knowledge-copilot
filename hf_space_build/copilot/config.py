from __future__ import annotations
import os
from dataclasses import dataclass, field

# Which document-access labels each role may read. Mirrors TaskFlow's roles.
ROLE_ACCESS: dict[str, frozenset[str]] = {
    "ADMIN": frozenset({"public", "internal", "restricted"}),
    "MANAGER": frozenset({"public", "internal"}),
    "MEMBER": frozenset({"public"}),
}


@dataclass
class Settings:
    jwt_secret: str = field(default_factory=lambda: os.environ.get("JWT_SECRET") or "dev-only-change-me")
    provider: str = field(default_factory=lambda: os.environ.get("PROVIDER", "offline"))  # offline | gemini
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    gcp_project: str | None = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT"))
    gcp_location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "gemini-embedding-001"))
    chat_model: str = field(default_factory=lambda: os.environ.get("CHAT_MODEL", "gemini-2.5-flash"))
    database_url: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URL"))  # pgvector; None = in-memory
    # Hits scoring below this are dropped; if none remain the answer is "not found" (no LLM call). 0 = off.
    # The score scale depends on the retrieval mode (rerank ~0-2, RRF ~0.01-0.03, cosine 0-1): tune on your golden set.
    min_score: float = field(default_factory=lambda: float(os.environ.get("MIN_SCORE") or 0))
    # DEMO_MODE=1 exposes /demo/token so the web UI can switch roles without a login. LOCAL USE ONLY unless
    # DEMO_ROLES is limited (for a public demo use DEMO_ROLES=MEMBER): anyone who can open the page gets those roles.
    demo_mode: bool = field(default_factory=lambda: os.environ.get("DEMO_MODE") in ("1", "true", "yes"))
    demo_roles: tuple = field(default_factory=lambda: tuple(r.strip().upper() for r in (os.environ.get("DEMO_ROLES") or "MEMBER,MANAGER,ADMIN").split(",") if r.strip()))
    # Abuse brakes for a public demo (per instance). 0 = off. In DEMO_MODE they default to 20/min per IP and 300/day.
    rate_limit_per_min: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_PER_MIN") or 0))
    daily_cap: int = field(default_factory=lambda: int(os.environ.get("DAILY_CAP") or 0))
    docs_root: str = field(default_factory=lambda: os.environ.get("DOCS_ROOT") or "sample_data/docs")
    sql_db_path: str = field(default_factory=lambda: os.environ.get("SQL_DB_PATH", "sample_data/sales.db"))
    # USD per 1M tokens, used only for the cost estimate. Check current Google pricing and set these.
    price_in_per_m: float = field(default_factory=lambda: float(os.environ.get("PRICE_IN_PER_M", "0")))
    price_out_per_m: float = field(default_factory=lambda: float(os.environ.get("PRICE_OUT_PER_M", "0")))
