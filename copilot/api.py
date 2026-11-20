from __future__ import annotations
import logging
from dataclasses import asdict
from pathlib import Path

import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .auth import AuthError, create_token, verify_token
from .config import ROLE_ACCESS
from .pipeline import Copilot


class RateLimiter:
    """Per-client sliding-window limit plus a daily cap on total requests (a cost brake for public demos).
    In-memory and per instance: a basic abuse brake, not a security boundary. Pair it with Cloud Run
    --max-instances and a billing budget alert."""

    def __init__(self, per_minute: int = 0, daily_cap: int = 0, clock=time.monotonic, today=lambda: datetime.now(timezone.utc).date()) -> None:
        self.per_minute, self.daily_cap, self.clock, self.today = per_minute, daily_cap, clock, today
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.count = today(), 0

    def check(self, client: str) -> None:
        if self.today() != self.day:
            self.day, self.count = self.today(), 0
        if self.daily_cap and self.count >= self.daily_cap:
            raise HTTPException(429, "daily demo quota reached, try again tomorrow", headers={"Retry-After": "3600"})
        if self.per_minute:
            now, q = self.clock(), self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, "too many requests, slow down", headers={"Retry-After": "30"})
            q.append(now)
        self.count += 1


def client_id(request: Request) -> str:
    """Cloud Run appends the real client address to X-Forwarded-For, so the LAST entry is the one to trust."""
    xff = request.headers.get("x-forwarded-for", "")
    return xff.split(",")[-1].strip() if xff else (request.client.host if request.client else "unknown")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


log = logging.getLogger("copilot")


def create_app(copilot: Copilot, secret: str, docs_root: str, demo_roles: tuple = (), rate_limit_per_min: int = 0, daily_cap: int = 0) -> FastAPI:
    """demo_roles: roles the web UI may request a token for via /demo/token. Empty (default) = endpoint disabled."""
    app = FastAPI(title="Knowledge Copilot", version="0.1.0")
    limiter = RateLimiter(rate_limit_per_min, daily_cap)

    def claims(authorization: str | None) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            return verify_token(authorization[7:], secret)
        except AuthError as e:
            raise HTTPException(e.status, str(e))

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/config")
    def config():
        return {"demo_roles": list(demo_roles), "provider": copilot.llm.name, "embedder": copilot.embedder.name,
                "chunks": copilot.store.count()}

    @app.get("/demo/token")
    def demo_token(role: str, request: Request):
        limiter.check(client_id(request))
        role = role.upper()
        if role not in demo_roles or role not in ROLE_ACCESS:
            raise HTTPException(404, "demo login is disabled for this role")
        return {"token": create_token(f"demo-{role.lower()}", role, secret, ttl_s=3600), "role": role}

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": copilot.store.count()}

    @app.post("/ask")
    def ask(req: AskRequest, request: Request, authorization: str | None = Header(default=None)):
        c = claims(authorization)
        limiter.check(client_id(request))
        try:
            return asdict(copilot.ask(req.question, c["role"]))
        except HTTPException:
            raise
        except Exception as e:  # show a readable reason in the UI and the logs instead of a bare "HTTP 500"
            log.exception("ask failed")
            raise HTTPException(502, f"The answer step failed: {e.__class__.__name__}: {str(e)[:300]}")

    @app.post("/ingest")
    def ingest(authorization: str | None = Header(default=None)):
        c = claims(authorization)
        if c["role"] != "ADMIN" or str(c.get("sub", "")).startswith("demo-"):
            raise HTTPException(403, "admin only (demo logins cannot re-ingest)")
        return {"chunks_added": copilot.ingest_folder(docs_root)}

    return app
