import os

import pytest
from fastapi.testclient import TestClient

from copilot.api import RateLimiter, create_app
from copilot.auth import create_token
from copilot.main import build_app

DSN = os.environ.get("TEST_DATABASE_URL")


def auth(role, sub="u"):
    return {"Authorization": f"Bearer {create_token(sub, role, 'test-secret')}"}


# ---------- rate limiting and the daily cost cap ----------
def test_limiter_blocks_after_n_per_minute_and_isolates_clients():
    now = [0.0]
    rl = RateLimiter(per_minute=2, clock=lambda: now[0])
    rl.check("a"); rl.check("a")
    with pytest.raises(Exception, match="429|too many"):
        rl.check("a")
    rl.check("b")                      # another client is unaffected
    now[0] = 61.0
    rl.check("a")                      # the window slides


def test_daily_cap_resets_on_a_new_day():
    day = ["d1"]
    rl = RateLimiter(daily_cap=2, today=lambda: day[0])
    rl.check("a"); rl.check("b")
    with pytest.raises(Exception, match="quota"):
        rl.check("c")
    day[0] = "d2"
    rl.check("c")


def test_api_returns_429_and_uses_the_last_forwarded_address(copilot):
    c = TestClient(create_app(copilot, "test-secret", "sample_data/docs", rate_limit_per_min=2))
    ask = lambda ip: c.post("/ask", json={"question": "backups?"}, headers={**auth("MEMBER"), "X-Forwarded-For": ip})
    assert [ask("9.9.9.9, 1.1.1.1").status_code for _ in range(2)] == [200, 200]
    r = ask("6.6.6.6, 1.1.1.1")        # a spoofed first entry must not give a fresh allowance
    assert r.status_code == 429 and "Retry-After" in r.headers
    assert ask("6.6.6.6, 2.2.2.2").status_code == 200


def test_demo_tokens_cannot_trigger_reingestion(copilot):
    c = TestClient(create_app(copilot, "test-secret", "sample_data/docs", demo_roles=("ADMIN",)))
    tok = c.get("/demo/token", params={"role": "ADMIN"}).json()["token"]
    assert c.post("/ingest", headers={"Authorization": f"Bearer {tok}"}).status_code == 403
    assert c.post("/ingest", headers=auth("ADMIN", sub="real-admin")).status_code == 200


# ---------- demo mode must never run on real documents ----------
@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("PROVIDER", "offline")
    monkeypatch.setenv("SQL_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ALLOW_DEMO_ADMIN", raising=False)
    monkeypatch.setenv("DEMO_MODE", "1")
    return monkeypatch


def test_demo_with_privileged_roles_refuses_to_start_on_real_documents(env, tmp_path):
    (tmp_path / "docs" / "public").mkdir(parents=True)
    (tmp_path / "docs" / "public" / "a.md").write_text("# A\n" + "word " * 50)
    env.setenv("DOCS_ROOT", str(tmp_path / "docs"))
    env.setenv("DEMO_ROLES", "MEMBER,ADMIN")
    with pytest.raises(RuntimeError, match="synthetic sample data"):
        build_app()
    env.setenv("DEMO_ROLES", "MEMBER")
    assert TestClient(build_app()).get("/config").json()["demo_roles"] == ["MEMBER"]


def test_demo_with_all_roles_is_fine_on_the_bundled_sample_and_gets_default_limits(env):
    env.setenv("DOCS_ROOT", "sample_data/docs")
    env.setenv("DEMO_ROLES", "MEMBER,MANAGER,ADMIN")
    c = TestClient(build_app())
    assert c.get("/config").json()["demo_roles"] == ["MEMBER", "MANAGER", "ADMIN"]
    codes = [c.get("/demo/token", params={"role": "MEMBER"}).status_code for _ in range(22)]
    assert codes[:20] == [200] * 20 and codes[20] == 429      # default 20 per minute in demo mode


def test_weak_jwt_secret_is_refused_outside_offline_mode(env):
    env.setenv("PROVIDER", "gemini")
    env.setenv("DEMO_MODE", "0")
    env.setenv("JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        build_app()


# ---------- serverless Postgres drops idle connections ----------
@pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to run against a real Postgres with pgvector")
def test_store_reconnects_after_the_server_kills_its_connection():
    import psycopg
    from copilot.embeddings import HashEmbedder
    from copilot.llm import OfflineLLM
    from copilot.pipeline import Copilot
    from copilot.config import Settings
    from copilot.store import PgVectorStore
    store = PgVectorStore(DSN, dim=HashEmbedder().dim, table="copilot_test_reconnect")
    try:
        cp = Copilot(store, HashEmbedder(), OfflineLLM(), Settings())
        cp.ingest_folder("sample_data/docs")
        with psycopg.connect(DSN, autocommit=True) as admin:           # what an idle timeout or restart does
            admin.execute("SELECT pg_terminate_backend(%s)", (store.conn.info.backend_pid,))
        assert store.count() == 23                                      # reconnected transparently
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (store.conn.info.backend_pid,))
        assert "6 hours" in cp.ask("How often are backups taken?", "MEMBER").text   # search path too
    finally:
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute("DROP TABLE IF EXISTS copilot_test_reconnect")


def test_ask_failure_is_readable_not_bare_500():
    from fastapi.testclient import TestClient
    from copilot.api import create_app
    from copilot.auth import create_token as issue_token

    class Boom:
        store = type("S", (), {"count": lambda self: 0})()
        def ask(self, q, role): raise RuntimeError("gemini exploded")
    app = create_app(Boom(), "x" * 40, ".", demo_roles=())
    tok = issue_token("u", "MEMBER", "x" * 40)
    r = TestClient(app, raise_server_exceptions=False).post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 502 and "gemini exploded" in r.json()["detail"]


class _Err(Exception):
    def __init__(self, code): super().__init__(f"{code} UNAVAILABLE"); self.code = code


def _fake_client(behaviour):
    calls = []
    class M:
        def generate_content(self, model, contents, config):
            calls.append(model)
            r = behaviour(model)
            if isinstance(r, Exception): raise r
            return type("R", (), {"text": r, "usage_metadata": type("U", (), {"prompt_token_count": 1, "candidate_token_count": 1, "candidates_token_count": 1})()})()
    return type("C", (), {"models": M()})(), calls


def test_overloaded_model_retries_then_falls_back():
    from copilot.llm import GeminiLLM
    client, calls = _fake_client(lambda m: _Err(503) if m == "big" else "answer from small")
    llm = GeminiLLM(client, "big", ("small",), sleep=lambda s: None)
    text, _ = llm.generate("q")
    assert text == "answer from small" and calls == ["big", "big", "small"] and llm.name == "small"


def test_real_errors_are_not_hidden_by_fallback():
    import pytest
    from copilot.llm import GeminiLLM
    client, calls = _fake_client(lambda m: _Err(404))
    with pytest.raises(_Err):
        GeminiLLM(client, "big", ("small",), sleep=lambda s: None).generate("q")
    assert calls == ["big"]


def test_all_models_overloaded_raises():
    import pytest
    from copilot.llm import GeminiLLM
    client, calls = _fake_client(lambda m: _Err(503))
    with pytest.raises(_Err):
        GeminiLLM(client, "a", ("b",), sleep=lambda s: None).generate("q")
    assert calls == ["a", "a", "b", "b"]
