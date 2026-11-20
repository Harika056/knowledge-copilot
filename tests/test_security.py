import sqlite3
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from copilot import sql_path
from copilot.api import create_app
from copilot.auth import AuthError, create_token, verify_token
from copilot.config import ROLE_ACCESS
from copilot.llm import ScriptedLLM
from copilot.pipeline import NOT_FOUND, Copilot
from copilot.router import route
from copilot.store import MemoryStore
from copilot.embeddings import HashEmbedder

T = {"orders", "customers"}


# ---------- document access control ----------
def test_member_never_receives_restricted_content(copilot):
    ans = copilot.ask("What is the pay range for an L4 engineer salary band?", "MEMBER")
    blob = ans.text + " ".join(c["snippet"] for c in ans.citations)
    assert "28 to 40" not in blob and "lakh" not in blob
    assert all(c["source"].startswith("public/") for c in ans.citations)


def test_manager_sees_internal_but_not_restricted(copilot):
    ans = copilot.ask("How long do we have to write a postmortem after a Sev1?", "MANAGER")
    assert ans.citations[0]["source"] == "internal/incident-process.md"
    assert not any(c["source"].startswith("restricted/") for c in copilot.ask("salary band", "MANAGER").citations)


def test_admin_can_reach_restricted(copilot):
    ans = copilot.ask("What is the approved budget ceiling for the Nimbus Data acquisition?", "ADMIN")
    assert ans.citations[0]["source"] == "restricted/acquisition-plan.md"


def test_empty_store_returns_not_found_without_calling_llm(settings):
    llm = ScriptedLLM([])
    cp = Copilot(MemoryStore(), HashEmbedder(), llm, settings)
    assert cp.ask("anything about pricing", "ADMIN").text == NOT_FOUND and llm.prompts == []


def test_ingest_rejects_documents_outside_access_folders(tmp_path, settings):
    (tmp_path / "loose.md").write_text("# secret\nhello")
    with pytest.raises(ValueError, match="public, internal or restricted"):
        Copilot(MemoryStore(), HashEmbedder(), ScriptedLLM([]), settings).ingest_folder(tmp_path)


# ---------- SQL guard ----------
def test_valid_select_passes_and_gets_a_limit():
    out = sql_path.validate("SELECT region, COUNT(*) FROM customers GROUP BY region", T)
    assert "LIMIT 100" in out


def test_large_limit_is_clamped():
    assert "LIMIT 100" in sql_path.validate("SELECT * FROM orders LIMIT 100000", T)


def test_small_limit_is_kept():
    assert "LIMIT 5" in sql_path.validate("SELECT * FROM orders LIMIT 5", T)


@pytest.mark.parametrize("bad", [
    "DROP TABLE orders",
    "DELETE FROM orders",
    "UPDATE orders SET amount = 0",
    "INSERT INTO orders VALUES (1,1,'x',1,'PAID')",
    "SELECT 1; DROP TABLE orders",
    "SELECT * FROM secrets",
    "SELECT * FROM orders UNION SELECT id, note, 1, 1, 1 FROM secrets",
    "WITH s AS (SELECT * FROM secrets) SELECT * FROM s",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
])
def test_dangerous_sql_is_rejected(bad):
    with pytest.raises(sql_path.SqlRejected):
        sql_path.validate(bad, T)


def test_comment_trick_does_not_smuggle_a_second_statement():
    out = sql_path.validate("SELECT COUNT(*) FROM orders -- ; DROP TABLE orders", T)
    assert "DROP" not in out.upper()


def test_database_connection_is_read_only_even_if_guard_is_bypassed(settings):
    with pytest.raises(sqlite3.OperationalError):
        sql_path.run_readonly(settings.sql_db_path, "DELETE FROM orders")


# ---------- routing + SQL path ----------
@pytest.mark.parametrize("q,expected", [
    ("How many orders were placed in March?", "sql"),
    ("What is the total revenue by region?", "sql"),
    ("What is the pay range for an L4 engineer?", "docs"),
    ("How often are backups taken?", "docs"),
])
def test_router(q, expected):
    assert route(q) == expected


def _sql_copilot(settings, reply):
    return Copilot(MemoryStore(), HashEmbedder(), ScriptedLLM([reply]), settings)


def test_manager_gets_grounded_sql_answer(settings):
    ans = _sql_copilot(settings, "```sql\nSELECT COUNT(*) AS n FROM orders;\n```").ask("How many orders are there?", "MANAGER")
    assert ans.route == "sql" and ans.rows == [{"n": 200}] and ans.text == "Result: 200"


def test_member_cannot_query_business_data(settings):
    cp = _sql_copilot(settings, "SELECT COUNT(*) FROM orders")
    assert "not allowed" in cp.ask("How many orders are there?", "MEMBER").text


def test_model_written_drop_is_blocked_and_data_survives(settings):
    ans = _sql_copilot(settings, "DROP TABLE orders").ask("How many orders are there?", "ADMIN")
    assert ans.sql is None and "could not safely run" in ans.text
    assert sql_path.run_readonly(settings.sql_db_path, "SELECT COUNT(*) AS n FROM orders") == [{"n": 200}]


# ---------- auth + API ----------
def test_token_roundtrip_and_bad_secret():
    tok = create_token("u1", "MANAGER", "s")
    assert verify_token(tok, "s")["role"] == "MANAGER"
    with pytest.raises(AuthError):
        verify_token(tok, "other")


def test_expired_and_unknown_role_tokens():
    with pytest.raises(AuthError, match="expired"):
        verify_token(jwt.encode({"sub": "u", "role": "ADMIN", "exp": int(time.time()) - 5}, "s", "HS256"), "s")
    with pytest.raises(AuthError) as e:
        verify_token(create_token("u", "HACKER", "s"), "s")
    assert e.value.status == 403


@pytest.fixture()
def client(copilot):
    return TestClient(create_app(copilot, "test-secret", "sample_data/docs"))


def auth(role):
    return {"Authorization": f"Bearer {create_token('u', role, 'test-secret')}"}


def test_api_requires_a_token(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 401
    assert client.post("/ask", json={"question": "hi"}, headers={"Authorization": "Bearer junk"}).status_code == 401


def test_api_validates_input(client):
    assert client.post("/ask", json={"question": ""}, headers=auth("MEMBER")).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1001}, headers=auth("MEMBER")).status_code == 422


def test_api_answers_with_citations_and_hides_restricted_from_members(client):
    r = client.post("/ask", json={"question": "How much is the Business plan per month?"}, headers=auth("MEMBER"))
    assert r.status_code == 200 and r.json()["citations"][0]["source"] == "public/pricing.md"
    r = client.post("/ask", json={"question": "L4 salary band range"}, headers=auth("MEMBER"))
    assert "lakh" not in r.text


def test_ingest_is_admin_only_and_idempotent(client):
    before = client.get("/health").json()["chunks"]
    assert client.post("/ingest", headers=auth("MANAGER")).status_code == 403
    assert client.post("/ingest", headers=auth("ADMIN")).status_code == 200
    assert client.get("/health").json()["chunks"] == before  # upsert, no duplicates


def test_empty_jwt_secret_env_var_falls_back_to_the_guarded_default(monkeypatch):
    from copilot.config import Settings
    monkeypatch.setenv("JWT_SECRET", "")
    assert Settings().jwt_secret == "dev-only-change-me"  # never an empty signing key


def test_relevance_cutoff_turns_weak_matches_into_not_found_without_calling_the_llm(settings):
    from dataclasses import replace
    llm = ScriptedLLM([])
    cp = Copilot(MemoryStore(), HashEmbedder(), llm, replace(settings, min_score=50.0))
    cp.ingest_folder("sample_data/docs")
    assert cp.ask("How often are backups taken?", "ADMIN").text == NOT_FOUND and llm.prompts == []


# ---------- web UI + demo mode ----------
def test_ui_page_is_served_and_has_no_inline_html_injection(copilot):
    html = TestClient(create_app(copilot, "test-secret", "sample_data/docs")).get("/").text
    assert "Knowledge Copilot" in html and "innerHTML" not in html  # untrusted text is only ever set via textContent


def test_demo_token_endpoint_is_off_by_default(copilot):
    c = TestClient(create_app(copilot, "test-secret", "sample_data/docs"))
    assert c.get("/demo/token", params={"role": "ADMIN"}).status_code == 404
    assert c.get("/config").json()["demo_roles"] == []


def test_demo_token_only_for_whitelisted_roles_and_tokens_really_work(copilot):
    c = TestClient(create_app(copilot, "test-secret", "sample_data/docs", demo_roles=("MEMBER",)))
    assert c.get("/demo/token", params={"role": "ADMIN"}).status_code == 404      # not whitelisted
    assert c.get("/demo/token", params={"role": "HACKER"}).status_code == 404     # not a role
    tok = c.get("/demo/token", params={"role": "member"}).json()["token"]
    r = c.post("/ask", json={"question": "How often are backups taken?"}, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["citations"][0]["acl"] == "public"
    assert verify_token(tok, "test-secret")["role"] == "MEMBER"


def test_sources_list_only_what_the_answer_cites(copilot):
    ok = copilot.ask("How often are backups taken?", "MEMBER")
    assert ok.text.endswith("[1]") and [c["n"] for c in ok.citations] == [1]
    none = copilot.ask("What is the L4 engineer salary band?", "MEMBER")   # restricted: nothing to cite
    assert none.text == NOT_FOUND and none.citations == []


def test_citation_numbers_stay_aligned_with_the_answer_text(settings):
    cp = Copilot(MemoryStore(), HashEmbedder(), ScriptedLLM(["Backups run every 6 hours [2]."]), settings)
    cp.ingest_folder("sample_data/docs")
    ans = cp.ask("How often are backups taken?", "ADMIN")
    assert [c["n"] for c in ans.citations] == [2]
