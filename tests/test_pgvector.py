"""Integration tests against a REAL PostgreSQL + pgvector. Skipped unless TEST_DATABASE_URL is set:

    TEST_DATABASE_URL=postgresql://copilot:copilot@127.0.0.1:5432/copilot pytest tests/test_pgvector.py
"""
import os

import pytest

from copilot.embeddings import HashEmbedder
from copilot.evaluation import evaluate_retrieval, load_golden
from copilot.llm import OfflineLLM
from copilot.pipeline import Copilot
from copilot.retrieval import MODES
from copilot.store import MemoryStore, PgVectorStore

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to run against a real Postgres with pgvector")
DOCS = "sample_data/docs"


@pytest.fixture()
def pg():
    # A table of its own, so these tests never touch a real "chunks" table in the same database.
    store = PgVectorStore(DSN, dim=HashEmbedder().dim, table="copilot_test_chunks")
    with store.conn.cursor() as cur:
        cur.execute("TRUNCATE copilot_test_chunks")
    yield store
    with store.conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS copilot_test_chunks")
    store.conn.close()


@pytest.fixture()
def pg_copilot(pg, settings):
    cp = Copilot(pg, HashEmbedder(), OfflineLLM(), settings)
    cp.ingest_folder(DOCS)
    return cp


def test_ingest_is_idempotent(pg_copilot):
    n = pg_copilot.store.count()
    assert n == 23
    pg_copilot.ingest_folder(DOCS)
    assert pg_copilot.store.count() == n


def test_acl_is_enforced_inside_both_queries(pg_copilot):
    s, emb = pg_copilot.store, HashEmbedder()
    q = "What is the L4 engineer salary band?"
    member = frozenset({"public"})
    assert all(h.chunk.acl == "public" for h in s.vector_search(emb.embed([q])[0][0], 20, member))
    assert all(h.chunk.acl == "public" for h in s.keyword_search(q, 20, member))
    admin = pg_copilot.ask(q, "ADMIN")
    assert admin.citations[0]["source"] == "restricted/salary-bands.md"
    assert "lakh" not in pg_copilot.ask(q, "MEMBER").text


def test_natural_language_question_matches_keywords_with_or_semantics(pg_copilot):
    # Words like "tier" and "each" never appear in the pricing doc; a strict AND query would return nothing.
    hits = pg_copilot.store.keyword_search("How much is the Business tier each month?", 5, frozenset({"public"}))
    assert hits and hits[0].chunk.source == "public/pricing.md"


def test_special_characters_cannot_break_the_keyword_query(pg_copilot):
    for q in ["'; DROP TABLE chunks; --", "business & | ! ( ) :*", "   ", "日本語"]:
        pg_copilot.store.keyword_search(q, 5, frozenset({"public"}))  # must not raise
    assert pg_copilot.store.count() == 23


def test_vector_scores_are_cosine_similarity(pg_copilot):
    emb = HashEmbedder()
    text = "The Business plan costs $99 per month and includes 25 users, 200 GB of storage and priority email support."
    top = pg_copilot.store.vector_search(emb.embed([text])[0][0], 1, frozenset({"public"}))[0]
    assert top.chunk.source == "public/pricing.md" and 0.0 < top.score <= 1.0001


@pytest.mark.parametrize("mode", MODES)
def test_postgres_matches_memory_store_on_the_golden_set(pg_copilot, settings, mode):
    mem = Copilot(MemoryStore(), HashEmbedder(), OfflineLLM(), settings)
    mem.ingest_folder(DOCS)
    golden = load_golden("eval/golden.jsonl")
    a = evaluate_retrieval(pg_copilot, golden, mode)
    b = evaluate_retrieval(mem, golden, mode)
    # Same data, same embeddings: hit@4 should agree closely (FTS stemming and BM25 differ slightly).
    assert abs(a["hit@4"] - b["hit@4"]) <= 0.15, (a, b)
