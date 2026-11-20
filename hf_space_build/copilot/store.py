"""Chunk stores. Access control is applied INSIDE the query (never after ranking),
so a chunk a user may not read can neither be returned nor influence ranking."""
from __future__ import annotations
import math
import threading
import re
from collections import Counter
from typing import Protocol

import numpy as np

from .embeddings import STOP, tokenize
from .models import Chunk, Hit


class Store(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def vector_search(self, qvec: list[float], k: int, allowed: frozenset[str]) -> list[Hit]: ...
    def keyword_search(self, query: str, k: int, allowed: frozenset[str]) -> list[Hit]: ...
    def count(self) -> int: ...


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(chunks)) if chunks else 0.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int) -> list[Hit]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
            if s > 0:
                scored.append(Hit(self.chunks[i], s))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


class MemoryStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._bm25: dict[frozenset[str], BM25] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"chunk {c.id} has no embedding")
        by_id = {c.id: i for i, c in enumerate(self.chunks)}
        for c in chunks:  # upsert, same as the pgvector store
            if c.id in by_id:
                self.chunks[by_id[c.id]] = c
            else:
                by_id[c.id] = len(self.chunks)
                self.chunks.append(c)
        self._matrix, self._bm25 = None, {}

    def count(self) -> int:
        return len(self.chunks)

    def vector_search(self, qvec, k, allowed):
        idx = [i for i, c in enumerate(self.chunks) if c.acl in allowed]
        if not idx:
            return []
        if self._matrix is None:
            self._matrix = np.array([c.embedding for c in self.chunks], dtype=np.float32)
        sims = self._matrix[idx] @ np.array(qvec, dtype=np.float32)
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[idx[j]], float(sims[j])) for j in order]

    def keyword_search(self, query, k, allowed):
        if allowed not in self._bm25:
            self._bm25[allowed] = BM25([c for c in self.chunks if c.acl in allowed])
        return self._bm25[allowed].search(query, k)


class PgVectorStore:
    """PostgreSQL + pgvector (Cloud SQL / AlloyDB / Neon / Supabase). Vector search uses cosine distance with an HNSW
    index; keyword search uses Postgres full-text search. Serverless Postgres closes idle connections, so every
    statement transparently reconnects once if the connection has been dropped."""

    def __init__(self, dsn: str, dim: int = 768, table: str = "chunks") -> None:
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", table):
            raise ValueError("table must be lowercase letters, digits and underscores")
        self.dsn, self.dim, self.table = dsn, dim, table
        self._lock = threading.Lock()
        self.conn = None
        self._connect()
        self._run(self._create_schema)

    def _connect(self) -> None:
        import psycopg
        with self._lock:
            if self.conn is not None and not self.conn.closed:
                return
            self.conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=10)

    def _run(self, fn):
        """Run fn(cursor); if the connection was dropped (idle timeout, restart), reconnect and retry once."""
        import psycopg
        for attempt in (1, 2):
            try:
                self._connect()
                with self.conn.cursor() as cur:
                    return fn(cur)
            except (psycopg.OperationalError, psycopg.InterfaceError):
                if attempt == 2:
                    raise
                try:
                    self.conn.close()
                except Exception:
                    pass

    def _create_schema(self, cur) -> None:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {self.table} (
            id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, source TEXT NOT NULL, heading TEXT NOT NULL DEFAULT '',
            acl TEXT NOT NULL, text TEXT NOT NULL, embedding vector({self.dim}) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)""")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_vec ON {self.table} USING hnsw (embedding vector_cosine_ops)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_tsv ON {self.table} USING gin (tsv)")

    @staticmethod
    def _vec(v: list[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

    def add(self, chunks: list[Chunk]) -> None:
        rows = [(c.id, c.doc_id, c.source, c.heading, c.acl, c.text, self._vec(c.embedding)) for c in chunks]
        self._run(lambda cur: cur.executemany(
            f"""INSERT INTO {self.table} (id, doc_id, source, heading, acl, text, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s::vector)
                ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, acl=EXCLUDED.acl, embedding=EXCLUDED.embedding""", rows))

    def count(self) -> int:
        def q(cur):
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return cur.fetchone()[0]
        return self._run(q)

    def _rows(self, sql: str, params: tuple) -> list[Hit]:
        def q(cur):
            cur.execute(sql, params)
            return cur.fetchall()
        return [Hit(Chunk(id=r[0], doc_id=r[1], source=r[2], heading=r[3], acl=r[4], text=r[5]), float(r[6]))
                for r in self._run(q)]

    def vector_search(self, qvec, k, allowed):
        v = self._vec(qvec)
        return self._rows(
            f"""SELECT id, doc_id, source, heading, acl, text, 1 - (embedding <=> %s::vector) AS score
                FROM {self.table} WHERE acl = ANY(%s) ORDER BY embedding <=> %s::vector LIMIT %s""",
            (v, list(allowed), v, k))

    def keyword_search(self, query, k, allowed):
        # OR semantics, like BM25: a natural-language question rarely contains only words that all appear in one
        # chunk. (websearch_to_tsquery would AND every word and return nothing.) Tokens are [a-z0-9]+ only,
        # so the query string cannot carry tsquery operators.
        terms = list(dict.fromkeys(t for t in tokenize(query) if t not in STOP))
        if not terms:
            return []
        tsq = " | ".join(terms)
        return self._rows(
            f"""SELECT id, doc_id, source, heading, acl, text, ts_rank_cd(tsv, to_tsquery('english', %s)) AS score
                FROM {self.table} WHERE acl = ANY(%s) AND tsv @@ to_tsquery('english', %s)
                ORDER BY score DESC LIMIT %s""",
            (tsq, list(allowed), tsq, k))
