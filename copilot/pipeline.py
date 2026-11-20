from __future__ import annotations
import re
import time
from pathlib import Path

from . import sql_path
from .chunking import make_chunks
from .config import ROLE_ACCESS, Settings
from .models import NOT_FOUND, Answer, Usage
from .retrieval import Retriever
from .router import route
from .store import Store

SYSTEM = ("You answer questions using ONLY the numbered passages. Cite passages like [1]. "
          "If the passages do not contain the answer, say you could not find it. Do not use outside knowledge.")
SQL_ROLES = frozenset({"ADMIN", "MANAGER"})
SQL_TABLES = {"orders", "customers"}


class Copilot:
    def __init__(self, store: Store, embedder, llm, settings: Settings | None = None,
                 mode: str = "hybrid+rerank", k: int = 4, reranker=None) -> None:
        self.store, self.embedder, self.llm = store, embedder, llm
        self.settings, self.mode, self.k = settings or Settings(), mode, k
        self.retriever = Retriever(store, embedder, reranker)

    # ---- ingestion ----
    def ingest_folder(self, root: str | Path, strategy: str = "heading", **chunk_kw) -> int:
        """Folder layout decides access: <root>/<label>/**/*.md, label in public|internal|restricted."""
        root, chunks = Path(root), []
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(root)
            label = rel.parts[0] if len(rel.parts) > 1 else ""
            if not any(label in labels for labels in ROLE_ACCESS.values()):
                raise ValueError(f"{rel}: put documents under a folder named public, internal or restricted")
            chunks += make_chunks(str(rel), path.read_text(encoding="utf-8"), label, str(rel), strategy, **chunk_kw)
        if chunks:
            vecs, _ = self.embedder.embed([c.text for c in chunks])
            for c, v in zip(chunks, vecs):
                c.embedding = v
            self.store.add(chunks)
        return len(chunks)

    # ---- answering ----
    def _cost(self, u: Usage) -> float:
        s = self.settings
        return (u.prompt_tokens * s.price_in_per_m + u.completion_tokens * s.price_out_per_m) / 1e6

    def ask(self, question: str, role: str) -> Answer:
        if role not in ROLE_ACCESS:
            raise ValueError(f"unknown role '{role}'")
        t0 = time.perf_counter()
        ans = self._ask_sql(question, role) if route(question) == "sql" else self._ask_docs(question, role)
        ans.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        ans.cost_usd = self._cost(ans.usage)
        return ans

    def _ask_docs(self, question: str, role: str) -> Answer:
        res = self.retriever.search(question, ROLE_ACCESS[role], self.mode, self.k)
        ans = Answer(text=NOT_FOUND, route="docs", timings_ms=dict(res.timings_ms), usage=res.usage)
        res.hits = [h for h in res.hits if h.score >= self.settings.min_score]
        if not res.hits:
            return ans  # no LLM call: nothing to ground an answer on
        passages = "\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(res.hits, 1))
        t = time.perf_counter()
        text, usage = self.llm.generate(f"Passages:\n{passages}\n\nQuestion: {question}", system=SYSTEM)
        ans.timings_ms["generate"] = (time.perf_counter() - t) * 1000
        ans.text, ans.usage = text, res.usage
        ans.usage.add(usage)
        # Show only the passages the answer actually cites ([n] markers), keeping their original numbers.
        # A "could not find" answer therefore has no sources, instead of unrelated passages that merely ranked highest.
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        ans.citations = [{"n": i, "source": h.chunk.source, "heading": h.chunk.heading,
                          "acl": h.chunk.acl, "snippet": h.chunk.text[:160], "score": round(h.score, 4)}
                         for i, h in enumerate(res.hits, 1) if i in cited]
        return ans

    def _ask_sql(self, question: str, role: str) -> Answer:
        ans = Answer(text="", route="sql")
        if role not in SQL_ROLES:
            ans.text = "Your role is not allowed to query business data."
            return ans
        db = self.settings.sql_db_path
        prompt = (f"Schema (SQLite):\n{sql_path.schema_text(db, SQL_TABLES)}\n\n"
                  f"Write ONE read-only SQLite SELECT query that answers the question. Return only the SQL.\n"
                  f"Question: {question}")
        t = time.perf_counter()
        raw, usage = self.llm.generate(prompt)
        ans.timings_ms["generate_sql"], ans.usage = (time.perf_counter() - t) * 1000, usage
        try:
            ans.sql = sql_path.validate(sql_path.clean_model_sql(raw), SQL_TABLES)
            t = time.perf_counter()
            ans.rows = sql_path.run_readonly(db, ans.sql)
            ans.timings_ms["run_sql"] = (time.perf_counter() - t) * 1000
            ans.text = sql_path.describe_rows(ans.rows)
        except sql_path.SqlRejected as e:
            ans.sql, ans.text = None, f"I could not safely run a query for that: {e}."
        except Exception as e:  # database errors: report, never crash the API
            ans.text = f"The query failed to run ({e.__class__.__name__})."
        return ans
