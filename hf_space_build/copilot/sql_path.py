"""Guarded text-to-SQL. The model may write SQL, but nothing it writes runs until it passes the guard,
and it runs on a read-only connection. Defence in depth: validate, restrict tables, force LIMIT, read-only DB."""
from __future__ import annotations
import re
import sqlite3

import sqlglot
from sqlglot import exp

MAX_ROWS = 100
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
             exp.Attach, exp.Pragma, exp.Merge, exp.TruncateTable, exp.Transaction)


class SqlRejected(ValueError):
    """The generated SQL failed a safety check."""


def validate(sql: str, allowed_tables: set[str]) -> str:
    allowed = {t.lower() for t in allowed_tables}
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SqlRejected(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise SqlRejected("exactly one SQL statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise SqlRejected("only SELECT statements are allowed")
    if any(tree.find(f) for f in FORBIDDEN):
        raise SqlRejected("statement contains a forbidden operation")
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name not in allowed and name not in ctes:
            raise SqlRejected(f"table '{t.name}' is not allowed")
    limit = tree.args.get("limit")
    if limit is None or not isinstance(limit.expression, exp.Literal) or int(limit.expression.name) > MAX_ROWS:
        tree = tree.limit(MAX_ROWS)
    return tree.sql(dialect="sqlite", comments=False)


def clean_model_sql(raw: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.S | re.I)
    return (m.group(1) if m else raw).strip().rstrip(";")


def schema_text(db_path: str, allowed_tables: set[str]) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        lines = []
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            if name.lower() in {t.lower() for t in allowed_tables}:
                cols = ", ".join(f"{c[1]} {c[2]}" for c in con.execute(f"PRAGMA table_info({name})"))
                lines.append(f"{name}({cols})")
        return "\n".join(lines)
    finally:
        con.close()


def run_readonly(db_path: str, sql: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = ON")
        cur = con.execute(sql)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    finally:
        con.close()


def describe_rows(rows: list[dict]) -> str:
    """Answer text built only from the query result, so numbers cannot be hallucinated."""
    if not rows:
        return "The query returned no rows."
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"Result: {next(iter(rows[0].values()))}"
    head = ", ".join(rows[0].keys())
    body = "; ".join(", ".join(str(v) for v in r.values()) for r in rows[:10])
    more = f" (showing 10 of {len(rows)})" if len(rows) > 10 else ""
    return f"Columns: {head}. Rows: {body}{more}"
