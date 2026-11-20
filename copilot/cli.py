"""Command line: prepare a corpus, ingest it, ask questions, check your Gemini/Postgres setup, draft golden questions.

    python -m copilot.cli check
    python -m copilot.cli prepare --src ~/website/content/en/docs --dst data/corpus --restricted security
    python -m copilot.cli ingest --docs data/corpus
    python -m copilot.cli ask "How do I rotate certificates?" --role MEMBER
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

from .config import Settings
from .corpus import draft_golden, prepare_corpus
from .factory import build_components
from .pipeline import Copilot


def cmd_prepare(a) -> int:
    counts = prepare_corpus(a.src, a.dst, a.internal, a.restricted, a.limit)
    print(json.dumps(counts))
    return 0 if sum(v for k, v in counts.items() if k != "skipped_short") else 1


def _copilot(strategy_kw: dict | None = None):
    s = Settings()
    embedder, llm, store = build_components(s)
    return s, Copilot(store, embedder, llm, s), store


def cmd_ingest(a) -> int:
    s, cp, store = _copilot()
    if not s.database_url:
        print("WARNING: DATABASE_URL is not set, so the index lives in memory and is lost when this command ends.")
    kw = {"size": a.size, "overlap": a.overlap} if a.strategy == "fixed" else {}
    n = cp.ingest_folder(a.docs, a.strategy, **kw)
    print(f"ingested {n} chunks from {a.docs}; store now holds {store.count()} chunks (provider={s.provider})")
    return 0


def cmd_ask(a) -> int:
    s, cp, store = _copilot()
    if store.count() == 0:
        cp.ingest_folder(s.docs_root)  # memory store: index on the fly
    cp.mode = a.mode
    ans = cp.ask(a.question, a.role)
    if a.json:
        from dataclasses import asdict
        print(json.dumps(asdict(ans), indent=2, default=str))
        return 0
    print(f"\n{ans.text}\n")
    for c in ans.citations:
        print(f"  [{c['n']}] {c['source']}  ({c['heading']})  score={c['score']}")
    if ans.sql:
        print(f"  SQL: {ans.sql}")
    t = {k: round(v, 1) for k, v in ans.timings_ms.items()}
    print(f"\n  route={ans.route} timings_ms={t} tokens={ans.usage.prompt_tokens}+{ans.usage.completion_tokens} cost=${ans.cost_usd:.6f}")
    return 0


def cmd_check(a) -> int:
    """Verifies the pieces you configured, one by one, and says what to fix."""
    s, ok = Settings(), True
    print(f"provider={s.provider}  embed_model={s.embed_model}  chat_model={s.chat_model}")
    try:
        embedder, llm, store = build_components(s)
        vecs, _ = embedder.embed(["hello world"])
        norm = math.sqrt(sum(x * x for x in vecs[0]))
        print(f"OK  embeddings: dimension {len(vecs[0])}, norm {norm:.3f}")
        text, usage = llm.generate("Reply with the single word: ok")
        print(f"OK  generation: '{text[:40]}' ({usage.prompt_tokens}+{usage.completion_tokens} tokens)")
    except ModuleNotFoundError as e:
        ok = False
        print(f"FAIL missing package '{e.name}'. Install the cloud extras:  pip install -e \".[gcp]\"")
        store = None
    except Exception as e:
        ok = False
        print(f"FAIL model access: {e.__class__.__name__}: {e}")
        print("     Check GEMINI_API_KEY (or GOOGLE_CLOUD_PROJECT + 'gcloud auth application-default login'),"
              " and that EMBED_MODEL / CHAT_MODEL are model names your account can use.")
        store = None
    if s.database_url:
        try:
            if store is None:
                from .store import PgVectorStore
                store = PgVectorStore(s.database_url)
            print(f"OK  postgres + pgvector: {store.count()} chunks stored")
        except ModuleNotFoundError as e:
            ok = False
            print(f"FAIL missing package '{e.name}'. Install the cloud extras:  pip install -e \".[gcp]\"")
        except Exception as e:
            ok = False
            print(f"FAIL postgres: {e.__class__.__name__}: {e}")
    else:
        print("--  DATABASE_URL not set: using the in-memory store")
    return 0 if ok else 1


def cmd_draft_golden(a) -> int:
    s = Settings()
    _, llm, _ = build_components(s)
    rows = draft_golden(a.docs, llm, a.n)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} draft questions to {a.out}. Review every line before using it for metrics.")
    return 0 if rows else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="copilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("prepare", help="clean a folder of markdown docs and sort it into access folders")
    x.add_argument("--src", required=True); x.add_argument("--dst", required=True)
    x.add_argument("--internal", nargs="*", default=[], help="folder names that make a doc 'internal'")
    x.add_argument("--restricted", nargs="*", default=[], help="folder names that make a doc 'restricted'")
    x.add_argument("--limit", type=int); x.set_defaults(fn=cmd_prepare)
    x = sub.add_parser("ingest", help="chunk, embed and store a prepared corpus")
    x.add_argument("--docs", required=True); x.add_argument("--strategy", choices=["heading", "fixed"], default="heading")
    x.add_argument("--size", type=int, default=120); x.add_argument("--overlap", type=int, default=20)
    x.set_defaults(fn=cmd_ingest)
    x = sub.add_parser("ask", help="ask one question as a given role")
    x.add_argument("question"); x.add_argument("--role", default="MEMBER", choices=["ADMIN", "MANAGER", "MEMBER"])
    x.add_argument("--mode", default="hybrid+rerank", choices=["keyword", "vector", "hybrid", "hybrid+rerank"])
    x.add_argument("--json", action="store_true"); x.set_defaults(fn=cmd_ask)
    x = sub.add_parser("check", help="verify Gemini + Postgres configuration"); x.set_defaults(fn=cmd_check)
    x = sub.add_parser("draft-golden", help="have the LLM draft golden questions for you to review")
    x.add_argument("--docs", required=True); x.add_argument("--n", type=int, default=50)
    x.add_argument("--out", default="eval/golden.draft.jsonl"); x.set_defaults(fn=cmd_draft_golden)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
