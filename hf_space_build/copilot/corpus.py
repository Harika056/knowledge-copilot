"""Turn a folder of real markdown docs (for example a cloned docs repository) into the layout the copilot
expects: <dst>/<public|internal|restricted>/<same relative path>.md, cleaned of front matter and shortcodes."""
from __future__ import annotations
import random
import re
from pathlib import Path

from .chunking import make_chunks

FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
MIN_WORDS = 40


def clean_markdown(text: str) -> str:
    title = None
    m = FRONT.match(text)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip("\"'") if t else None
        text = text[m.end():]
    text = SHORTCODE.sub("", HTML_COMMENT.sub("", text))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if title and not re.match(r"#\s", text):
        text = f"# {title}\n\n{text}"
    return text + "\n"


def label_for(rel: Path, internal: tuple[str, ...], restricted: tuple[str, ...]) -> str:
    parts = set(rel.parts)
    if parts & set(restricted):
        return "restricted"
    if parts & set(internal):
        return "internal"
    return "public"


def prepare_corpus(src: str | Path, dst: str | Path, internal=(), restricted=(), limit: int | None = None) -> dict:
    src, dst = Path(src), Path(dst)
    counts = {"public": 0, "internal": 0, "restricted": 0, "skipped_short": 0}
    for path in sorted(src.rglob("*.md")):
        if limit is not None and sum(v for k, v in counts.items() if k != "skipped_short") >= limit:
            break
        rel = path.relative_to(src)
        text = clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
        if len(text.split()) < MIN_WORDS:
            counts["skipped_short"] += 1
            continue
        label = label_for(rel, tuple(internal), tuple(restricted))
        out = dst / label / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        counts[label] += 1
    return counts


def draft_golden(docs_root: str | Path, llm, n: int = 50, seed: int = 1, strategy: str = "heading") -> list[dict]:
    """Ask the LLM to write one question per sampled chunk. A human MUST review the result: delete vague
    questions and ones several documents could answer, otherwise the metrics will be misleading."""
    root, chunks = Path(docs_root), []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        chunks += make_chunks(str(rel), path.read_text(encoding="utf-8"), rel.parts[0], str(rel), strategy)
    chunks = [c for c in chunks if len(c.text.split()) >= 25]
    random.Random(seed).shuffle(chunks)
    rows, seen = [], set()
    for c in chunks:
        if len(rows) >= n:
            break
        q, _ = llm.generate(
            "Write ONE specific question that a user could ask and that is answered by the text below. "
            "Do not mention 'the text', 'the passage' or 'the document'. Return only the question.\n\n" + c.text[:1500])
        q = q.strip().strip('"')
        if q.endswith("?") and q.lower() not in seen:
            seen.add(q.lower())
            rows.append({"question": q, "expected_source": c.source})
    return rows
