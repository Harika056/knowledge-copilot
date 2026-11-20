"""Chunking strategies. Comparing these is one of the experiments in the eval harness."""
from __future__ import annotations
import re

from .models import Chunk


def _words(text: str) -> list[str]:
    return text.split()


def fixed_chunks(text: str, size: int = 120, overlap: int = 20) -> list[tuple[str, str]]:
    """Plain sliding window over words. Returns (heading, text) pairs."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    words, out, i = _words(text), [], 0
    while i < len(words):
        out.append(("", " ".join(words[i:i + size])))
        if i + size >= len(words):
            break
        i += size - overlap
    return out


def heading_chunks(text: str, max_words: int = 160) -> list[tuple[str, str]]:
    """Split on markdown headings so a chunk stays inside one topic; long sections are windowed."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if buf:
                sections.append((heading, " ".join(buf)))
            heading, buf = m.group(2).strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((heading, " ".join(buf)))
    out: list[tuple[str, str]] = []
    for h, body in sections:
        if len(_words(body)) <= max_words:
            out.append((h, body))
        else:
            out.extend((h, t) for _, t in fixed_chunks(body, max_words, max_words // 6))
    return out


STRATEGIES = {"fixed": fixed_chunks, "heading": heading_chunks}


def make_chunks(doc_id: str, text: str, acl: str, source: str, strategy: str = "heading", **kw) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy '{strategy}'")
    chunks = []
    for n, (heading, body) in enumerate(STRATEGIES[strategy](text, **kw)):
        # Prefix the heading so the topic is searchable even when the body never repeats it.
        full = f"{heading}. {body}" if heading else body
        chunks.append(Chunk(id=f"{doc_id}#{n}", doc_id=doc_id, text=full, acl=acl, source=source, heading=heading))
    return chunks
