from __future__ import annotations
from dataclasses import dataclass, field


NOT_FOUND = "I could not find this in the documents you can access."


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    acl: str                      # "public" | "internal" | "restricted"
    source: str                   # file path shown in citations
    heading: str = ""
    embedding: list[float] | None = None


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass
class Answer:
    text: str
    route: str                                   # "docs" | "sql"
    citations: list[dict] = field(default_factory=list)
    sql: str | None = None
    rows: list[dict] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
