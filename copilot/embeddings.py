"""Embedders. `HashEmbedder` works offline (for tests and demos) but is NOT semantic:
it only captures shared words. Use GeminiEmbedder for real quality numbers."""
from __future__ import annotations
import hashlib
import math
import re
from typing import Protocol

from .models import Usage

TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an the is are was were of to in on for and or how what which who when where do does can i my our it be by with at as from".split())


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Embedder(Protocol):
    name: str
    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]: ...


class HashEmbedder:
    name = "hash-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        vec = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts):
        return [self._one(t) for t in texts], Usage(prompt_tokens=sum(len(tokenize(t)) for t in texts))


class GeminiEmbedder:
    """Gemini embeddings through the google-genai SDK (Gemini API key, or Vertex AI with a project)."""

    def __init__(self, client, model: str, dim: int = 768) -> None:
        self.client, self.model, self.dim, self.name = client, model, dim, model

    def embed(self, texts):
        from google.genai import types
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # API batch limit
            resp = self.client.models.embed_content(
                model=self.model, contents=texts[i:i + 100],
                config=types.EmbedContentConfig(output_dimensionality=self.dim))
            vecs = [list(e.values) for e in resp.embeddings]
            for v in vecs:  # truncated embeddings must be re-normalised
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / n for x in v])
        return out, Usage(prompt_tokens=sum(len(t) // 4 for t in texts))  # approximate
