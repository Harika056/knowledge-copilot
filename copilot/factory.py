"""Builds the embedder, LLM and store from environment settings (shared by the API and the CLI)."""
from __future__ import annotations

from .config import Settings
from .embeddings import GeminiEmbedder, HashEmbedder
from .llm import GeminiLLM, OfflineLLM, make_gemini_client
from .store import MemoryStore, PgVectorStore


def build_components(s: Settings):
    if s.provider == "gemini":
        client = make_gemini_client(s.gemini_api_key, s.gcp_project, s.gcp_location)
        embedder, llm = GeminiEmbedder(client, s.embed_model), GeminiLLM(client, s.chat_model, tuple(m.strip() for m in s.chat_fallbacks.split(",") if m.strip()))
    elif s.provider == "offline":
        embedder, llm = HashEmbedder(), OfflineLLM()
    else:
        raise ValueError(f"PROVIDER must be 'offline' or 'gemini', got '{s.provider}'")
    store = PgVectorStore(s.database_url, dim=embedder.dim) if s.database_url else MemoryStore()
    return embedder, llm, store
