"""LLMs. `OfflineLLM` is extractive (quotes the best chunk) so the pipeline runs without any key."""
from __future__ import annotations
import re
from typing import Protocol

from .embeddings import tokenize
from .models import NOT_FOUND, Usage
from .retrieval import STOP


class LLM(Protocol):
    name: str
    def generate(self, prompt: str, *, system: str = "") -> tuple[str, Usage]: ...


class OfflineLLM:
    """Extractive stand-in so everything runs without an API key: quotes the passage sentence that best
    matches the question, or says it could not find an answer. A real model replaces it in production."""
    name = "offline-extractive"

    def generate(self, prompt: str, *, system: str = ""):
        qtoks = {t for t in tokenize(prompt.rsplit("Question:", 1)[-1]) if t not in STOP}
        best, best_key = None, (0, 0)
        for line in prompt.split("\n"):
            m = re.match(r"\[(\d+)\] (.*)", line)
            if not m:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", m.group(2).strip()):
                key = (len(qtoks & set(tokenize(sentence))), len(sentence.split()))
                if key[0] > 0 and key > best_key:
                    best, best_key = f"{sentence} [{m.group(1)}]", key
        text = best or NOT_FOUND
        return text, Usage(len(prompt) // 4, len(text) // 4)


class ScriptedLLM:
    """Returns canned replies in order. Used in tests."""
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt: str, *, system: str = ""):
        self.prompts.append(prompt)
        return self.replies.pop(0), Usage(len(prompt) // 4, 10)


class GeminiLLM:
    def __init__(self, client, model: str) -> None:
        self.client, self.model, self.name = client, model, model

    def generate(self, prompt: str, *, system: str = ""):
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system or None, temperature=0.1))
        u = resp.usage_metadata
        usage = Usage(getattr(u, "prompt_token_count", 0) or 0, getattr(u, "candidates_token_count", 0) or 0)
        return (resp.text or "").strip(), usage


def make_gemini_client(api_key: str | None, project: str | None, location: str):
    from google import genai
    if api_key:
        return genai.Client(api_key=api_key)
    if project:
        return genai.Client(vertexai=True, project=project, location=location)
    raise RuntimeError("Set GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT for Vertex AI.")
