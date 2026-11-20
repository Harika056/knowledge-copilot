import json
import math

import pytest

from copilot import cli
from copilot.corpus import clean_markdown, draft_golden, prepare_corpus
from copilot.embeddings import GeminiEmbedder
from copilot.evaluation import load_golden
from copilot.llm import GeminiLLM, ScriptedLLM

LONG = " ".join(f"word{i}" for i in range(60))


def test_clean_markdown_strips_front_matter_and_shortcodes_and_keeps_title():
    raw = f'---\ntitle: "Pods"\nweight: 5\n---\n{{{{< note >}}}}\nhidden\n{{{{< /note >}}}}\n<!-- c -->\nBody text.\n'
    out = clean_markdown(raw)
    assert out.startswith("# Pods") and "weight" not in out and "{{" not in out and "<!--" not in out
    assert "Body text." in out


def test_prepare_corpus_assigns_access_labels_and_skips_stubs(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for rel in ["concepts/a.md", "security/b.md", "admin/c.md", "concepts/stub.md"]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("# T\n" + (LONG if "stub" not in rel else "tiny"))
    counts = prepare_corpus(src, dst, internal=["admin"], restricted=["security"])
    assert counts == {"public": 1, "internal": 1, "restricted": 1, "skipped_short": 1}
    assert (dst / "restricted" / "security" / "b.md").exists()


def test_draft_golden_writes_reviewable_questions(tmp_path):
    d = tmp_path / "docs" / "public"
    d.mkdir(parents=True)
    (d / "a.md").write_text("# A\n\n## One\n" + LONG + "\n\n## Two\n" + LONG)
    # two chunks -> two LLM calls; the reply that is not a question must be dropped
    rows = draft_golden(tmp_path / "docs", ScriptedLLM(["What is one?", "not a question"]), n=5)
    assert rows == [{"question": "What is one?", "expected_source": "public/a.md"}]
    out = tmp_path / "g.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(load_golden(out)) == len(rows)


def test_cli_prepare_ingest_ask_roundtrip(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src"
    (src / "guide").mkdir(parents=True)
    (src / "guide" / "x.md").write_text("# Backups\n\nBackups run every 6 hours and are kept for 30 days. " + LONG)
    dst = tmp_path / "corpus"
    assert cli.main(["prepare", "--src", str(src), "--dst", str(dst)]) == 0
    monkeypatch.setenv("DOCS_ROOT", str(dst))
    monkeypatch.setenv("PROVIDER", "offline")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert cli.main(["ask", "How often do backups run?", "--role", "MEMBER"]) == 0
    assert "6 hours" in capsys.readouterr().out


def test_cli_check_reports_failure_with_a_hint_when_gemini_is_not_configured(monkeypatch, capsys):
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert cli.main(["check"]) == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().out


# ---- Gemini adapters, driven by the SDK's real response types (no network) ----
try:
    from google.genai import types as genai_types
except ImportError:  # only the two adapter tests below need the SDK; the rest of this file still runs
    genai_types = None
needs_sdk = pytest.mark.skipif(genai_types is None, reason='pip install -e ".[gcp]" to run the Gemini adapter tests')


class FakeModels:
    def __init__(self):
        self.embed_calls, self.gen_calls = [], []

    def embed_content(self, model, contents, config):
        self.embed_calls.append((model, len(contents), config.output_dimensionality))
        return genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=[3.0, 4.0, 0.0, 0.0]) for _ in contents])

    def generate_content(self, model, contents, config):
        self.gen_calls.append((model, config.system_instruction))
        return genai_types.GenerateContentResponse(
            candidates=[genai_types.Candidate(content=genai_types.Content(role="model", parts=[genai_types.Part(text=" Hello [1] ")]))],
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(prompt_token_count=11, candidates_token_count=4))


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


@needs_sdk
def test_gemini_embedder_batches_normalises_and_requests_the_configured_dimension():
    c = FakeClient()
    vecs, usage = GeminiEmbedder(c, "gemini-embedding-001", dim=4).embed([f"t{i}" for i in range(250)])
    assert len(vecs) == 250 and [n for _, n, _ in c.models.embed_calls] == [100, 100, 50]
    assert all(d == 4 for _, _, d in c.models.embed_calls)
    assert math.isclose(math.sqrt(sum(x * x for x in vecs[0])), 1.0) and vecs[0][:2] == [0.6, 0.8]


@needs_sdk
def test_gemini_llm_reads_text_and_real_token_usage():
    c = FakeClient()
    text, usage = GeminiLLM(c, "gemini-2.5-flash").generate("q", system="be brief")
    assert text == "Hello [1]" and (usage.prompt_tokens, usage.completion_tokens) == (11, 4)
    assert c.models.gen_calls == [("gemini-2.5-flash", "be brief")]
