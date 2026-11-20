import pytest

from copilot.chunking import fixed_chunks, heading_chunks, make_chunks
from copilot.models import Chunk, Hit
from copilot.retrieval import rrf
from copilot.store import BM25


def test_heading_chunks_keep_sections_together():
    text = "# Title\n\n## A\nalpha one. alpha two.\n\n## B\nbeta one."
    out = heading_chunks(text)
    assert [h for h, _ in out] == ["A", "B"]
    assert "alpha two" in out[0][1] and "beta" not in out[0][1]


def test_fixed_chunks_overlap_and_validation():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = fixed_chunks(words, size=40, overlap=10)
    assert len(chunks) == 3
    assert chunks[0][1].split()[-10:] == chunks[1][1].split()[:10]
    with pytest.raises(ValueError):
        fixed_chunks(words, size=10, overlap=10)


def test_make_chunks_prefix_heading_and_reject_unknown_strategy():
    cs = make_chunks("d", "## Pricing\nCosts 5 dollars.", "public", "d.md")
    assert cs[0].text.startswith("Pricing.") and cs[0].id == "d#0"
    with pytest.raises(ValueError):
        make_chunks("d", "x", "public", "d.md", strategy="nope")


def _c(i, text, acl="public"):
    return Chunk(id=i, doc_id=i, text=text, acl=acl, source=i)


def test_bm25_ranks_rare_term_matches_first():
    bm = BM25([_c("a", "cats and dogs"), _c("b", "quarterly revenue report"), _c("c", "dogs and more dogs")])
    assert bm.search("revenue", 3)[0].chunk.id == "b"
    assert bm.search("zzz", 3) == []


def test_rrf_rewards_agreement_between_rankers():
    a, b, c = _c("a", ""), _c("b", ""), _c("c", "")
    fused = rrf([[Hit(a, 1), Hit(b, .9), Hit(c, .8)], [Hit(c, 1), Hit(b, .5)]])
    assert fused[0].chunk.id in ("b", "c") and fused[-1].chunk.id == "a"


def test_offline_llm_quotes_the_matching_sentence_and_admits_ignorance():
    from copilot.llm import OfflineLLM
    from copilot.models import NOT_FOUND
    llm = OfflineLLM()
    prompt = "Passages:\n[1] Backups. Backups run every 6 hours and are kept for 30 days.\n[2] Pricing. Starter costs 29 dollars.\n\nQuestion: How often are backups taken?"
    text, _ = llm.generate(prompt)
    assert "6 hours" in text and text.endswith("[1]")
    assert llm.generate("Passages:\n[1] Starter costs 29 dollars.\n\nQuestion: What is the salary band?")[0] == NOT_FOUND
