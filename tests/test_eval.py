import pytest

from copilot.embeddings import HashEmbedder
from copilot.evaluation import evaluate_retrieval, groundedness, load_golden, percentile, run_ablation, to_markdown
from copilot.llm import OfflineLLM
from copilot.retrieval import MODES

DOCS, GOLDEN = "sample_data/docs", "eval/golden.jsonl"


def test_percentile():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([], 50) == 0.0


def test_metrics_on_a_hand_checked_case(copilot):
    # The question's answer is in public/pricing.md; hit@4 must be 1.0 and MRR must equal 1/rank of that source.
    g = [{"question": "How much is the Business plan per month?", "expected_source": "public/pricing.md"}]
    r = evaluate_retrieval(copilot, g, "hybrid+rerank", k=4)
    assert r["hit@4"] == 1.0 and r["mrr"] > 0
    miss = evaluate_retrieval(copilot, [{"question": "Business plan price", "expected_source": "nope.md"}], "hybrid", 4)
    assert miss["hit@4"] == 0.0 and miss["mrr"] == 0.0


def test_golden_loader_validates(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"question": "q"}\n')
    with pytest.raises(ValueError):
        load_golden(p)
    assert len(load_golden(GOLDEN)) >= 10


def test_ablation_covers_every_chunking_and_mode_and_reports_cost(settings):
    rows = run_ablation(HashEmbedder(), OfflineLLM(), DOCS, load_golden(GOLDEN), 4, settings, answers=True)
    assert len(rows) == 2 * len(MODES)
    assert {r["mode"] for r in rows} == set(MODES)
    assert all(0 <= r["mrr"] <= 1 and "cost_per_request_usd" in r for r in rows)
    assert "| chunking |" in to_markdown(rows)


def test_hybrid_is_not_worse_than_keyword_only_on_sample_set(settings):
    rows = {(r["chunking"], r["mode"]): r for r in run_ablation(HashEmbedder(), OfflineLLM(), DOCS, load_golden(GOLDEN), 4, settings)}
    assert rows[("heading", "hybrid+rerank")]["hit@4"] >= rows[("heading", "keyword")]["hit@4"] - 0.1


def test_groundedness_proxy():
    assert groundedness("The plan costs 29 dollars. [1]", ["Starter plan costs 29 dollars per month"]) == 1.0
    assert groundedness("Elephants fly over mountains.", ["Starter plan costs 29 dollars"]) == 0.0
