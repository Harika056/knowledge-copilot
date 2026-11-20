"""Compare chunking strategies and retrieval modes on a golden set.

    python eval/run_eval.py                       # offline demo (hash embeddings: NOT semantic)
    PROVIDER=gemini GEMINI_API_KEY=... python eval/run_eval.py --answers   # real numbers

Write your own golden set (eval/golden.jsonl, one JSON per line: question + expected_source) over YOUR corpus.
The sample set only proves the harness works; do not quote its numbers on a resume."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.config import Settings  # noqa: E402
from copilot.embeddings import GeminiEmbedder, HashEmbedder  # noqa: E402
from copilot.evaluation import load_golden, run_ablation, to_markdown  # noqa: E402
from copilot.llm import GeminiLLM, OfflineLLM, make_gemini_client  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--docs", default="sample_data/docs")
ap.add_argument("--golden", default="eval/golden.jsonl")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--answers", action="store_true", help="also generate answers (tokens, cost, latency, groundedness proxy)")
args = ap.parse_args()

s = Settings()
if s.provider == "gemini":
    client = make_gemini_client(s.gemini_api_key, s.gcp_project, s.gcp_location)
    embedder, llm = GeminiEmbedder(client, s.embed_model), GeminiLLM(client, s.chat_model)
else:
    embedder, llm = HashEmbedder(), OfflineLLM()
    print("NOTE: offline provider. Hash embeddings are lexical only, so vector results are not meaningful.\n")

rows = run_ablation(embedder, llm, args.docs, load_golden(args.golden), args.k, s, args.answers)
print(to_markdown(rows))
out = Path("eval") / f"results-{s.provider}.json"
out.write_text(json.dumps(rows, indent=2))
print(f"\nSaved {out}")
