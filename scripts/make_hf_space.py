"""Builds a ready-to-upload folder for a free Hugging Face Space (Docker). Usage: python scripts/make_hf_space.py"""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "hf_space_build"
if out.exists():
    shutil.rmtree(out)
out.mkdir()
for name in ("Dockerfile", "pyproject.toml", "LICENSE"):
    shutil.copy(root / name, out / name)
for d in ("copilot", "sample_data"):
    shutil.copytree(root / d, out / d, ignore=shutil.ignore_patterns("__pycache__", "*.db", "*.pyc"))
(out / "README.md").write_text("""---
title: Knowledge Copilot
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
pinned: false
---

Knowledge Copilot: hybrid RAG + guarded text-to-SQL with role-based access control. Demo uses synthetic documents only.
""", encoding="utf-8")
print("Ready. Upload everything inside:", out)
