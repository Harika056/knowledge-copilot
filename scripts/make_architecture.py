"""Draws docs/images/architecture.svg (GitHub renders it in the README) and architecture.png (for LinkedIn).
    python scripts/make_architecture.py          # the PNG step needs:  pip install playwright && playwright install chromium"""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
INK, MUTE, LINE, SOFT, KEY, KEYFILL = "#14181f", "#5b6472", "#c9d1dd", "#f4f6fa", "#2457d6", "#e8f0ff"
parts = []


def box(x, y, w, h, title, subs=(), key=False):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{KEYFILL if key else SOFT}" stroke="{KEY if key else LINE}" stroke-width="{2 if key else 1.2}"/>')
    ty = y + (h / 2) - (len(subs) * 8) + 5
    parts.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="14" font-weight="700" fill="{INK}">{title}</text>')
    for i, s in enumerate(subs):
        parts.append(f'<text x="{x + w / 2}" y="{ty + 18 + i * 15}" text-anchor="middle" font-size="11.5" fill="{MUTE}">{s}</text>')


def arrow(d):
    parts.append(f'<path d="{d}" fill="none" stroke="{MUTE}" stroke-width="1.6" marker-end="url(#a)"/>')


def label(x, y, t):
    parts.append(f'<text x="{x}" y="{y}" font-size="11.5" font-weight="700" letter-spacing="1" fill="{MUTE}">{t}</text>')


parts.append(f'<text x="30" y="34" font-size="20" font-weight="700" fill="{INK}">Knowledge Copilot: hybrid RAG + guarded text-to-SQL</text>')
box(30, 56, 150, 58, "Web UI / API client", ["Bearer JWT, role"])
box(225, 56, 150, 58, "FastAPI", ["auth · input validation"])
arrow("M180 85 H225")
box(30, 225, 120, 200, "Router", ["documents", "or data?"])
arrow("M300 114 V145 H90 V225")

label(185, 208, "DOCUMENT PATH")
box(185, 225, 140, 80, "Embed query", ["Gemini embeddings"])
box(350, 225, 210, 80, "Postgres + pgvector", ["keyword (FTS) + vector (HNSW)", "role filter INSIDE the query"], key=True)
box(585, 225, 130, 80, "RRF + rerank", ["relevance cutoff"])
box(740, 225, 150, 80, "Gemini answer", ["cites sources [n]"])
arrow("M150 265 H185"); arrow("M325 265 H350"); arrow("M560 265 H585"); arrow("M715 265 H740")

label(185, 358, "DATA PATH · ADMIN & MANAGER ONLY")
box(185, 375, 170, 80, "Gemini writes SELECT", ["one query, nothing runs yet"])
box(380, 375, 230, 80, "SQL safety guard", ["one statement · SELECT only", "table allow-list · LIMIT 100"], key=True)
box(635, 375, 180, 80, "Read-only database", ["answer built from the rows"])
arrow("M150 415 H185"); arrow("M355 415 H380"); arrow("M610 415 H635")

box(920, 225, 130, 230, "Response", ["answer", "citations", "latency", "tokens", "cost per request"])
arrow("M890 265 H920"); arrow("M815 415 H865 V370 H920")

parts.append(f'<rect x="30" y="480" width="1020" height="42" rx="10" fill="none" stroke="{LINE}" stroke-width="1.4" stroke-dasharray="6 5"/>')
parts.append(f'<text x="540" y="506" text-anchor="middle" font-size="12.5" fill="{MUTE}">Evaluation harness: golden questions → hit@k · MRR · p95 latency · cost per request, compared across chunking and retrieval strategies</text>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 545" width="1080" height="545" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
       f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{MUTE}"/></marker></defs>'
       f'<rect width="1080" height="545" rx="14" fill="#ffffff"/>' + "".join(parts) + "</svg>")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "architecture.svg").write_text(svg, encoding="utf-8")
print("wrote", OUT / "architecture.svg")
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1080, "height": 545}, device_scale_factor=2)
        pg.set_content(f'<body style="margin:0">{svg}</body>'); pg.screenshot(path=str(OUT / "architecture.png")); b.close()
    print("wrote", OUT / "architecture.png")
except ImportError:
    print("playwright not installed: SVG written, PNG skipped (pip install playwright && playwright install chromium)")
