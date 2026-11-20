"""Captures LinkedIn-ready screenshots from a RUNNING copilot (use your real Gemini + Postgres setup!).

    DEMO_MODE=1 PROVIDER=gemini ... uvicorn copilot.main:build_app --factory --port 8000
    pip install playwright && playwright install chromium
    python scripts/screenshots.py --url http://localhost:8000 --out docs/images

Screenshots show whatever your deployment answers, so run it against the real thing, not the offline demo."""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

SCENARIOS = [  # (file, role, question)
    ("01-member-cannot-see-restricted", "MEMBER", "What is the L4 engineer salary band?"),
    ("02-admin-can-see-restricted", "ADMIN", "What is the L4 engineer salary band?"),
    ("03-public-answer-with-citation", "MEMBER", "How often are backups taken and how long are they kept?"),
    ("04-sql-answer-with-guard", "MANAGER", "What is the total order amount by region?"),
]
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--out", default="docs/images")
ap.add_argument("--skip", nargs="*", default=[], help="scenario file prefixes to skip, e.g. 04")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 700}, device_scale_factor=2)
    for name, role, question in SCENARIOS:
        if any(name.startswith(s) for s in a.skip):
            continue
        page.goto(a.url); page.wait_for_function("document.querySelectorAll('#roles button').length === 3")
        page.click(f'#roles button[data-role="{role}"]'); page.fill("#q", question); page.click("#go")
        page.wait_for_function("!document.getElementById('go').disabled && !document.getElementById('out').textContent.startsWith('Thinking')", timeout=60000)
        page.wait_for_timeout(300)
        page.locator(".wrap").screenshot(path=str(out / f"{name}.png"))
        print("saved", out / f"{name}.png")
    browser.close()
