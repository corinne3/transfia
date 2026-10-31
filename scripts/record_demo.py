"""Enregistre automatiquement une vidéo de démonstration de l'interface TransfIA (pour LinkedIn).

Prérequis (une seule fois) :
    pip install playwright
    python -m playwright install chromium
Avant de lancer : `transfia serve` (terminal 1) et `cd frontend; npm run dev` (terminal 2).
Lancement :
    python scripts/record_demo.py
Résultat : data/demo/transfia-demo.webm (+ scenes.json : l'heure de début de chaque scène, pour le montage).

Le navigateur pilote l'interface tout seul : questions, refus, recherche, graphe. Ne touche pas à la souris.
Les attentes du modèle (20 s et plus sur CPU) seront accélérées au montage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:5173"
OUT = Path(__file__).resolve().parents[1] / "data" / "demo"
SIZE = {"width": 1280, "height": 720}
QUESTIONS = [
    ("question_sourcee", "Pourquoi le colostrum est-il indispensable au veau ?"),
    ("refus_hors_corpus", "Quelle est la capitale de l'Australie ?"),
]
SEARCH = "émissions de méthane entérique"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=60)
        context = browser.new_context(viewport=SIZE, record_video_dir=str(OUT), record_video_size=SIZE)
        page = context.new_page()
        t0 = time.monotonic()

        def scene(name: str) -> None:
            scenes.append({"scene": name, "start_s": round(time.monotonic() - t0, 1)})
            print(f"[{scenes[-1]['start_s']:>6} s] {name}")

        scene("ouverture")
        page.goto(URL)
        page.wait_for_selector(".dot.ok", timeout=90_000)          # l'API répond
        page.wait_for_timeout(2500)

        for name, q in QUESTIONS:
            scene(name)
            box = page.get_by_placeholder("Pose une question sur le corpus…")
            box.click()
            box.press_sequentially(q, delay=45)
            page.wait_for_timeout(600)
            box.press("Enter")
            page.wait_for_timeout(1500)
            page.get_by_role("button", name="Demander").wait_for(timeout=300_000)   # fin de la réponse
            scene(name + "_fin")
            page.wait_for_timeout(1500)
            page.mouse.wheel(0, 900)                                  # montrer les sources et la vérification
            page.wait_for_timeout(4000)

        scene("recherche")
        page.get_by_role("button", name="Recherche").first.click()
        page.wait_for_timeout(1200)
        box = page.locator("section input").first
        box.click()
        box.press_sequentially(SEARCH, delay=45)
        box.press("Enter")
        page.wait_for_timeout(15_000)                                 # 3 colonnes : BM25, dense, hybride + rerank
        page.mouse.wheel(0, 600)
        page.wait_for_timeout(3000)

        scene("graphe")
        page.get_by_role("button", name="Graphe").first.click()
        page.wait_for_selector("svg.graph g", timeout=60_000)
        page.wait_for_timeout(2500)
        nodes = page.locator("svg.graph g[style*='cursor']")
        if nodes.count():
            nodes.nth(min(3, nodes.count() - 1)).click()               # voisins d'un nœud
            page.wait_for_timeout(4000)

        scene("evaluation")
        page.get_by_role("button", name="Évaluation").first.click()
        page.wait_for_timeout(5000)
        scene("fin")

        video = page.video
        context.close()
        browser.close()
        target = OUT / "transfia-demo.webm"
        if target.exists():
            target.unlink()
        Path(video.path()).rename(target)

    (OUT / "scenes.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ vidéo : {target}\n→ scènes : {OUT / 'scenes.json'}")


if __name__ == "__main__":
    main()
