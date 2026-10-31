"""Évaluation humaine (F11) : faire noter des sorties par le métier, puis CALIBRER le juge LLM.

1. ``export_for_annotation`` : un CSV (ouvrable dans un tableur) avec question, réponse, sources,
   et des colonnes vides à remplir (exactitude, pertinence, niveau_adapte, commentaire).
2. ``agreement`` : compare notes humaines et notes du jury (kappa de Cohen, Spearman).
   -> on sait jusqu'où faire confiance à l'évaluation automatique.
"""
from __future__ import annotations

import csv
from pathlib import Path

from transfia.evaluation.metrics import cohen_kappa, spearman

COLUMNS = ["id", "question", "answer", "sources", "exactitude", "pertinence", "niveau_adapte", "commentaire"]


def export_for_annotation(run: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:          # utf-8-sig : accents OK dans Excel
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        w.writeheader()
        for r in run["rows"]:
            w.writerow({"id": r["id"], "question": r["question"], "answer": r["answer"], "sources": "",
                        "exactitude": "", "pertinence": "", "niveau_adapte": "", "commentaire": ""})
    return path


def load_annotations(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig") as f:
        return {row["id"]: row for row in csv.DictReader(f, delimiter=";") if row.get("exactitude")}


def agreement(run: dict, annotations: dict[str, dict], criterion: str = "exactitude") -> dict:
    human, judge = [], []
    for r in run["rows"]:
        a = annotations.get(r["id"])
        j = (r.get("judge") or {}).get("per_judge") or {}
        if not a or not j:
            continue
        human.append(int(a[criterion]))
        judge.append(round(sum(s[criterion] for s in j.values()) / len(j)))
    if len(human) < 3:
        return {"n": len(human), "message": "pas assez d'annotations communes (≥ 3)"}
    # kappa sur des notes "bonnes" (≥ 4) vs "pas bonnes" : plus robuste qu'un kappa à 5 classes sur peu de données
    hb, jb = [h >= 4 for h in human], [x >= 4 for x in judge]
    return {"n": len(human), "criterion": criterion, "kappa_binaire": round(cohen_kappa(hb, jb), 3),
            "spearman": round(spearman(human, judge), 3),
            "accord_exact": round(sum(h == x for h, x in zip(human, judge)) / len(human), 3)}
