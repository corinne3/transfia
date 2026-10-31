"""Construit le jeu de données de fine-tuning (F10) : passage -> QCM au bon format.

Principe : DISTILLATION. Un modèle "professeur" (le plus gros dont tu disposes : qwen2.5:3b en
local, ou un modèle cloud) génère des exemples ; on FILTRE (JSON valide, sources existantes) ;
un petit modèle "élève" apprendra ensuite à reproduire ce comportement.

    python finetuning/build_dataset.py --n 200 --teacher qwen2.5:3b
    python finetuning/build_dataset.py --n 40 --teacher qwen3:4b      # professeur Apache 2.0 (recommandé)

Sortie : finetuning/data/train.jsonl et val.jsonl, au format "messages" (chat) attendu par TRL.
⚠️ Vérifie la licence du modèle professeur : certaines interdisent d'entraîner d'autres modèles
avec leurs sorties.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from transfia.config import load_settings
from transfia.container import AppContext
from transfia.core.models import Hit
from transfia.llm.base import StructuredOutputError

REJECT = ("n'existe pas", "ambiguë", "reprend les mêmes", "retirées", "À REVOIR", "ABSENT")
from transfia.generation.generator import FINETUNE_QCM_SYSTEM as SYSTEM   # même consigne qu'à l'utilisation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--teacher", default=None, help="modèle Ollama professeur (défaut : big_model de la config)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--min-words", type=int, default=60, help="taille minimale d'un passage utilisé")
    ap.add_argument("--split-only", action="store_true",
                    help="ne génère rien : redécoupe data/all.jsonl en train/val (après une interruption)")
    args = ap.parse_args()
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    all_path = out / "all.jsonl"
    if args.split_only:
        split(all_path, out, args.val)
        return

    settings = load_settings(args.config)
    ctx = AppContext(settings)
    ctx.ensure_index()
    teacher = args.teacher or settings.llm.big_model or settings.llm.model
    from transfia.llm.factory import build_llm

    gen = ctx.generator
    gen.llm = build_llm(settings, model=teacher)
    chunks = [c for c in ctx.store.all_chunks() if len(c.text.split()) > args.min_words]
    print(f"{len(chunks)} passages candidats (> {args.min_words} mots), professeur : {teacher}")
    random.Random(0).shuffle(chunks)

    rows = [json.loads(line) for line in all_path.read_text(encoding="utf-8").splitlines()] if all_path.exists() else []
    done_ids = {r.get("chunk_id") for r in rows}
    if rows:
        print(f"Reprise : {len(rows)} exemples déjà générés dans {all_path}")
    rejected: dict[str, int] = {}
    t0 = time.perf_counter()
    for c in chunks:
        if len(rows) >= args.n:
            break
        if c.id in done_ids:
            continue
        niveau = random.choice(["bac_pro", "bts", "grand_public"])
        try:
            res = gen.generate("quiz", c.meta.get("title", "élevage"), niveau, n_questions=2,
                               passages=[Hit(chunk=c, score=1.0)])
        except (StructuredOutputError, ValueError) as exc:
            key = f"erreur {type(exc).__name__}"            # compté et affiché : un échec silencieux ne s'analyse pas
            rejected[key] = rejected.get(key, 0) + 1
            continue
        # FILTRE QUALITÉ (distillation : l'élève recopie TOUT, y compris les défauts du professeur).
        # On rejette : source inventée, question ambiguë ou en doublon, questions retirées, termes absents.
        reason = next((r for r in REJECT if any(r in w for w in res.warnings)), None)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        user = f"Niveau : {niveau}\nPassage [1] : {c.text}"
        row = {"chunk_id": c.id, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                                              {"role": "assistant", "content": res.content.model_dump_json()}]}
        rows.append(row)
        with all_path.open("a", encoding="utf-8") as f:     # sauvegarde AU FIL DE L'EAU (reprise possible)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        el = time.perf_counter() - t0
        print(f"{len(rows)}/{args.n} — {el / 60:.0f} min écoulées — rejets : {rejected or 0}")

    split(all_path, out, args.val)
    print(f"Rejets par le filtre qualité : {rejected or 'aucun'}")


def split(all_path: Path, out: Path, val_ratio: float) -> None:
    rows = ([json.loads(line) for line in all_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if all_path.exists() else [])
    if not rows:
        print("Aucun exemple.")
        return
    n_val = max(1, int(len(rows) * val_ratio))
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    print(f"→ {len(rows) - n_val} exemples d'entraînement, {n_val} de validation dans {out}")


if __name__ == "__main__":
    main()
