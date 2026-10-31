"""Machine learning classique (F3) : classer les passages par THÈME, trois approches comparées.

1. Baseline : TF-IDF + régression logistique.
   TF-IDF(t, d) = fréquence de t dans d × log(N / nb de documents contenant t) :
   un mot pèse s'il est fréquent dans CE document mais rare dans le corpus.
2. Embeddings + régression logistique : on remplace TF-IDF par des vecteurs SÉMANTIQUES.
3. LLM zero-shot : on demande au LLM de choisir un thème (aucun entraînement).

Bonnes pratiques illustrées (pièges classiques) :
- PIPELINE scikit-learn : le vectoriseur est ajusté SEULEMENT sur le train de chaque fold
  -> pas de DATA LEAKAGE.
- VALIDATION CROISÉE STRATIFIÉE (StratifiedKFold) : mêmes proportions de classes dans chaque fold.
- FUITE PAR GROUPE (cas vécu) : l'étiquette est le thème de l'ARTICLE, et un article donne ~80 passages.
  Si on mélange les passages, ceux d'un même article sont à la fois en train et en test : le modèle apprend
  à RECONNAÎTRE L'ARTICLE (mots discriminants vus : "aubrac", "1994", "vab"), pas le thème. Score gonflé.
  Parade : StratifiedGroupKFold, groupes = articles (un article entier est soit en train, soit en test).
  On affiche les DEUX scores, À PÉRIMÈTRE ÉGAL (mêmes classes) : l'écart mesure alors la fuite, et elle seule.
- CLASSES DÉSÉQUILIBRÉES : class_weight="balanced" et F1 MACRO (l'accuracy serait trompeuse).
- RÉGULARISATION : C = 1/λ ; petit C = forte régularisation (modèle plus simple, moins de variance).
- Rapport par classe + MATRICE DE CONFUSION : où le modèle se trompe-t-il ?
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from transfia.core.models import Chunk
from transfia.core.text import STOPWORDS_FR, normalize
from transfia.ingestion.themes import THEMES, weak_label


def labeled_data(chunks: list[Chunk], min_per_class: int = 3, with_groups: bool = False):
    """Étiquetage faible (themes.py) ; on retire les classes trop rares pour la validation croisée.
    ``with_groups`` : renvoie aussi l'article de chaque passage (pour la validation croisée groupée)."""
    X, y, g = [], [], []
    for c in chunks:
        label = c.meta.get("article_theme") or weak_label(c.text)
        if label:
            X.append(c.text)
            y.append(label)
            g.append(c.article_id)
    counts = Counter(y)
    keep = [i for i, lab in enumerate(y) if counts[lab] >= min_per_class]
    X, y, g = [X[i] for i in keep], [y[i] for i in keep], [g[i] for i in keep]
    return (X, y, g) if with_groups else (X, y)


def _cv(n_samples: int, y: list[str]) -> StratifiedKFold:
    k = max(2, min(5, min(Counter(y).values())))
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


def grouped_subset(X, y, groups, min_groups: int = 2):
    """Pour une CV groupée, chaque classe doit avoir au moins 2 ARTICLES (un pour apprendre, un pour tester).
    Renvoie (X, y, groups, n_splits, classes retirées, indices gardés)."""
    arts: dict[str, set[str]] = {}
    for lab, grp in zip(y, groups):
        arts.setdefault(lab, set()).add(grp)
    dropped = sorted(lab for lab, a in arts.items() if len(a) < min_groups)
    keep = [i for i, lab in enumerate(y) if lab not in dropped]
    X, y, groups = [X[i] for i in keep], [y[i] for i in keep], [groups[i] for i in keep]
    n_splits = max(2, min(5, min((len(a) for lab, a in arts.items() if lab not in dropped), default=2)))
    return X, y, groups, n_splits, dropped, keep


def tfidf_pipeline(C: float = 2.0):
    return make_pipeline(
        TfidfVectorizer(preprocessor=normalize, stop_words=sorted({normalize(w) for w in STOPWORDS_FR}), ngram_range=(1, 2),
                        min_df=1, sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
    )


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 3),
        "accuracy": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 3),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def compare_classifiers(chunks: list[Chunk], embedder=None, llm=None, max_llm: int = 30) -> dict:
    X, y, groups = labeled_data(chunks, with_groups=True)
    if len(set(y)) < 2:
        return {"error": "pas assez de classes étiquetées (corpus trop petit ?)"}
    cv = _cv(len(X), y)
    results = {"n": len(X), "classes": dict(Counter(y)), "articles_par_classe":
               {lab: len({g for g, yy in zip(groups, y) if yy == lab}) for lab in sorted(set(y))}}
    E = embedder.embed_documents(X) if embedder is not None else None
    # A. CV NAÏVE (passages mélangés) : gardée pour MONTRER la fuite
    results["tfidf_logreg_naif"] = evaluate_predictions(y, list(cross_val_predict(tfidf_pipeline(), X, y, cv=cv)))
    if E is not None:
        pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0), E, y, cv=cv)
        results["embeddings_logreg_naif"] = evaluate_predictions(y, list(pred))
    # B. CV GROUPÉE PAR ARTICLE : le score honnête (le modèle n'a jamais vu l'article de test)
    Xg, yg, gg, k, dropped, keep = grouped_subset(X, y, groups)
    results["groupe"] = {"classes_retirees": dropped, "n_splits": k, "n": len(Xg)}
    if len(set(yg)) >= 2:
        # Comparaison À PÉRIMÈTRE ÉGAL : CV naïve sur les MÊMES classes que la CV groupée.
        # (Sinon on compare un problème à 4 thèmes et un problème à 2 thèmes : l'écart ne mesure pas la fuite.)
        ncv = _cv(len(Xg), yg)
        results["tfidf_logreg_naif_meme_perimetre"] = evaluate_predictions(
            yg, list(cross_val_predict(tfidf_pipeline(), Xg, yg, cv=ncv)))
        gcv = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42).split(Xg, yg, gg))
        results["tfidf_logreg"] = evaluate_predictions(yg, list(cross_val_predict(tfidf_pipeline(), Xg, yg, cv=gcv)))
        if E is not None:
            Eg = E[keep]
            pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0), Eg, yg, cv=ncv)
            results["embeddings_logreg_naif_meme_perimetre"] = evaluate_predictions(yg, list(pred))
            pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0), Eg, yg, cv=gcv)
            results["embeddings_logreg"] = evaluate_predictions(yg, list(pred))
    else:
        results["groupe"]["erreur"] = "moins de 2 thèmes ont au moins 2 articles : CV groupée impossible"
    # C. LLM zero-shot (sur un échantillon : c'est lent sur CPU). Aucun entraînement -> aucune fuite possible.
    if llm is not None:
        sub = list(range(min(max_llm, len(X))))
        preds = [zero_shot_theme(llm, X[i]) for i in sub]
        results["llm_zero_shot"] = evaluate_predictions([y[i] for i in sub], preds)
    return results


def zero_shot_theme(llm, text: str) -> str:
    from transfia.llm.base import Message

    themes = ", ".join(THEMES)
    out = llm.chat([Message("user", f"Classe ce passage dans UN thème parmi : {themes}.\n"
                                    f"Réponds uniquement par le nom du thème.\n\nPassage : {text[:1500]}")],
                   temperature=0.0, max_tokens=10).content.strip().lower()
    return next((t for t in THEMES if t in out), "nutrition")


def top_terms_per_class(chunks: list[Chunk], n: int = 8) -> dict[str, list[str]]:
    """Interprétabilité : les mots qui pèsent le plus pour chaque classe (coefficients du modèle)."""
    X, y = labeled_data(chunks)
    if len(set(y)) < 2:
        return {}
    pipe = tfidf_pipeline().fit(X, y)
    vec, clf = pipe.steps[0][1], pipe.steps[1][1]
    vocab = np.array(vec.get_feature_names_out())
    if len(clf.classes_) == 2:
        coefs = np.vstack([-clf.coef_[0], clf.coef_[0]])
    else:
        coefs = clf.coef_
    return {cls: vocab[np.argsort(-coefs[i])[:n]].tolist() for i, cls in enumerate(clf.classes_)}
