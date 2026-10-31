"""F4 — embeddings, similarité, BM25, fusion RRF, reranking, filtres, moteur complet."""
import math

import numpy as np
from helpers import shared_ctx

from transfia.core.models import Chunk, Hit
from transfia.embeddings.embedders import HashingEmbedder
from transfia.retrieval.bm25 import BM25
from transfia.retrieval.hybrid import rrf, weighted_fusion
from transfia.retrieval.rerank import LexicalReranker
from transfia.retrieval.similarity import cosine_matrix, cosine_python, top_k
from transfia.retrieval.vector_store import InMemoryVectorStore, _match


def test_cosine_by_hand_matches_numpy():
    a, b = [1.0, 2.0, 3.0], [2.0, 0.5, 1.0]
    expected = float(cosine_matrix(np.array([a]), np.array([b]))[0, 0])
    assert math.isclose(cosine_python(a, b), expected, rel_tol=1e-9)
    assert math.isclose(cosine_python(a, a), 1.0)


def test_top_k_is_sorted():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k(scores, 2).tolist() == [1, 3]
    assert top_k(scores, 10).tolist() == [1, 3, 2, 0]


def test_hashing_embedder_is_normalized_and_captures_surface_similarity():
    e = HashingEmbedder(256)
    v = e.embed_documents(["mammite des vaches", "mammites chez la vache", "émissions de méthane"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_bm25_prefers_rare_matching_terms_and_roundtrips(tmp_path=None):
    import tempfile
    from pathlib import Path

    bm = BM25().fit(["a", "b", "c"], ["la vache mange du fourrage", "la mammite de la vache", "le porc mange"])
    res = bm.search("mammite", k=3)
    assert res[0][0] == "b" and len(res) == 1
    assert bm.idf("mammite") > bm.idf("vache")             # terme plus rare = plus de poids
    p = Path(tempfile.mkdtemp()) / "bm25.json"
    bm.save(p)
    assert BM25.load(p).search("mammite", 1)[0][0] == "b"


def _hit(cid, score=1.0):
    return Hit(chunk=Chunk(id=cid, article_id=cid[0], text=cid), score=score)


def test_rrf_rewards_documents_ranked_well_in_both_lists():
    fused = rrf([[_hit("x"), _hit("y"), _hit("z")], [_hit("y"), _hit("w"), _hit("x")]], k=60)
    assert [h.chunk.id for h in fused][:2] == ["y", "x"]
    assert weighted_fusion([[_hit("a", 10), _hit("b", 1)], [_hit("b", 0.9), _hit("a", 0.1)]], [0.5, 0.5])


def test_metadata_filters():
    meta = {"species": ["bovin"], "year": 2022}
    assert _match(meta, {"species": "bovin"})
    assert not _match(meta, {"species": "porcin"})
    assert _match(meta, {"year_min": 2020}) and not _match(meta, {"year_min": 2023})


def test_in_memory_store_upsert_and_persistence():
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    store = InMemoryVectorStore(d)
    e = HashingEmbedder(64)
    chunks = [Chunk(id=f"c{i}", article_id="a", text=t, meta={"species": [s]})
              for i, (t, s) in enumerate([("vache lait", "bovin"), ("porc sevrage", "porcin")])]
    store.upsert(chunks, e.embed_documents([c.text for c in chunks]))
    store.upsert([chunks[0]], e.embed_documents([chunks[0].text]))       # mise à jour, pas de doublon
    assert store.count() == 2
    store.save()
    again = InMemoryVectorStore(d)
    hits = again.search(e.embed_query("porc"), 1, {"species": "porcin"})
    assert hits[0].chunk.id == "c1"


def test_lexical_reranker_moves_best_passage_up():
    hits = [_hit("bruit"), Hit(chunk=Chunk(id="ok", article_id="o", text="facteurs de risque des mammites"), score=0.5)]
    out = LexicalReranker().rerank("facteurs de risque mammites", hits, 2)
    assert out[0].chunk.id == "ok"


def test_engine_modes_find_the_right_article():
    ctx = shared_ctx()
    for mode in ("bm25", "dense", "hybrid"):
        hits = ctx.engine.search("facteurs de risque des mammites", k=3, mode=mode)
        assert hits and hits[0].chunk.article_id == "exemple-001", mode


def test_engine_species_filter():
    hits = shared_ctx().engine.search("alimentation", k=5, filters={"species": "porcin"})
    assert hits and all("porcin" in h.chunk.meta["species"] for h in hits)


def test_local_model_path_and_lazy_loading():
    import tempfile
    from pathlib import Path

    from transfia.config import load_settings
    from transfia.container import Lazy

    root = Path(tempfile.mkdtemp())
    s = load_settings("config/test.yaml", models_dir=str(root))
    assert s.model_path("intfloat/multilingual-e5-small") == "intfloat/multilingual-e5-small"   # pas copié
    local = root / "intfloat__multilingual-e5-small"
    local.mkdir()
    (local / "config.json").write_text("{}")
    assert s.model_path("intfloat/multilingual-e5-small") == str(local)                       # copie locale

    built = []
    lazy = Lazy(lambda: built.append(1) or HashingEmbedder(32))
    assert not built and not lazy.loaded            # rien n'est construit tant qu'on ne s'en sert pas
    assert lazy.dim == 32 and built == [1]


def test_bm25_search_loads_no_model():
    from helpers import make_ctx

    ctx = make_ctx()
    ctx.__dict__.pop("engine", None)                 # moteur neuf : rien de chargé
    ctx.engine.search("mammites", k=2, mode="bm25", rerank=False)
    assert not ctx.engine.embedder.loaded and not ctx.engine.reranker.loaded


def test_conversational_wording_is_removed_before_search():
    from transfia.retrieval.query import clean_query

    # cas réel : reranker -5,7 (refus) avec la tournure, -2,6 (bon passage) sans
    assert clean_query("Que dit le corpus sur le colostrum ?") == "le colostrum"
    assert clean_query("Parle-moi du sevrage des porcelets") == "sevrage des porcelets"
    assert clean_query("Peux-tu m'expliquer la mammite ?") == "la mammite"
    assert clean_query("Selon les articles, comment réduire le méthane ?") == "comment réduire le méthane"
    q = "Pourquoi le colostrum est-il indispensable au veau ?"
    assert clean_query(q) == q                                   # question normale : inchangée
    assert clean_query("Que dit le corpus ?") == "Que dit le corpus ?"   # il ne resterait rien : on garde


def test_task_instructions_are_removed_before_search():
    from transfia.retrieval.query import clean_query

    # cas réel : plan refusé ("corpus ne couvre pas") alors qu'un article entier traite du méthane entérique
    assert clean_query("Construis une séquence de 2 séances sur la réduction du méthane entérique") == \
        "la réduction du méthane entérique"
    assert clean_query("Propose une activité sur la santé du veau") == "la santé du veau"
    assert clean_query("Prépare 2 questions de QCM niveau bac pro sur le sevrage des porcelets") == \
        "le sevrage des porcelets"
    assert clean_query("Fais-moi une fiche de synthèse sur les mammites") == "les mammites"
