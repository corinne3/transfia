"""Fondations from scratch : on vérifie que chaque algorithme fait ce qu'il prétend."""
import numpy as np

from transfia.foundations import lora_quant, ml_scratch, neural_net, rl, search_symbolic, transformer, word2vec


def test_search_astar_explores_less_than_bfs():
    grid = ["S....#....", ".##..#.##.", ".#.......#", ".#.###.#..", "...#...#.G"]
    p1, e1 = search_symbolic.bfs(grid, (0, 0), (4, 9))
    p2, e2 = search_symbolic.astar(grid, (0, 0), (4, 9))
    assert len(p1) == len(p2) and e2 <= e1
    facts, trace = search_symbolic.forward_chaining({"lait_grumeleux"})
    assert "appeler_veterinaire" in facts and len(trace) == 2


def test_linear_regression_gd_matches_normal_equation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X @ np.array([3.0, -2.0]) + 1.0 + rng.normal(0, 0.1, 200)
    m = ml_scratch.LinearRegressionGD(lr=0.1, epochs=500).fit(X, y)
    w, b = ml_scratch.normal_equation(X, y)
    assert np.allclose(m.w, w, atol=1e-2) and abs(m.b - b) < 1e-2


def test_classifiers_from_scratch():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(3, 1, (60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    for model in (ml_scratch.LogisticRegressionGD(), ml_scratch.KNN(3), ml_scratch.DecisionTree(3)):
        assert np.mean(model.fit(X, y).predict(X) == y) > 0.9
    assert ml_scratch.gini(np.array([1, 1, 1])) == 0.0 and ml_scratch.gini(np.array([0, 1])) == 0.5
    labels, _, _ = ml_scratch.kmeans(X, 2)
    assert len(set(labels[:60])) == 1 and labels[0] != labels[-1]
    _, explained = ml_scratch.pca(X, 2)
    assert abs(explained.sum() - 1) < 1e-9
    p, r, f = ml_scratch.precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert p == r == f == 0.5


def test_neural_net_gradient_and_learning():
    X, y = neural_net.spirals(40)
    m = neural_net.MLP(2, 32, 3)
    assert neural_net.gradient_check(m, X[:10], y[:10]) < 1e-5
    hist = m.fit(X, y, epochs=150, lr=0.02)
    assert hist[-1] < hist[0] / 3 and np.mean(m.predict(X) == y) > 0.9


def test_transformer_pieces():
    tok = transformer.BPETokenizer().train(transformer.CORPUS * 2, 30)
    assert tok.decode(tok.encode("la vache mange")) == "la vache mange"
    x = np.random.default_rng(0).normal(size=(4, 8))
    _, w = transformer.scaled_dot_product_attention(x, x, x, transformer.causal_mask(4))
    assert np.allclose(w.sum(-1), 1) and np.allclose(np.triu(w, 1), 0)        # pas d'accès au futur
    lm = transformer.TinyTransformerLM(30)
    assert lm.logits([1, 2, 3]).shape == (3, 30)
    logits = np.array([1.0, 5.0, 2.0])
    assert transformer.sample_next(logits, temperature=0) == 1
    assert transformer.sample_next(logits, temperature=1, top_k=1) == 1
    big = transformer.BigramLM().fit(transformer.CORPUS)
    assert big.perplexity(transformer.CORPUS) < len(big.vocab)


def test_rl_and_dpo():
    res = rl.epsilon_greedy_bandit([0.1, 0.9], steps=1000)
    assert res["pulls"][1] > res["pulls"][0]
    _, path = rl.q_learning()
    assert path[-1] == (2, 3)
    assert rl.dpo_loss(-5, -9, -6, -6) < rl.dpo_loss(-9, -5, -6, -6)


def test_lora_and_quantization():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (64, 64))
    lora = lora_quant.LoRALinear(W, r=4)
    x = rng.normal(size=(5, 64))
    assert np.allclose(lora(x), x @ W.T)                                     # B = 0 : pas de perturbation au départ
    assert lora.trainable_params() == 2 * 4 * 64
    assert lora_quant.quantization_error(W, 8) < lora_quant.quantization_error(W, 4)


def test_word2vec_learns_neighbors():
    m = word2vec.Word2Vec(seed=0).fit(word2vec.SENTENCES, epochs=40)
    neighbors = [w for w, _ in m.most_similar("veau", 4)]
    assert {"agneau", "chevreau", "colostrum", "boit"} & set(neighbors)
