"""Calculs de dimensionnement (section 11.1 du dossier) — à connaître pour dimensionner.

Toutes les fonctions renvoient des ORDRES DE GRANDEUR (les vrais chiffres dépendent du modèle,
du moteur d'inférence et du matériel). Commande : ``transfia size --params 3 --bits 4``.
"""
from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_PARAM = {32: 4.0, 16: 2.0, 8: 1.0, 5: 0.69, 4: 0.6, 3: 0.48, 2: 0.33}   # 4 bits ≈ 0,6 (Q4_K_M + métadonnées)


def weights_gb(params_billion: float, bits: int = 4) -> float:
    """Mémoire des POIDS ≈ nombre de paramètres × octets par paramètre."""
    return round(params_billion * BYTES_PER_PARAM[bits], 2)


def kv_cache_gb(context_tokens: int, n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_value: int = 2,
                batch: int = 1) -> float:
    """KV cache = 2 (K et V) × couches × têtes KV × dim. tête × tokens × octets × batch.

    Il grandit LINÉAIREMENT avec le contexte : un contexte 8 fois plus long = 8 fois plus de mémoire.
    (Les modèles récents réduisent n_kv_heads : c'est le "grouped-query attention".)
    """
    return round(2 * n_layers * n_kv_heads * head_dim * context_tokens * bytes_per_value * batch / 1e9, 3)


def decode_tokens_per_s(memory_bandwidth_gbs: float, model_gb: float, efficiency: float = 0.6) -> float:
    """Génération limitée par la BANDE PASSANTE : à chaque token, on relit tous les poids.

    tokens/s ≈ bande passante / taille du modèle (× une efficacité réelle de ~50-70 %).
    """
    return round(memory_bandwidth_gbs * efficiency / model_gb, 1)


def latency_s(prompt_tokens: int, output_tokens: int, prefill_tps: float, decode_tps: float) -> dict:
    ttft = prompt_tokens / prefill_tps
    return {"ttft_s": round(ttft, 2), "total_s": round(ttft + output_tokens / decode_tps, 2)}


def index_size_mb(n_chunks: int, dim: int, bytes_per_value: float = 4, hnsw_overhead: float = 1.3) -> float:
    return round(n_chunks * dim * bytes_per_value * hnsw_overhead / 1e6, 1)


def api_cost_per_day(requests_per_day: int, tokens_in: int, tokens_out: int, price_in_per_m: float,
                     price_out_per_m: float) -> float:
    """Coût/jour = requêtes × (tokens entrée × prix entrée + tokens sortie × prix sortie), prix par million."""
    return round(requests_per_day * (tokens_in * price_in_per_m + tokens_out * price_out_per_m) / 1e6, 2)


def rag_prompt_tokens(n_chunks: int, chunk_tokens: int, system_tokens: int = 250, question_tokens: int = 30) -> int:
    return n_chunks * chunk_tokens + system_tokens + question_tokens


def energy_wh(watts: float, seconds: float) -> float:
    return round(watts * seconds / 3600, 4)


def co2_g(wh: float, grid_gco2_per_kwh: float = 50.0) -> float:
    return round(wh / 1000 * grid_gco2_per_kwh, 5)


@dataclass
class ModelShape:
    """Formes typiques (ordres de grandeur) pour les calculs de KV cache."""
    params_b: float
    n_layers: int
    n_kv_heads: int
    head_dim: int


TYPICAL = {
    "1.5b": ModelShape(1.5, 28, 2, 128),
    "3b": ModelShape(3, 36, 2, 128),
    "7b": ModelShape(7, 28, 4, 128),
    "8b": ModelShape(8, 32, 8, 128),
}


def sizing_report(params_b: float = 3, bits: int = 4, context: int = 4096, ram_gb: float = 8,
                  bandwidth_gbs: float = 40, shape: str | None = None) -> dict:
    shape_ = TYPICAL.get(shape or f"{params_b:g}b") or ModelShape(params_b, 32, 8, 128)
    w = weights_gb(params_b, bits)
    kv = kv_cache_gb(context, shape_.n_layers, shape_.n_kv_heads, shape_.head_dim)
    need = round(w + kv + 1.5, 2)
    return {
        "poids_go": w, "kv_cache_go": kv, "ram_necessaire_go (avec marge 1,5 Go)": need,
        "tient_dans_la_ram": need <= ram_gb * 0.7,      # on garde ~30 % pour l'OS et le reste de l'appli
        "tokens_par_s_estimes": decode_tokens_per_s(bandwidth_gbs, w),
        "toutes_precisions_go": {f"{b} bits": weights_gb(params_b, b) for b in (16, 8, 4)},
    }
