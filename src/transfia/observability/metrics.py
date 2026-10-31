"""Mesures de sobriété (F14) : temps, mémoire, énergie, carbone.

- Énergie (kWh) = puissance (W) × durée (h) / 1000
- Carbone (gCO2e) = énergie (kWh) × intensité carbone de l'électricité (gCO2e/kWh)

Trois niveaux de précision :
1. "estimate" : puissance CPU moyenne supposée (ex. 25 W). Grossier, mais sans dépendance.
2. "codecarbon" : CodeCarbon lit les compteurs matériels quand ils sont disponibles.
3. Wattmètre physique : la vraie mesure (hors code !).
Rappel du dossier : l'électricité n'est qu'une partie de l'empreinte (fabrication du matériel).
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def rss_mb() -> float | None:
    """Mémoire résidente du processus, en Mo (psutil si présent ; Windows n'a pas `resource`)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux : Ko
        except ImportError:
            return None


@dataclass
class Measurement:
    seconds: float = 0.0
    kwh: float = 0.0
    gco2: float = 0.0
    rss_mb: float | None = None
    method: str = "estimate"
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seconds": round(self.seconds, 4),
            "wh": round(self.kwh * 1000, 6),
            "gco2": round(self.gco2, 6),
            "rss_mb": None if self.rss_mb is None else round(self.rss_mb, 1),
            "method": self.method,
            **self.extra,
        }


class EnergyMeter:
    def __init__(self, method: str = "estimate", cpu_watts: float = 25.0, grid_gco2_per_kwh: float = 50.0):
        self.method = method
        self.cpu_watts = cpu_watts
        self.grid = grid_gco2_per_kwh

    @contextmanager
    def measure(self, label: str = "task"):
        m = Measurement(method=self.method)
        tracker = None
        if self.method == "codecarbon":
            try:
                from codecarbon import OfflineEmissionsTracker

                tracker = OfflineEmissionsTracker(country_iso_code="FRA", project_name=label,
                                                  log_level="error", save_to_file=False)
                tracker.start()
            except Exception:  # CodeCarbon absent ou non supporté : on retombe sur l'estimation
                tracker, m.method = None, "estimate"
        t0 = time.perf_counter()
        try:
            yield m
        finally:
            m.seconds = time.perf_counter() - t0
            m.rss_mb = rss_mb()
            if tracker is not None:
                kg = tracker.stop() or 0.0
                data = getattr(tracker, "final_emissions_data", None)
                m.kwh = float(getattr(data, "energy_consumed", 0.0) or 0.0)
                m.gco2 = kg * 1000
            else:
                m.kwh = self.cpu_watts * m.seconds / 3600 / 1000
                m.gco2 = m.kwh * self.grid


def percentile(values: list[float], p: float) -> float:
    """Percentile par interpolation linéaire (p50 = médiane, p95 = latence "de queue")."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
