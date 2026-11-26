"""Métricas usadas em toda a pipeline."""
from __future__ import annotations

import numpy as np

COMPS = ["C1", "C2", "C3", "C4", "C5"]


def qwk(real: np.ndarray, prev: np.ndarray) -> float:
    """Quadratic weighted kappa para notas 0, 40, ..., 200 (1 = concordância perfeita)."""
    k = 6
    a = np.clip(np.asarray(real, dtype=int) // 40, 0, 5)
    b = np.clip(np.asarray(prev, dtype=int) // 40, 0, 5)
    obs = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(float)
    peso = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    total = obs.sum()
    if total == 0:
        return 0.0
    esperado = np.outer(obs.sum(1), obs.sum(0)) / total
    den = (peso * esperado).sum()
    return 0.0 if den == 0 else float(1 - (peso * obs).sum() / den)


def qwk_por_competencia(real: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """real e prev: matrizes (n redações x 5 competências). Devolve o QWK de cada competência."""
    return np.array([qwk(real[:, i], prev[:, i]) for i in range(5)])


def modelo_nao_regride(qwk_novo: float, qwk_atual: float, tolerancia: float = 0.01) -> bool:
    """Teste de regressão de desempenho."""
    return qwk_novo >= qwk_atual - tolerancia
