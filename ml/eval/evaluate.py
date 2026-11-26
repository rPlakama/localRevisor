"""Tabela de avaliação por competência e geração de relatórios."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DEFAULT_RESULTS_DIR
from .metrics import COMPS, qwk


def avaliar(y_real: np.ndarray, y_prev: np.ndarray) -> pd.DataFrame:
    """y_real e y_prev: matrizes (n redações x 5) com notas de 0 a 200."""
    linhas = []
    for i, nome in enumerate(COMPS):
        real, prev = y_real[:, i], y_prev[:, i]
        linhas.append(
            {
                "competencia": nome,
                "QWK": qwk(real, prev),
                "erro_medio": float(np.abs(real - prev).mean()),
                "acerto_aprox": float((np.abs(real - prev) <= 40).mean()),
                "vies_medio": float((prev - real).mean()),
            }
        )
    return pd.DataFrame(linhas).round(3)


def evaluate_predictions(
    y_real: np.ndarray, y_prev: np.ndarray
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calcula tabela por competência e resumo agregado (QWK médio, MAE, acurácia ±40)."""
    table = avaliar(y_real, y_prev)
    total_real = y_real.sum(axis=1)
    total_prev = y_prev.sum(axis=1)
    summary = {
        "n_samples": int(len(y_real)),
        "mean_qwk": round(float(table["QWK"].mean()), 4),
        "mean_mae": round(float(table["erro_medio"].mean()), 2),
        "mean_approx_acc": round(float(table["acerto_aprox"].mean()), 4),
        "mean_bias": round(float(table["vies_medio"].mean()), 2),
        "total_mae": round(float(np.abs(total_real - total_prev).mean()), 2),
        "per_competency": table.to_dict(orient="records"),
    }
    return table, summary


def save_evaluation_report(
    summary: dict[str, object],
    table: pd.DataFrame,
    *,
    model_name: str,
    split_name: str,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Grava relatórios CSV e JSON em ``ml/resultados/``."""
    dest = output_dir or DEFAULT_RESULTS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / f"eval_{model_name}_{split_name}.csv"
    json_path = dest / f"eval_{model_name}_{split_name}.json"
    table.to_csv(csv_path, index=False, encoding="utf-8")
    payload = {"model": model_name, "split": split_name, **summary}
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return csv_path, json_path
