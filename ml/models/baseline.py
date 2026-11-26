"""Baselines B0 (moda) e B1 (atributos linguísticos + Ridge) com persistência."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import DEFAULT_MODELS_DIR
from ..data.features import FEATURE_NAMES, matriz_atributos, split_paragraphs
from ..eval.metrics import COMPS

DEFAULT_BASELINE_DIR = DEFAULT_MODELS_DIR / "baseline_ridge"


def previsoes_b0_b1(train: pd.DataFrame, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Previsões do B0 e do B1 para ``df`` (ajustado apenas no treino)."""
    modas = [int(train[c].mode()[0]) for c in COMPS]
    prev_b0 = np.tile(modas, (len(df), 1))
    X_train = matriz_atributos([split_paragraphs(t) for t in train["texto"]])
    X_df = matriz_atributos([split_paragraphs(t) for t in df["texto"]])
    b1 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        X_train, train[COMPS].values
    )
    prev_b1 = np.clip(np.round(b1.predict(X_df) / 40.0) * 40.0, 0, 200).astype(int)
    return prev_b0, prev_b1


def train_baseline(
    train_df: pd.DataFrame,
    output_dir: Path | None = None,
    *,
    alpha: float = 1.0,
) -> tuple[Pipeline, dict[str, object]]:
    """Treina o modelo B1 (StandardScaler + Ridge) e persiste em ``model.joblib``."""
    dest = output_dir or DEFAULT_BASELINE_DIR
    dest.mkdir(parents=True, exist_ok=True)

    X_train = matriz_atributos([split_paragraphs(t) for t in train_df["texto"]])
    y_train = train_df[COMPS].values
    b0_modes = [int(train_df[c].mode()[0]) for c in COMPS]

    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pipeline.fit(X_train, y_train)

    artifact = {
        "pipeline": pipeline,
        "b0_modes": b0_modes,
        "comps": COMPS,
        "feature_names": FEATURE_NAMES,
        "alpha": alpha,
    }
    joblib.dump(artifact, dest / "model.joblib")

    metadata = {
        "model_type": "baseline_ridge",
        "n_train": int(len(train_df)),
        "alpha": alpha,
        "comps": COMPS,
        "feature_names": FEATURE_NAMES,
        "b0_modes": b0_modes,
    }
    (dest / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    export_baseline_weights(artifact, dest / "weights.json")
    return pipeline, metadata


def export_baseline_weights(artifact: dict[str, object], dest_path: Path) -> Path:
    """Exporta os pesos do scaler + Ridge em JSON para inferência em WebAssembly."""
    pipeline: Pipeline = artifact["pipeline"]  # type: ignore[assignment]
    scaler: StandardScaler = pipeline.named_steps["standardscaler"]
    ridge: Ridge = pipeline.named_steps["ridge"]
    payload = {
        "version": 1,
        "model_type": "baseline_ridge",
        "comps": list(artifact["comps"]),  # type: ignore[arg-type]
        "feature_names": list(artifact["feature_names"]),  # type: ignore[arg-type]
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [[float(x) for x in row] for row in ridge.coef_],
        "intercept": [float(v) for v in ridge.intercept_],
    }
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return dest_path


def load_baseline(model_dir: Path | None = None) -> dict[str, object]:
    """Carrega o artefato ``model.joblib`` treinado."""
    source = (model_dir or DEFAULT_BASELINE_DIR) / "model.joblib"
    if not source.is_file():
        raise FileNotFoundError(
            f"Modelo baseline não encontrado em {source}. "
            "Execute `python -m ml.pipeline train --model baseline` primeiro."
        )
    return joblib.load(source)


def predict_baseline(texts: list[str], model_dir: Path | None = None) -> np.ndarray:
    """Gera previsões C1–C5 (múltiplos de 40 entre 0 e 200) para uma lista de textos."""
    if not texts:
        return np.empty((0, len(COMPS)), dtype=int)
    artifact = load_baseline(model_dir)
    pipeline: Pipeline = artifact["pipeline"]  # type: ignore[assignment]
    X = matriz_atributos([split_paragraphs(t) for t in texts])
    raw = pipeline.predict(X)
    return np.clip(np.round(raw / 40.0) * 40.0, 0, 200).astype(int)
