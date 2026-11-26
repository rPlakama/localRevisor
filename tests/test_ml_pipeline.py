"""Testes da pipeline automatizada de ML (collect, preprocess, train, evaluate, predict)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.pipeline import main as pipeline_main, predict_essay


def _write_synthetic_essay_br(path: Path, n_rows: int = 30) -> Path:
    rows = []
    for i in range(n_rows):
        theme = f"{i % 6}"
        score = [80, 120, 160][i % 3]
        paras = [
            f"Introdução da redação {i}. O tema {theme} exige reflexão crítica.",
            f"Portanto, além disso, o argumento {i} demonstra a relevância social, pois há impacto direto.",
            f"Dessa forma, cabe ao Estado agir por meio de políticas públicas para que haja mudança {i}.",
        ]
        rows.append(
            {
                "prompt": theme,
                "title": f"Título {i}",
                "essay": repr(paras),
                "c1": score,
                "c2": score,
                "c3": score,
                "c4": score,
                "c5": score,
                "score": score * 5,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def test_ml_pipeline_end_to_end_and_predict(tmp_path: Path) -> None:
    raw_csv = _write_synthetic_essay_br(tmp_path / "source.csv", n_rows=36)
    data_dir = tmp_path / "dados"
    model_dir = tmp_path / "modelo_ridge"
    results_dir = tmp_path / "resultados"

    rc = pipeline_main(
        [
            "run",
            "--model",
            "baseline",
            "--csv",
            str(raw_csv),
            "--no-download",
            "--data-dir",
            str(data_dir),
            "--model-dir",
            str(model_dir),
            "--results-dir",
            str(results_dir),
            "--seed",
            "42",
        ]
    )
    assert rc == 0

    for split in ("raw", "train", "valid", "test"):
        assert (data_dir / f"{split}.csv").is_file()

    assert (model_dir / "model.joblib").is_file()
    assert (model_dir / "weights.json").is_file()
    assert (model_dir / "metadata.json").is_file()

    eval_json = results_dir / "eval_baseline_valid.json"
    assert eval_json.is_file()
    report = json.loads(eval_json.read_text(encoding="utf-8"))
    assert report["model"] == "baseline"
    assert report["n_samples"] > 0

    sample_essay = (
        "A educação é fundamental para o desenvolvimento do país.\n\n"
        "Portanto, além disso, investimentos contínuos são necessários, pois ampliam oportunidades.\n\n"
        "Dessa forma, cabe ao governo federal investir por meio de programas para que todos tenham acesso."
    )
    pred = predict_essay(sample_essay, model_type="baseline", model_dir=model_dir)
    assert set(pred["scores"]) == {"C1", "C2", "C3", "C4", "C5"}
    for val in pred["scores"].values():
        assert 0 <= val <= 200
        assert val % 40 == 0
    assert pred["total"] == sum(pred["scores"].values())
