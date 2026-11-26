"""Comparação dos modelos na validação/teste do Essay-BR com intervalos de confiança."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from ..models.baseline import previsoes_b0_b1
from .metrics import COMPS, qwk_por_competencia

N_BOOT = 1000
SEMENTE = 42


def bootstrap(real: np.ndarray, previsoes: dict[str, np.ndarray], indices: np.ndarray):
    """QWK por competência em cada reamostragem."""
    saida = {nome: np.empty((len(indices), 5)) for nome in previsoes}
    for b, idx in enumerate(indices):
        for nome, prev in previsoes.items():
            saida[nome][b] = qwk_por_competencia(real[idx], prev[idx])
    return saida


def resumir(real: np.ndarray, previsoes: dict[str, np.ndarray], titulo: str, caminho_csv: Path):
    n = len(real)
    rng = np.random.default_rng(SEMENTE)
    indices = rng.integers(0, n, size=(N_BOOT, n))
    boot = bootstrap(real, previsoes, indices)

    qwk_linhas, erro_linhas, vies_linhas = [], [], []
    for nome, prev in previsoes.items():
        pontual = qwk_por_competencia(real, prev)
        media_boot = boot[nome].mean(axis=1)
        celulas = {}
        for i, c in enumerate(COMPS):
            lo, hi = np.percentile(boot[nome][:, i], [2.5, 97.5])
            celulas[c] = f"{pontual[i]:.2f} [{lo:.2f}; {hi:.2f}]"
        lo, hi = np.percentile(media_boot, [2.5, 97.5])
        celulas["Média"] = f"{pontual.mean():.2f} [{lo:.2f}; {hi:.2f}]"
        qwk_linhas.append({"modelo": nome, **celulas})
        erro_linhas.append(
            {
                "modelo": nome,
                **{
                    c: round(float(np.abs(real[:, i] - prev[:, i]).mean()), 1)
                    for i, c in enumerate(COMPS)
                },
            }
        )
        vies_linhas.append(
            {
                "modelo": nome,
                **{
                    c: round(float((prev[:, i] - real[:, i]).mean()), 1)
                    for i, c in enumerate(COMPS)
                },
            }
        )

    tab_qwk = pd.DataFrame(qwk_linhas).set_index("modelo")
    tab_erro = pd.DataFrame(erro_linhas).set_index("modelo")
    tab_vies = pd.DataFrame(vies_linhas).set_index("modelo")

    print(f"\n==== {titulo} ({n} redações) ====")
    print("\nQWK [intervalo de confiança de 95%]")
    print(tab_qwk.to_string())
    print("\nErro médio (pontos)")
    print(tab_erro.to_string())
    print("\nViés médio (pontos)")
    print(tab_vies.to_string())

    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        {
            "QWK": tab_qwk,
            "erro_medio": tab_erro.astype(str),
            "vies_medio": tab_vies.astype(str),
        },
        axis=0,
    ).to_csv(caminho_csv, encoding="utf-8-sig")
    print("\nSalvo em", caminho_csv)


def _carregar_llm(caminho: str) -> dict[int, list[int]]:
    notas = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            d = json.loads(linha)
            if d.get("notas"):
                notas[int(d["id"])] = d["notas"]
    return notas


def main() -> int:
    ap = argparse.ArgumentParser(description="Compara B0, B1, BERTimbau e LLMs.")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    df = pd.read_csv(args.data_dir / f"{args.split}.csv").set_index("id")
    y_true = df[COMPS].values
    ids = df.index.to_numpy()
    prev_b0, prev_b1 = previsoes_b0_b1(train, df)

    bert_file = args.results_dir / f"bert_{args.split}_previsoes.csv"
    previsoes: dict[str, np.ndarray] = {
        "B0 (moda)": prev_b0,
        "B1 (atributos)": prev_b1,
    }
    if bert_file.is_file():
        bert = pd.read_csv(bert_file).set_index("id").loc[ids]
        previsoes["BERTimbau"] = bert[COMPS].values.astype(int)

    resumir(
        y_true,
        previsoes,
        f"Avaliação ({args.split})",
        args.results_dir / f"comparacao_{args.split}.csv",
    )

    if args.split == "valid":
        llms = {}
        for caminho in sorted(glob.glob(str(args.results_dir / "zero_shot*.jsonl"))):
            nome = os.path.basename(caminho).replace("zero_shot_", "").replace(".jsonl", "")
            llms[nome] = _carregar_llm(caminho)
        if llms:
            comuns = sorted(set.intersection(*[set(d) for d in llms.values()]))
            pos = [int(np.where(ids == i)[0][0]) for i in comuns]
            sub_prev = {k: v[pos] for k, v in previsoes.items()}
            for nome, d in llms.items():
                sub_prev[nome] = np.array([d[i] for i in comuns])
            resumir(
                y_true[pos],
                sub_prev,
                "Redações em comum com os LLMs",
                args.results_dir / "comparacao_validacao_comum.csv",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
