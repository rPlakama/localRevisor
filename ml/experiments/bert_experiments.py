"""Experimentos com o BERTimbau: generalização para temas novos e estabilidade."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from ..eval.metrics import COMPS, qwk_por_competencia
from ..models.bert import liberar, predict_bert_df, train_bert


def escolher_temas_novos(todos: pd.DataFrame, repeticao: int, minimo_redacoes: int = 600):
    rng = np.random.default_rng(100 + repeticao)
    temas = sorted(todos["tema"].unique())
    novos, total = set(), 0
    for tema in rng.permutation(temas):
        if total >= minimo_redacoes:
            break
        novos.add(tema)
        total += int((todos["tema"] == tema).sum())
    return novos


def temas_novos(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    repeticoes: int = 3,
    minimo_redacoes: int = 600,
    saida: Path = DEFAULT_RESULTS_DIR / "temas_novos_resultado.csv",
) -> None:
    todos = pd.concat([train, valid], ignore_index=True)
    linhas = []
    for rep in range(repeticoes):
        novos = escolher_temas_novos(todos, rep, minimo_redacoes)
        treino_t = train[~train["tema"].isin(novos)]
        vistos = valid[~valid["tema"].isin(novos)]
        novos_df = todos[todos["tema"].isin(novos)]
        modelo, _ = train_bert(treino_t, semente=rep)
        q_vistos = qwk_por_competencia(vistos[COMPS].values, predict_bert_df(modelo, vistos))
        q_novos = qwk_por_competencia(novos_df[COMPS].values, predict_bert_df(modelo, novos_df))
        linhas.append(
            {"repeticao": rep + 1, "grupo": "temas vistos", **dict(zip(COMPS, q_vistos)), "media": q_vistos.mean()}
        )
        linhas.append(
            {"repeticao": rep + 1, "grupo": "temas novos", **dict(zip(COMPS, q_novos)), "media": q_novos.mean()}
        )
        del modelo
        liberar()
    res = pd.DataFrame(linhas).round(3)
    saida.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(saida, index=False)


def sementes(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    lista: tuple[int, ...] = (0, 1, 2),
    saida: Path = DEFAULT_RESULTS_DIR / "sementes_resultado.csv",
) -> None:
    linhas = []
    for s in lista:
        modelo, _ = train_bert(train, semente=s)
        prev = predict_bert_df(modelo, valid)
        q = qwk_por_competencia(valid[COMPS].values, prev)
        erro = np.abs(valid[COMPS].values - prev).mean(axis=0)
        linhas.append(
            {
                "semente": s,
                **{f"QWK_{c}": v for c, v in zip(COMPS, q)},
                "QWK_media": q.mean(),
                "erro_medio": erro.mean(),
            }
        )
        del modelo
        liberar()
    res = pd.DataFrame(linhas).round(3)
    saida.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(saida, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Experimentos com o BERTimbau.")
    ap.add_argument("experimento", choices=["temas-novos", "sementes"])
    ap.add_argument("--treino", type=Path, default=DEFAULT_DATA_DIR / "train.csv")
    ap.add_argument("--validacao", type=Path, default=DEFAULT_DATA_DIR / "valid.csv")
    args = ap.parse_args()
    train, valid = pd.read_csv(args.treino), pd.read_csv(args.validacao)
    if args.experimento == "temas-novos":
        temas_novos(train, valid)
    else:
        sementes(train, valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
