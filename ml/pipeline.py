"""CLI unificado da pipeline de Machine Learning do localRevisor.

Etapas automatizadas:
  1. collect     — Coleta/carga do corpus Essay-BR (local ou download) -> raw.csv
  2. preprocess  — Limpeza, validação de notas e divisão train/valid/test
  3. train       — Treino do modelo (baseline Ridge ou BERTimbau)
  4. evaluate    — Avaliação (QWK, MAE, acerto ±40, viés) e relatórios
  5. run         — Executa etapas 1 → 4 em sequência
  6. predict     — Gera predições C1–C5 coerentes para novas redações
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from .data.collect import collect_dataset
from .data.preprocess import preprocess_dataset
from .eval.evaluate import evaluate_predictions, save_evaluation_report
from .eval.metrics import COMPS
from .models.baseline import DEFAULT_BASELINE_DIR, predict_baseline, train_baseline
from .models.bert import DEFAULT_BERT_DIR, predict_bert_texts, train_bert


def _default_model_dir(model_type: str) -> Path:
    return DEFAULT_BASELINE_DIR if model_type == "baseline" else DEFAULT_BERT_DIR


def predict_essay(
    text: str,
    *,
    model_type: str = "baseline",
    model_dir: Path | None = None,
) -> dict[str, object]:
    """Prevê notas C1–C5 (múltiplos de 40 entre 0 e 200) e nota total para uma redação."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("O texto da redação está vazio.")

    target_dir = model_dir or _default_model_dir(model_type)
    if model_type == "baseline":
        preds = predict_baseline([cleaned], model_dir=target_dir)[0]
    elif model_type == "bert":
        preds = predict_bert_texts([cleaned], model_dir=target_dir)[0]
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    scores = {comp: int(preds[i]) for i, comp in enumerate(COMPS)}
    return {
        "model": model_type,
        "scores": scores,
        "total": int(sum(scores.values())),
    }


def _cmd_collect(args: argparse.Namespace) -> int:
    out = collect_dataset(
        csv_path=args.csv,
        corpus_dir=args.corpus,
        output_path=args.output,
        allow_download=not args.no_download,
    )
    print(f"[collect] Dataset bruto salvo em {out}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    splits = preprocess_dataset(
        raw_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    for name, path in splits.items():
        rows = len(pd.read_csv(path))
        print(f"[preprocess] {name}: {rows} redações -> {path}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    train_df = pd.read_csv(data_dir / "train.csv")
    output_dir = args.output_dir or _default_model_dir(args.model)

    if args.model == "baseline":
        _, meta = train_baseline(train_df, output_dir=output_dir, alpha=args.alpha)
        print(
            f"[train] Baseline Ridge treinado ({meta['n_train']} redações) -> {output_dir}"
        )
    else:
        valid_path = data_dir / "valid.csv"
        valid_df = pd.read_csv(valid_path) if valid_path.is_file() else None
        train_bert(
            train_df,
            semente=args.seed,
            epocas=args.epochs,
            df_validacao=valid_df,
            output_dir=output_dir,
        )
        print(f"[train] BERTimbau treinado -> {output_dir}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    split_path = data_dir / f"{args.split}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split não encontrado: {split_path}")
    df = pd.read_csv(split_path)
    model_dir = args.model_dir or _default_model_dir(args.model)

    if args.model == "baseline":
        preds = predict_baseline(df["texto"].tolist(), model_dir=model_dir)
    else:
        preds = predict_bert_texts(df["texto"].tolist(), model_dir=model_dir)

    table, summary = evaluate_predictions(df[COMPS].values, preds)
    csv_path, json_path = save_evaluation_report(
        summary,
        table,
        model_name=args.model,
        split_name=args.split,
        output_dir=args.output_dir or DEFAULT_RESULTS_DIR,
    )
    print(f"\n[evaluate] {args.model} ({args.split}, n={summary['n_samples']}):")
    print(table.to_string(index=False))
    print(
        f"QWK médio: {summary['mean_qwk']:.4f} | MAE médio: {summary['mean_mae']:.2f} | "
        f"Acerto ±40: {summary['mean_approx_acc']:.3f}"
    )
    print(f"Relatórios salvos em {csv_path} e {json_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    raw_path = collect_dataset(
        csv_path=args.csv,
        corpus_dir=args.corpus,
        output_path=data_dir / "raw.csv",
        allow_download=not args.no_download,
    )
    print(f"[1/4 collect] {raw_path}")

    splits = preprocess_dataset(raw_path=raw_path, output_dir=data_dir, seed=args.seed)
    print(
        "[2/4 preprocess] "
        + ", ".join(f"{k}={len(pd.read_csv(v))}" for k, v in splits.items())
    )

    model_dir = args.model_dir or _default_model_dir(args.model)
    train_df = pd.read_csv(splits["train"])
    valid_df = pd.read_csv(splits["valid"])
    if args.model == "baseline":
        train_baseline(train_df, output_dir=model_dir)
        preds = predict_baseline(valid_df["texto"].tolist(), model_dir=model_dir)
    else:
        modelo, _ = train_bert(
            train_df,
            semente=args.seed,
            epocas=args.epochs,
            df_validacao=valid_df,
            output_dir=model_dir,
        )
        from .models.bert import predict_bert_df

        preds = predict_bert_df(modelo, valid_df)
    print(f"[3/4 train] Modelo salvo em {model_dir}")

    table, summary = evaluate_predictions(valid_df[COMPS].values, preds)
    csv_path, json_path = save_evaluation_report(
        summary,
        table,
        model_name=args.model,
        split_name="valid",
        output_dir=args.results_dir or DEFAULT_RESULTS_DIR,
    )
    print(
        f"[4/4 evaluate] QWK médio={summary['mean_qwk']:.4f} | "
        f"MAE={summary['mean_mae']:.2f} -> {json_path}"
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        if str(args.file) == "-":
            text = sys.stdin.read()
        else:
            text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    result = predict_essay(text, model_type=args.model, model_dir=args.model_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        scores = result["scores"]
        parts = " | ".join(f"{k}: {v}" for k, v in scores.items())  # type: ignore[union-attr]
        print(f"Modelo: {result['model']} | {parts} | Total: {result['total']}/1000")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-pipeline",
        description="Pipeline automatizada de ML do localRevisor (ENEM C1–C5).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Coleta/carga do corpus Essay-BR.")
    p_collect.add_argument("--csv", type=Path, default=None)
    p_collect.add_argument("--corpus", type=Path, default=None)
    p_collect.add_argument("--output", type=Path, default=None)
    p_collect.add_argument("--no-download", action="store_true")
    p_collect.set_defaults(func=_cmd_collect)

    p_prep = sub.add_parser("preprocess", help="Pré-processamento e split.")
    p_prep.add_argument("--input", type=Path, default=None)
    p_prep.add_argument("--output-dir", type=Path, default=None)
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.set_defaults(func=_cmd_preprocess)

    p_train = sub.add_parser("train", help="Treinamento do modelo.")
    p_train.add_argument("--model", choices=["baseline", "bert"], default="baseline")
    p_train.add_argument("--data-dir", type=Path, default=None)
    p_train.add_argument("--output-dir", type=Path, default=None)
    p_train.add_argument("--alpha", type=float, default=1.0)
    p_train.add_argument("--epochs", type=int, default=4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Avaliação em validação ou teste.")
    p_eval.add_argument("--model", choices=["baseline", "bert"], default="baseline")
    p_eval.add_argument("--split", choices=["valid", "test"], default="valid")
    p_eval.add_argument("--data-dir", type=Path, default=None)
    p_eval.add_argument("--model-dir", type=Path, default=None)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_run = sub.add_parser("run", help="Executa collect -> preprocess -> train -> evaluate.")
    p_run.add_argument("--model", choices=["baseline", "bert"], default="baseline")
    p_run.add_argument("--csv", type=Path, default=None)
    p_run.add_argument("--corpus", type=Path, default=None)
    p_run.add_argument("--no-download", action="store_true")
    p_run.add_argument("--data-dir", type=Path, default=None)
    p_run.add_argument("--model-dir", type=Path, default=None)
    p_run.add_argument("--results-dir", type=Path, default=None)
    p_run.add_argument("--epochs", type=int, default=4)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.set_defaults(func=_cmd_run)

    p_pred = sub.add_parser("predict", help="Gera predições C1–C5 para uma redação.")
    p_pred.add_argument("--model", choices=["baseline", "bert"], default="baseline")
    p_pred.add_argument("--model-dir", type=Path, default=None)
    p_pred.add_argument("--text", type=str, default=None)
    p_pred.add_argument("--file", type=Path, default=None)
    p_pred.add_argument("--json", action="store_true")
    p_pred.set_defaults(func=_cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
