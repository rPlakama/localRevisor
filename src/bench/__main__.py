"""Interface CLI para `localrevisor --debug benchredação <bert|bert_ai|ai>`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import VALID_SPLITS
from .runner import run_benchmark

_MODE_ALIASES = {
    "bert": "bert",
    "bert_ai": "bert_ai",
    "bert+ai": "bert_ai",
    "bert_llm": "bert_ai",
    "ai": "ai",
    "llm": "ai",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    cleaned = [arg for arg in argv if arg not in {"benchredação", "benchredacao"}]
    parser = argparse.ArgumentParser(
        prog="localrevisor --debug benchredação",
        description="Avalia o corretor em redações reais (Essay-BR) usando a configuração atual.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=sorted(_MODE_ALIASES),
        help="modo do benchmark: 'bert' (só BERT), 'bert_ai' (BERT + IA configurada) ou 'ai' (só IA)",
    )
    parser.add_argument(
        "--bert-only",
        "--only-bert",
        dest="bert_only",
        action="store_true",
        help="usa somente a pontuação do BERT (ignora chamada de IA mesmo em bert_ai/ai)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="mostra o detalhamento C1–C5, tokens e avisos por redação",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="quantidade de redações estratificadas (padrão: 15 para bert, 5 para bert_ai/ai)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="semente de amostragem estratificada (padrão: 42)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV local opcional (Essay-BR ou formato id,tema,texto,C1..C5)",
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="auto",
        help=(
            "recorte de temas no Essay-BR: 'auto' (padrão: só temas >85 fora do treino "
            "no extended-corpus), 'unseen' (>85), 'all' (todos) ou 'seen' (<=85)"
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="não baixar o Essay-BR automaticamente se o cache local não existir",
    )
    args = parser.parse_args(cleaned)
    if args.bert_only:
        args.mode = "bert"
    elif args.mode is None:
        parser.error("informe o modo (bert, bert_ai, ai) ou use --bert-only")
    else:
        args.mode = _MODE_ALIASES[args.mode]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        records = run_benchmark(
            mode=args.mode,
            limit=args.limit,
            seed=args.seed,
            csv_path=args.csv,
            debug=args.debug,
            allow_download=not args.no_download,
            split=args.split,
        )
    except Exception as exc:
        print(f"Erro no benchredação: {exc}", file=sys.stderr)
        return 1
    return 0 if any(not rec.error for rec in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
