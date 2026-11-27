"""Métricas de avaliação (MAE, viés, tolerância e QWK) e formatação de relatório."""
from __future__ import annotations

from dataclasses import dataclass

from ..bert_notas import COMPETENCIAS


@dataclass(frozen=True)
class BenchRecord:
    sample_id: str
    theme: str
    true_scores: dict[str, int]
    pred_scores: dict[str, int]
    true_total: int
    pred_total: int
    duration_ms: int
    truncated: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    quote_warning: bool = False


def quadratic_weighted_kappa(
    y_true: list[int], y_pred: list[int], *, step: int = 40, max_val: int = 200
) -> float:
    """Calcula QWK (Quadratic Weighted Kappa) em faixas discretas do ENEM."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    levels = max_val // step + 1
    n = len(y_true)
    observed = [[0] * levels for _ in range(levels)]
    hist_true = [0] * levels
    hist_pred = [0] * levels

    for true_v, pred_v in zip(y_true, y_pred):
        i = max(0, min(levels - 1, round(true_v / step)))
        j = max(0, min(levels - 1, round(pred_v / step)))
        observed[i][j] += 1
        hist_true[i] += 1
        hist_pred[j] += 1

    denom_sq = (levels - 1) ** 2
    if denom_sq == 0:
        return 0.0

    num = 0.0
    den = 0.0
    for i in range(levels):
        for j in range(levels):
            weight = ((i - j) ** 2) / denom_sq
            expected = (hist_true[i] * hist_pred[j]) / n
            num += weight * observed[i][j]
            den += weight * expected
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return 1.0 - (num / den)


def summarize(records: list[BenchRecord]) -> dict[str, dict[str, float]]:
    ok = [rec for rec in records if not rec.error]
    if not ok:
        return {}

    stats: dict[str, dict[str, float]] = {}
    for comp in COMPETENCIAS:
        y_true = [rec.true_scores[comp] for rec in ok]
        y_pred = [rec.pred_scores[comp] for rec in ok]
        diffs = [p - t for t, p in zip(y_true, y_pred)]
        abs_diffs = [abs(d) for d in diffs]
        within_40 = sum(1 for d in abs_diffs if d <= 40) / len(ok) * 100
        stats[comp] = {
            "mae": sum(abs_diffs) / len(ok),
            "bias": sum(diffs) / len(ok),
            "within": within_40,
            "qwk": quadratic_weighted_kappa(y_true, y_pred, step=40, max_val=200),
        }

    y_true_t = [rec.true_total for rec in ok]
    y_pred_t = [rec.pred_total for rec in ok]
    diffs_t = [p - t for t, p in zip(y_true_t, y_pred_t)]
    abs_t = [abs(d) for d in diffs_t]
    stats["TOTAL"] = {
        "mae": sum(abs_t) / len(ok),
        "bias": sum(diffs_t) / len(ok),
        "within": sum(1 for d in abs_t if d <= 80) / len(ok) * 100,
        "qwk": quadratic_weighted_kappa(y_true_t, y_pred_t, step=40, max_val=1000),
    }
    return stats


def format_summary(
    mode: str,
    records: list[BenchRecord],
    *,
    model_info: str,
    bert_limit: int,
    split_info: str = "",
    leak_warning: bool = False,
) -> str:
    ok = [rec for rec in records if not rec.error]
    failed = [rec for rec in records if rec.error]
    lines = [
        "",
        f"=== Resumo benchredação ({mode}) ===",
        f"Configuração: {model_info} | BERT max_tokens={bert_limit}",
    ]
    if split_info:
        lines.append(f"Recorte: {split_info}")
    if leak_warning:
        lines.append(
            "Aviso: este recorte inclui temas <= 85 do Essay-BR v1, que contêm "
            "redações do treino do BERT (métricas podem ficar infladas). "
            "Use --split unseen para avaliar apenas temas inéditos."
        )
    lines.append(
        f"Amostras: {len(ok)}/{len(records)} concluídas ({len(failed)} erro(s))"
    )
    if not ok:
        return "\n".join(lines)

    stats = summarize(ok)
    lines.append("")
    lines.append(f"{'Métrica':<8} {'MAE':>7} {'Viés':>8} {'±40/80':>8} {'QWK':>7}")
    lines.append("-" * 42)
    for key in (*COMPETENCIAS, "TOTAL"):
        item = stats[key]
        lines.append(
            f"{key:<8} {item['mae']:7.1f} {item['bias']:+8.1f} "
            f"{item['within']:7.1f}% {item['qwk']:7.3f}"
        )

    avg_s = sum(rec.duration_ms for rec in ok) / len(ok) / 1000
    truncated_n = sum(1 for rec in ok if rec.truncated)
    lines.append("-" * 42)
    lines.append(f"Tempo médio: {avg_s:.2f}s | Cortadas pelo BERT: {truncated_n}/{len(ok)}")
    total_in = sum(rec.tokens_in for rec in ok)
    total_out = sum(rec.tokens_out for rec in ok)
    if total_in or total_out:
        quote_warns = sum(1 for rec in ok if rec.quote_warning)
        lines.append(
            f"Tokens totais: {total_in} entrada / {total_out} saída "
            f"| Avisos de citação: {quote_warns}/{len(ok)}"
        )
    return "\n".join(lines)
