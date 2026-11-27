"""Executor do benchmark de redações usando as configurações reais do app."""
from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

from .. import bert_notas, db, grading, paths
from .dataset import (
    EssaySample,
    ensure_essay_br,
    filter_by_split,
    load_samples,
    sample_stratified,
)
from .metrics import BenchRecord, format_summary

VALID_MODES = ("bert", "bert_ai", "ai")


def _bert_limit_from(settings: dict[str, str]) -> int:
    try:
        value = int((settings.get("bert_max_tokens") or "").strip())
    except ValueError:
        return paths.BERT_MAX_TOKENS
    return max(bert_notas.MAX_TOKENS, min(16384, value))


def _resolve_samples(
    *,
    csv_path: Path | None,
    limit: int,
    seed: int,
    allow_download: bool,
    split: str = "auto",
) -> tuple[list[EssaySample], str, bool]:
    using_extended = False
    if csv_path is not None:
        prompts_guess = csv_path.parent / "prompts.csv"
        all_samples = load_samples(
            csv_path, prompts_guess if prompts_guess.is_file() else None
        )
    else:
        default_valid = paths.BASE_DIR / "ml" / "dados" / "valid.csv"
        if default_valid.is_file():
            all_samples = load_samples(default_valid)
        else:
            essays_csv, prompts_csv = ensure_essay_br(allow_download=allow_download)
            all_samples = load_samples(essays_csv, prompts_csv)
            using_extended = True
    if not all_samples:
        raise RuntimeError("Nenhuma redação válida encontrada para o benchmark.")
    filtered, split_info, leak_risk = filter_by_split(
        all_samples, split, default_unseen=using_extended
    )
    if not filtered:
        raise RuntimeError(
            f"Nenhuma redação encontrada no recorte {split!r}. "
            "Tente --split all ou informe outro --csv."
        )
    return sample_stratified(filtered, limit, seed=seed), split_info, leak_risk


def _evaluate_sample(
    sample: EssaySample,
    *,
    mode: str,
    settings: dict[str, str],
    bert_limit: int,
) -> BenchRecord:
    started = time.monotonic()
    try:
        if mode == "bert":
            scores, truncated = bert_notas.estimar(
                sample.text, max_total_tokens=bert_limit
            )
            duration_ms = int((time.monotonic() - started) * 1000)
            return BenchRecord(
                sample_id=sample.sample_id,
                theme=sample.theme,
                true_scores=sample.scores,
                pred_scores=scores,
                true_total=sample.total,
                pred_total=sum(scores.values()),
                duration_ms=duration_ms,
                truncated=truncated,
            )

        run_settings = dict(settings)
        run_settings["use_bert"] = "1" if mode == "bert_ai" else "0"
        run_settings["bert_max_tokens"] = str(bert_limit)
        result = grading.review(sample.theme, sample.text, run_settings)
        return BenchRecord(
            sample_id=sample.sample_id,
            theme=sample.theme,
            true_scores=sample.scores,
            pred_scores=result.scores,
            true_total=sample.total,
            pred_total=result.note,
            duration_ms=result.duration_ms,
            truncated=result.truncated,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            quote_warning="citação não encontrada" in (result.errors or "").lower(),
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        return BenchRecord(
            sample_id=sample.sample_id,
            theme=sample.theme,
            true_scores=sample.scores,
            pred_scores={},
            true_total=sample.total,
            pred_total=0,
            duration_ms=duration_ms,
            error=str(exc),
        )


def _format_row(index: int, total: int, rec: BenchRecord, *, debug: bool) -> str:
    short_theme = (rec.theme[:42] + "…") if len(rec.theme) > 43 else rec.theme
    if rec.error:
        return (
            f"[{index:02d}/{total:02d}] id={rec.sample_id} real={rec.true_total:4d} "
            f"| ERRO: {rec.error}"
        )
    diff = rec.pred_total - rec.true_total
    base = (
        f"[{index:02d}/{total:02d}] id={rec.sample_id:<5} "
        f"real={rec.true_total:4d} prev={rec.pred_total:4d} ({diff:+4d}) "
        f"{rec.duration_ms / 1000:5.2f}s | {short_theme}"
    )
    if not debug:
        return base
    comps = " ".join(
        f"{c}:{rec.true_scores[c]}->{rec.pred_scores[c]}"
        for c in bert_notas.COMPETENCIAS
    )
    extras: list[str] = []
    if rec.truncated:
        extras.append("TRUNCADA")
    if rec.quote_warning:
        extras.append("AVISO_CITACAO")
    if rec.tokens_in or rec.tokens_out:
        extras.append(f"tok={rec.tokens_in}/{rec.tokens_out}")
    suffix = f" [{' '.join(extras)}]" if extras else ""
    return f"{base}\n        {comps}{suffix}"


def run_benchmark(
    *,
    mode: str,
    limit: int | None = None,
    seed: int = 42,
    csv_path: Path | None = None,
    debug: bool = False,
    allow_download: bool = True,
    split: str = "auto",
) -> list[BenchRecord]:
    if mode not in VALID_MODES:
        raise ValueError(f"Modo inválido: {mode!r}. Use: {', '.join(VALID_MODES)}.")

    load_dotenv(paths.ENV_PATH)
    db.init_db()
    settings = db.get_settings()
    bert_limit = _bert_limit_from(settings)

    if mode in ("bert", "bert_ai"):
        motivo = bert_notas.motivo_indisponivel()
        if motivo:
            raise RuntimeError(f"BERT indisponível: {motivo}")

    if mode in ("bert_ai", "ai"):
        if not (settings.get("correction_model") or "").strip():
            raise RuntimeError(
                "Nenhum modelo de correção configurado no app. "
                "Abra /config, conecte o provedor e salve um modelo."
            )

    effective_limit = limit if limit is not None else (15 if mode == "bert" else 5)
    samples, split_info, leak_risk = _resolve_samples(
        csv_path=csv_path,
        limit=effective_limit,
        seed=seed,
        allow_download=allow_download,
        split=split,
    )
    leak_warning = leak_risk and mode in ("bert", "bert_ai")

    if mode == "bert":
        model_info = f"BERT local ({bert_notas._pasta()})"
    else:
        provider = grading.provider_name(settings.get("base_url", ""))
        model_info = (
            f"{settings.get('correction_model')} @ {provider} "
            f"({grading.PROMPT_VERSION_HYBRID if mode == 'bert_ai' else grading.PROMPT_VERSION_REVIEW})"
        )

    print(
        f"Iniciando benchredação [{mode}] — {len(samples)} redação(ões) "
        f"({model_info} | {split_info})"
    )
    records: list[BenchRecord] = []
    for idx, sample in enumerate(samples, start=1):
        rec = _evaluate_sample(
            sample, mode=mode, settings=settings, bert_limit=bert_limit
        )
        records.append(rec)
        print(_format_row(idx, len(samples), rec, debug=debug), flush=True)

    print(
        format_summary(
            mode,
            records,
            model_info=model_info,
            bert_limit=bert_limit,
            split_info=split_info,
            leak_warning=leak_warning,
        )
    )
    return records
