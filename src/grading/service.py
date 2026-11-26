"""Orquestração da correção, transcrição de imagem/PDF e leitura de redações."""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from openai import OpenAI

from .. import bert_notas, files, paths
from .models import GradingError, ModelOutputError, ReviewResult
from .parser import join_errors, parse_hybrid, parse_review, verify_quotes
from .prompts import (
    DEFAULT_MAX_TOKENS,
    HYBRID_PROMPT,
    PROMPT_VERSION_HYBRID,
    PROMPT_VERSION_REVIEW,
    REVIEW_PROMPT,
    TRANSCRIBE_MAX_TOKENS,
    TRANSCRIBE_PROMPT,
)
from .provider import api_client, create_completion, provider_name, repair_completion

LOGGER = logging.getLogger(__name__)


def _parse_response(
    content: str,
    *,
    mode: str,
    scores: dict[str, int],
    truncated: bool,
    text: str,
) -> tuple[int, dict[str, int], str, str]:
    if mode in {"bert+llm", "bert_wasm+llm"}:
        return parse_hybrid(content, scores, truncated, text)
    note, parsed_scores, explanation, errors = parse_review(content)
    quote_warning = verify_quotes(explanation, text)
    if quote_warning:
        errors = join_errors(errors, quote_warning)
    return note, parsed_scores, explanation, errors


def _bert_max_tokens(settings: dict[str, str]) -> int:
    raw = (settings.get("bert_max_tokens") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return paths.BERT_MAX_TOKENS
    return max(bert_notas.MAX_TOKENS, min(16384, value))


def review(
    theme: str,
    text: str,
    settings: dict[str, str],
    *,
    client: OpenAI | None = None,
    session_id: str | None = None,
    bert_only: bool = False,
    client_scores: dict[str, int] | None = None,
    client_explanation: str | None = None,
    client_errors: str = "",
    client_descriptor_model: str | None = None,
) -> ReviewResult:
    if not text.strip():
        raise GradingError("A redação está vazia.")
    started = time.monotonic()

    if client_scores is not None and client_explanation:
        return ReviewResult(
            note=sum(client_scores.values()),
            scores=dict(client_scores),
            explanation=client_explanation.strip(),
            errors=(client_errors or "").strip(),
            source_text=text,
            score_mode="bert_wasm+local_descriptor",
            model=client_descriptor_model or "wasm-local-descriptor",
            provider="browser",
            prompt_version="local-descriptor",
            duration_ms=int((time.monotonic() - started) * 1000),
            tokens_in=0,
            tokens_out=0,
            truncated=False,
        )

    if bert_only:
        if client_scores is not None:
            scores = dict(client_scores)
            truncated = False
            score_mode = "bert_wasm"
            model_name = "wasm-client"
            provider = "browser"
            prompt_ver = "wasm-only"
            note_source = "pelo modelo local no seu navegador (WebAssembly)"
        else:
            motivo = bert_notas.motivo_indisponivel()
            if motivo:
                raise GradingError(
                    "O modo Somente BERT não pode ser usado: " + motivo
                )
            scores, truncated = bert_notas.estimar(
                text, max_total_tokens=_bert_max_tokens(settings)
            )
            score_mode = "bert"
            model_name = "bertimbau-enem"
            provider = "local"
            prompt_ver = "bert-only"
            note_source = "pelo modelo BERTimbau local"
        linhas = [
            f"{name}: {value}/200" for name, value in scores.items()
        ]
        linhas.append(
            f"\nAvaliação feita somente {note_source} "
            "(sem explicação por IA)."
        )
        if truncated:
            linhas.append(
                "Aviso: a redação é longa e o modelo de notas leu apenas o início do texto."
            )
        return ReviewResult(
            note=sum(scores.values()),
            scores=dict(scores),
            explanation="\n".join(linhas),
            errors="",
            source_text=text,
            score_mode=score_mode,
            model=model_name,
            provider=provider,
            prompt_version=prompt_ver,
            duration_ms=int((time.monotonic() - started) * 1000),
            tokens_in=0,
            tokens_out=0,
            truncated=truncated,
        )

    model = (settings.get("correction_model") or "").strip()
    if not model:
        raise GradingError("Nenhum modelo de correção configurado.")
    tokens_in = tokens_out = 0

    if settings.get("use_bert") == "1":
        if client_scores is not None:
            scores = dict(client_scores)
            truncated = False
            mode = "bert_wasm+llm"
        else:
            motivo = bert_notas.motivo_indisponivel()
            if motivo:
                raise GradingError(
                    "O modo BERT está ativado, mas não pode ser usado: " + motivo
                )
            scores, truncated = bert_notas.estimar(
                text, max_total_tokens=_bert_max_tokens(settings)
            )
            mode = "bert+llm"
        scores_text = "\n".join(f"{name}: {value}" for name, value in scores.items())
        messages = [
            {"role": "system", "content": HYBRID_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<tema>\n{theme}\n</tema>\n\n"
                    f"Notas iniciais de referência (BERT local, sem leitura do tema):\n"
                    f"{scores_text}\n\n"
                    f"<redacao>\n{text}\n</redacao>"
                ),
            },
        ]
        prompt_version = PROMPT_VERSION_HYBRID
    else:
        messages = [
            {"role": "system", "content": REVIEW_PROMPT},
            {
                "role": "user",
                "content": f"<tema>\n{theme}\n</tema>\n\n<redacao>\n{text}\n</redacao>",
            },
        ]
        prompt_version = PROMPT_VERSION_REVIEW
        mode = "llm"
        scores = {}
        truncated = False

    if client is None:
        client = api_client(settings, session_id)

    content, tin, tout = create_completion(
        client,
        model=model,
        messages=messages,
        max_tokens=DEFAULT_MAX_TOKENS,
        json_mode=True,
    )
    tokens_in += tin
    tokens_out += tout

    try:
        note, scores, explanation, errors = _parse_response(
            content, mode=mode, scores=scores, truncated=truncated, text=text
        )
    except ModelOutputError as first_error:
        schema = (
            '{"scores": {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0},'
            ' "note": 0, "explanation": "...", "errors": []}'
            if mode == "llm"
            else '{"scores": {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0},'
            ' "explanation": {"C1": "...", "C2": "...", "C3": "...",'
            ' "C4": "...", "C5": "..."}, "errors": []}'
        )
        repaired, tin, tout = repair_completion(
            client,
            model=model,
            messages=messages,
            invalid_content=content,
            schema=schema,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        tokens_in += tin
        tokens_out += tout
        try:
            note, scores, explanation, errors = _parse_response(
                repaired, mode=mode, scores=scores, truncated=truncated, text=text
            )
        except ModelOutputError as second_error:
            raise ModelOutputError(
                f"O modelo não seguiu o contrato JSON ({first_error}); "
                f"a tentativa de reparo também falhou ({second_error})."
            ) from second_error

    duration_ms = int((time.monotonic() - started) * 1000)
    return ReviewResult(
        note=note,
        scores=scores,
        explanation=explanation,
        errors=errors,
        source_text=text,
        score_mode=mode,
        model=model,
        provider=provider_name(settings.get("base_url", "")),
        prompt_version=prompt_version,
        duration_ms=duration_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        truncated=truncated,
    )


def transcribe(
    path: Path,
    kind: str,
    settings: dict[str, str],
    *,
    client: OpenAI | None = None,
    session_id: str | None = None,
) -> str:
    model = (settings.get("image_model") or "").strip()
    if not model:
        raise GradingError(
            "Nenhum modelo de imagem configurado. Defina um em Configurações "
            "para transcrever fotos ou PDFs escaneados."
        )
    converted: Path | None = None
    try:
        suffix = path.suffix.lower()
        if kind == "pdf":
            source = path
        elif suffix in {".webp", ".gif"}:
            converted = files.convert_to_png(path)
            source = converted
        else:
            source = path
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    finally:
        if converted is not None:
            converted.unlink(missing_ok=True)
    mime = files.mime_for(source)
    if client is None:
        client = api_client(settings, session_id)
    messages = [
        {"role": "system", "content": TRANSCRIBE_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            ],
        },
    ]
    try:
        content, _, _ = create_completion(
            client,
            model=model,
            messages=messages,
            max_tokens=TRANSCRIBE_MAX_TOKENS,
            json_mode=False,
        )
    except GradingError as exc:
        raise GradingError(
            f"Falha ao transcrever {path.name}: {exc}. "
            "PDFs escaneados podem não ser aceitos por todos os modelos de visão; "
            "tente enviar a página como imagem."
        ) from exc
    transcription = content.strip()
    if not transcription:
        raise GradingError("A transcrição veio vazia.")
    return transcription


def obtain_text(
    path: Path,
    kind: str,
    settings: dict[str, str],
    *,
    client: OpenAI | None = None,
    session_id: str | None = None,
) -> str:
    if kind == "text":
        text = files.read_text(path)
    elif kind == "pdf":
        try:
            text = files.extract_pdf_text(path)
        except Exception as exc:  # PDF protegido/corrompido: tenta visão
            LOGGER.warning("Falha ao extrair texto de %s: %s", path.name, exc)
            text = ""
        if not text:
            text = transcribe(
                path, kind, settings, client=client, session_id=session_id
            )
    else:
        text = transcribe(path, kind, settings, client=client, session_id=session_id)
    if not text.strip():
        raise GradingError("Não foi possível extrair texto do arquivo.")
    return text
