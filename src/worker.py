"""Execução das correções em segundo plano, com concorrência limitada."""
from __future__ import annotations

import logging
import os
import threading
from uuid import uuid4

from . import db, files, grading, paths

LOGGER = logging.getLogger(__name__)


def _max_workers() -> int:
    try:
        value = int(os.environ.get("LOCALREVISOR_MAX_WORKERS", "2"))
    except ValueError:
        return 2
    return max(1, min(value, 8))


_active: set[int] = set()
_active_lock = threading.Lock()
_semaphore = threading.BoundedSemaphore(_max_workers())


def is_active(correction_id: int) -> bool:
    with _active_lock:
        return correction_id in _active


def active_count() -> int:
    with _active_lock:
        return len(_active)


def start(
    correction_id: int,
    theme: str,
    stored_name: str,
    *,
    bert_only: bool = False,
    client_scores: dict[str, int] | None = None,
) -> bool:
    """Agenda uma correção. Devolve False se ela já está na fila/execução."""
    with _active_lock:
        if correction_id in _active:
            return False
        _active.add(correction_id)
    thread = threading.Thread(
        target=_run,
        args=(correction_id, theme, stored_name, bert_only, client_scores),
        name=f"correcao-{correction_id}",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError:
        with _active_lock:
            _active.discard(correction_id)
        raise
    return True


def _run(
    correction_id: int,
    theme: str,
    stored_name: str,
    bert_only: bool = False,
    client_scores: dict[str, int] | None = None,
) -> None:
    try:
        with _semaphore:
            run_correction(
                correction_id,
                theme,
                stored_name,
                bert_only=bert_only,
                client_scores=client_scores,
            )
    finally:
        with _active_lock:
            _active.discard(correction_id)


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, grading.GradingError):
        return str(exc)
    return (
        "Falha inesperada na correção. "
        f"Detalhe técnico: {type(exc).__name__}: {exc}"
    )


def run_correction(
    correction_id: int,
    theme: str,
    stored_name: str,
    *,
    bert_only: bool = False,
    client_scores: dict[str, int] | None = None,
) -> None:
    """Executa uma correção de ponta a ponta. Também é usada nos testes."""
    row = db.get_correction(correction_id)
    if row is None:
        return
    path = paths.UPLOAD_DIR / stored_name
    kind = files.kind_from_name(stored_name)
    try:
        if not path.exists():
            raise grading.GradingError(
                "O arquivo original não foi encontrado. Envie a redação novamente."
            )
        settings = db.get_settings()
        session_id = uuid4().hex
        text = grading.obtain_text(
            path, kind, settings, session_id=session_id
        )
        review_kwargs: dict[str, object] = {"session_id": session_id}
        if bert_only:
            review_kwargs["bert_only"] = True
        if client_scores is not None:
            review_kwargs["client_scores"] = client_scores
        result = grading.review(theme, text, settings, **review_kwargs)
        db.finish_correction(
            correction_id,
            note=result.note,
            scores=result.scores,
            explanation=result.explanation,
            errors=result.errors,
            source_text=result.source_text,
            score_mode=result.score_mode,
            model=result.model,
            provider=result.provider,
            prompt_version=result.prompt_version,
            duration_ms=result.duration_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            truncated=result.truncated,
        )
    except Exception as exc:  # nunca deixar a linha presa em "corrigindo"
        LOGGER.exception("Falha na correção %s", correction_id)
        db.fail_correction(correction_id, _friendly_error(exc))
