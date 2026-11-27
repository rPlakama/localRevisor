"""Carga, cache local e amostragem estratificada de redações reais (Essay-BR)."""
from __future__ import annotations

import ast
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import httpx

from .. import paths
from ..bert_notas import COMPETENCIAS

ESSAY_BR_BASE_URL = (
    "https://raw.githubusercontent.com/lplnufpi/essay-br/main/extended-corpus"
)
ESSAYS_FILENAME = "extended_essay-br.csv"
PROMPTS_FILENAME = "prompts.csv"
# Prompts 0..85 compõem o Essay-BR original (usado no treino/validação/teste do BERT);
# prompts > 85 foram adicionados no extended-corpus e nunca entraram no treino.
ESSAY_BR_V1_MAX_PROMPT_ID = 85
VALID_SPLITS = ("auto", "unseen", "all", "seen")


@dataclass(frozen=True)
class EssaySample:
    sample_id: str
    theme: str
    text: str
    scores: dict[str, int]
    total: int
    prompt_id: int | None = None


def bench_dir() -> Path:
    return paths.DATA_DIR / "bench"


def _parse_essay_text(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return "\n\n".join(str(p).strip() for p in parsed if str(p).strip())
    return raw


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in resp.iter_bytes():
                handle.write(chunk)
    tmp.replace(dest)


def ensure_essay_br(*, allow_download: bool = True) -> tuple[Path, Path | None]:
    """Garante os CSVs do Essay-BR em data/bench/ (fora do Git)."""
    target_dir = bench_dir()
    essays_path = target_dir / ESSAYS_FILENAME
    prompts_path = target_dir / PROMPTS_FILENAME
    if essays_path.is_file():
        return essays_path, prompts_path if prompts_path.is_file() else None
    if not allow_download:
        raise FileNotFoundError(
            f"Corpus não encontrado em {essays_path}. Informe --csv ou permita download."
        )
    print(f"Baixando Essay-BR para {target_dir} (cache local, fora do Git)...")
    _download_file(f"{ESSAY_BR_BASE_URL}/{PROMPTS_FILENAME}", prompts_path)
    _download_file(f"{ESSAY_BR_BASE_URL}/{ESSAYS_FILENAME}", essays_path)
    return essays_path, prompts_path


def _load_prompts(prompts_path: Path | None) -> dict[str, str]:
    if prompts_path is None or not prompts_path.is_file():
        return {}
    mapping: dict[str, str] = {}
    with prompts_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pid = str(row.get("id", "")).strip()
            title = str(row.get("title", "")).strip()
            if pid and title:
                mapping[pid] = title
    return mapping


def _extract_scores(row: dict[str, str]) -> dict[str, int] | None:
    scores: dict[str, int] = {}
    for comp in COMPETENCIAS:
        raw = row.get(comp) if comp in row else row.get(comp.lower())
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = int(round(float(str(raw).strip())))
        except ValueError:
            return None
        if not 0 <= value <= 200:
            return None
        scores[comp] = value
    return scores


def load_samples(
    csv_path: Path, prompts_path: Path | None = None
) -> list[EssaySample]:
    """Lê redações nos formatos do Essay-BR ou de ml/preparar_dados.py."""
    prompts = _load_prompts(prompts_path)
    samples: list[EssaySample] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            raw_text = row.get("essay") if "essay" in row else row.get("texto", "")
            text = _parse_essay_text(str(raw_text or ""))
            if not text:
                continue
            scores = _extract_scores(row)
            if scores is None:
                continue
            prompt_key = str(row.get("prompt") or row.get("tema") or "").strip()
            prompt_id = int(prompt_key) if prompt_key.isdigit() else None
            theme = (
                prompts.get(prompt_key)
                or (prompt_key if prompt_key and not prompt_key.isdigit() else "")
                or str(row.get("title") or "").strip()
                or f"Tema #{prompt_key or index}"
            )
            sample_id = str(row.get("id") or index)
            samples.append(
                EssaySample(
                    sample_id=sample_id,
                    theme=theme,
                    text=text,
                    scores=scores,
                    total=sum(scores.values()),
                    prompt_id=prompt_id,
                )
            )
    return samples


def filter_by_split(
    samples: list[EssaySample],
    split: str = "auto",
    *,
    default_unseen: bool = False,
) -> tuple[list[EssaySample], str, bool]:
    """Filtra amostras por recorte de temas para evitar vazamento do treino do BERT.

    Devolve ``(amostras, descricao_split, risco_vazamento_treino)``.
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Split inválido: {split!r}. Use: {', '.join(VALID_SPLITS)}."
        )
    has_unseen = any(
        s.prompt_id is not None and s.prompt_id > ESSAY_BR_V1_MAX_PROMPT_ID
        for s in samples
    )
    has_seen = any(
        s.prompt_id is not None and s.prompt_id <= ESSAY_BR_V1_MAX_PROMPT_ID
        for s in samples
    )

    effective = split
    if split == "auto":
        effective = "unseen" if (default_unseen and has_unseen) else "all"

    if effective == "unseen":
        filtered = [
            s
            for s in samples
            if s.prompt_id is not None and s.prompt_id > ESSAY_BR_V1_MAX_PROMPT_ID
        ]
        return (
            filtered,
            f"unseen (temas > {ESSAY_BR_V1_MAX_PROMPT_ID}, fora do treino do BERT)",
            False,
        )
    if effective == "seen":
        filtered = [
            s
            for s in samples
            if s.prompt_id is not None and s.prompt_id <= ESSAY_BR_V1_MAX_PROMPT_ID
        ]
        return (
            filtered,
            f"seen (temas <= {ESSAY_BR_V1_MAX_PROMPT_ID}, inclui treino do BERT)",
            bool(filtered),
        )
    return (
        list(samples),
        "all (todos os temas)"
        if default_unseen
        else "arquivo local (sem filtro de tema)",
        bool(default_unseen and has_seen),
    )


def sample_stratified(
    samples: list[EssaySample], limit: int, *, seed: int = 42
) -> list[EssaySample]:
    """Seleciona amostras distribuídas entre faixas de nota (baixas a ~1000)."""
    if limit <= 0 or len(samples) <= limit:
        return list(samples)

    bands = [
        (0, 399),
        (400, 599),
        (600, 759),
        (760, 879),
        (880, 1000),
    ]
    rng = random.Random(seed)
    buckets: list[list[EssaySample]] = []
    for low, high in bands:
        bucket = [s for s in samples if low <= s.total <= high]
        rng.shuffle(bucket)
        if bucket:
            buckets.append(bucket)

    selected: list[EssaySample] = []
    while len(selected) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
    selected.sort(key=lambda item: item.total)
    return selected
