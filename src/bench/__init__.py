"""Benchmark de redações reais (Essay-BR) para BERT e BERT+IA."""
from __future__ import annotations

from .dataset import (
    ESSAY_BR_V1_MAX_PROMPT_ID,
    VALID_SPLITS,
    EssaySample,
    filter_by_split,
    load_samples,
    sample_stratified,
)
from .metrics import BenchRecord, format_summary, quadratic_weighted_kappa, summarize
from .runner import VALID_MODES, run_benchmark

__all__ = [
    "BenchRecord",
    "ESSAY_BR_V1_MAX_PROMPT_ID",
    "EssaySample",
    "VALID_MODES",
    "VALID_SPLITS",
    "filter_by_split",
    "format_summary",
    "load_samples",
    "quadratic_weighted_kappa",
    "run_benchmark",
    "sample_stratified",
    "summarize",
]
