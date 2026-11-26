"""Pipeline de Machine Learning do localRevisor (ENEM C1–C5)."""
from __future__ import annotations

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
ROOT_DIR = ML_DIR.parent
DEFAULT_DATA_DIR = ML_DIR / "dados"
DEFAULT_RESULTS_DIR = ML_DIR / "resultados"
DEFAULT_MODELS_DIR = ROOT_DIR / "modelos"
