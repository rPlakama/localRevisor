"""Estimativa opcional das notas por competência com um BERTimbau ajustado."""
from __future__ import annotations

from .download import HF_REPO_ID, baixar_modelo
from .inference import (
    DEFAULT_MAX_TOTAL_TOKENS,
    MAX_TOKENS,
    construir_janelas,
    construir_janelas_com_pesos,
    estimar,
)
from .validation import (
    COMPETENCIAS,
    ajustado as _ajustado,
    instalado,
    ler_config as _config,
    motivo_indisponivel,
    pasta as _pasta,
    rotulos_padrao as _rotulos_padrao,
    rotulos_validos as _rotulos_validos,
)

__all__ = [
    "COMPETENCIAS",
    "DEFAULT_MAX_TOTAL_TOKENS",
    "HF_REPO_ID",
    "MAX_TOKENS",
    "_ajustado",
    "_config",
    "_pasta",
    "_rotulos_padrao",
    "_rotulos_validos",
    "baixar_modelo",
    "construir_janelas",
    "construir_janelas_com_pesos",
    "estimar",
    "instalado",
    "motivo_indisponivel",
]
