""""Estimativa opcional das notas por competência com o BERTimbau ajustado.

O modo BERT é ligado e desligado na tela de Configurações do app. Este módulo informa se ele
pode ser usado (pasta do modelo e dependências presentes) e faz a estimativa das notas.
A pasta do modelo é, por padrão, modelos/bertimbau_enem na raiz do projeto (pode ser trocada
com a variável de ambiente BERT_DIR). O torch e o transformers só são importados quando o
modelo é realmente usado.
"""
import importlib.util
import os
import threading
from pathlib import Path

COMPETENCIAS = ["C1", "C2", "C3", "C4", "C5"]
MAX_TOKENS = 512

_trava = threading.Lock()  # uma estimativa por vez, mesmo com várias correções em paralelo
_modelo = None
_tokenizador = None


def _pasta():
    padrao = Path(__file__).resolve().parent.parent / "modelos" / "bertimbau_enem"
    return Path(os.environ.get("BERT_DIR", padrao))


def motivo_indisponivel():
    """Explica por que o modo BERT não pode ser usado, ou devolve None se está tudo pronto."""
    if not _pasta().is_dir():
        return "a pasta do modelo não foi encontrada (veja a seção Modo BERT do README)."
    faltando = [
        nome
        for nome in ("torch", "transformers", "numpy")
        if importlib.util.find_spec(nome) is None
    ]
    if faltando:
        return (
            "faltam dependências (" + ", ".join(faltando) + "). "
            "Rode: pip install -r requirements-bert.txt"
        )
    return None


def instalado():
    """True se a pasta do modelo e as dependências estão presentes."""
    return motivo_indisponivel() is None


def _carregar():
    global _modelo, _tokenizador
    if _modelo is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizador = AutoTokenizer.from_pretrained(_pasta())
        _modelo = AutoModelForSequenceClassification.from_pretrained(_pasta())
        _modelo.eval()
    return _tokenizador, _modelo


def estimar(texto):
    """Devolve ({'C1': 120, ...}, True se o texto passou de 512 pedaços e foi cortado)."""
    import numpy as np
    import torch

    with _trava:
        tokenizador, modelo = _carregar()
        cortada = len(tokenizador.encode(texto)) > MAX_TOKENS
        entrada = tokenizador(
            texto, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        )
        with torch.no_grad():
            bruto = modelo(**entrada).logits[0].numpy() * 200
    notas = np.clip(np.round(bruto / 40) * 40, 0, 200).astype(int)
    return dict(zip(COMPETENCIAS, notas.tolist())), cortada