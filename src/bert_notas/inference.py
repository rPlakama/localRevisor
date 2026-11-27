"""Carregamento lazy e inferência thread-safe com BERTimbau."""
from __future__ import annotations

import threading

from .. import paths
from .validation import COMPETENCIAS, pasta, rotulos_padrao

MAX_TOKENS = 512
DEFAULT_MAX_TOTAL_TOKENS = 2048
WINDOW_OVERLAP = 128

_trava = threading.Lock()  # uma estimativa por vez, mesmo com correções paralelas
_modelo = None
_tokenizador = None
_ordem = list(range(5))


def construir_janelas_com_pesos(
    token_ids: list[int],
    *,
    window_size: int = MAX_TOKENS,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    overlap: int = WINDOW_OVERLAP,
) -> tuple[list[tuple[list[int], int]], bool]:
    """Divide token_ids em janelas sobrepostas e devolve o peso de tokens novos.

    Cada janela recebe peso proporcional apenas aos tokens *novos* (ainda não
    cobertos por janelas anteriores). Isso evita que, em redações pouco maiores
    que 512 tokens, uma janela ancorada no final — que começa no meio de uma
    frase e se sobrepõe em ~90% à primeira — receba o mesmo peso de 50% da
    janela principal (treinada a partir do início da redação).
    """
    content_size = max(1, window_size - 2)
    max_content = max(content_size, max_total_tokens - 2)
    cortada = len(token_ids) > max_content
    usados = token_ids[:max_content]
    if len(usados) <= content_size:
        return [(usados, max(1, len(usados)))], cortada

    effective_overlap = min(overlap, content_size // 4)
    stride = max(1, content_size - effective_overlap)
    janelas: list[tuple[list[int], int]] = []
    inicio = 0
    total = len(usados)
    cobertos_ate = 0
    while inicio < total:
        fim = min(inicio + content_size, total)
        if fim == total:
            # Ancora a última janela no final para manter contexto cheio da conclusão,
            # mas pondera apenas pela parcela ainda não coberta.
            trecho = usados[total - content_size : total]
            peso = max(1, total - cobertos_ate)
            janelas.append((trecho, peso))
            break
        peso = max(1, fim - cobertos_ate)
        janelas.append((usados[inicio:fim], peso))
        cobertos_ate = max(cobertos_ate, fim)
        inicio += stride
    return janelas, cortada


def construir_janelas(
    token_ids: list[int],
    *,
    window_size: int = MAX_TOKENS,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    overlap: int = WINDOW_OVERLAP,
) -> tuple[list[list[int]], bool]:
    """Divide token_ids em janelas sobrepostas que cabem na janela do BERT.

    Devolve ``(janelas, cortada)``. Quando o texto passa de ``window_size`` mas
    cabe em ``max_total_tokens``, todas as partes — inclusive a conclusão — são
    cobertas por janelas sobrepostas e ``cortada`` é ``False``.
    """
    janelas_com_pesos, cortada = construir_janelas_com_pesos(
        token_ids,
        window_size=window_size,
        max_total_tokens=max_total_tokens,
        overlap=overlap,
    )
    return [janela for janela, _ in janelas_com_pesos], cortada


def _mapear_rotulos(modelo) -> list[int]:
    labels = getattr(modelo.config, "id2label", None) or {}
    por_nome: dict[str, int] = {}
    for chave, valor in labels.items():
        nome = str(valor).strip().upper()
        if nome in COMPETENCIAS:
            try:
                por_nome[nome] = int(chave)
            except (TypeError, ValueError):
                continue
    if len(por_nome) == 5:
        return [por_nome[nome] for nome in COMPETENCIAS]
    if labels and not rotulos_padrao({"id2label": labels}):
        raise RuntimeError(
            "O id2label do checkpoint não corresponde a C1–C5; "
            "não é possível mapear as notas com segurança."
        )
    # Checkpoints antigos sem id2label (ou com LABEL_0..4) são treinados na ordem C1..C5.
    return list(range(5))


def _carregar():
    global _modelo, _tokenizador, _ordem
    if _modelo is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        diretorio = pasta()
        _tokenizador = AutoTokenizer.from_pretrained(diretorio)
        modelo, info = AutoModelForSequenceClassification.from_pretrained(
            diretorio, output_loading_info=True
        )
        faltando = [
            chave for chave in info["missing_keys"] if "classifier" in chave
        ]
        if faltando:
            _modelo = None
            raise RuntimeError(
                "O checkpoint não tem a cabeça de classificação treinada "
                f"({', '.join(faltando)}). Use o modelo ajustado, não o BERTimbau base."
            )
        num_labels = getattr(modelo.config, "num_labels", None)
        classificador = getattr(modelo, "classifier", None)
        out_features = getattr(classificador, "out_features", None)
        if num_labels != 5 or out_features != 5:
            _modelo = None
            raise RuntimeError(
                "O checkpoint precisa de 5 saídas (C1 a C5); "
                f"encontrei num_labels={num_labels!r}, out_features={out_features!r}."
            )
        _ordem = _mapear_rotulos(modelo)
        modelo.eval()
        _modelo = modelo
    return _tokenizador, _modelo


def _envolver_tokens_especiais(tokenizador, janela: list[int]) -> list[int]:
    prefixo = (
        tokenizador.cls_token_id
        if getattr(tokenizador, "cls_token_id", None) is not None
        else getattr(tokenizador, "bos_token_id", None)
    )
    sufixo = (
        tokenizador.sep_token_id
        if getattr(tokenizador, "sep_token_id", None) is not None
        else getattr(tokenizador, "eos_token_id", None)
    )
    resultado = list(janela)
    if prefixo is not None:
        resultado.insert(0, int(prefixo))
    if sufixo is not None:
        resultado.append(int(sufixo))
    return resultado


def estimar(
    texto: str, *, max_total_tokens: int | None = None
) -> tuple[dict[str, int], bool]:
    """Devolve ({'C1': 120, ...}, True se passou do limite total de tokens).

    Redações maiores que a janela do modelo (512 tokens) são avaliadas em
    janelas deslizantes sobrepostas até ``max_total_tokens``.
    """
    if not texto or not texto.strip():
        raise ValueError("A redação está vazia.")
    import numpy as np
    import torch

    limite_total = max(
        MAX_TOKENS,
        int(max_total_tokens) if max_total_tokens else paths.BERT_MAX_TOKENS,
    )

    with _trava:
        tokenizador, modelo = _carregar()
        window_size = int(
            getattr(modelo.config, "max_position_embeddings", MAX_TOKENS) or MAX_TOKENS
        )
        token_ids = tokenizador.encode(texto, add_special_tokens=False)
        janelas_com_pesos, cortada = construir_janelas_com_pesos(
            token_ids,
            window_size=window_size,
            max_total_tokens=limite_total,
        )

        if len(janelas_com_pesos) == 1:
            entrada = tokenizador(
                texto, truncation=True, max_length=window_size, return_tensors="pt"
            )
            with torch.no_grad():
                bruto = modelo(**entrada).logits[0].detach().cpu().numpy()
        else:
            predicoes = []
            pesos = []
            with torch.no_grad():
                for janela, peso in janelas_com_pesos:
                    input_ids = _envolver_tokens_especiais(tokenizador, janela)
                    entrada = {
                        "input_ids": torch.tensor([input_ids], dtype=torch.long),
                        "attention_mask": torch.ones(
                            (1, len(input_ids)), dtype=torch.long
                        ),
                    }
                    logits = modelo(**entrada).logits[0].detach().cpu().numpy()
                    predicoes.append(logits)
                    pesos.append(peso)
            bruto = np.average(np.stack(predicoes, axis=0), axis=0, weights=pesos)

        if bruto.shape[0] != 5:
            raise RuntimeError(
                f"O modelo devolveu {bruto.shape[0]} notas; esperava 5 (C1 a C5)."
            )
        bruto = bruto[_ordem] * 200
    notas = np.clip(np.round(bruto / 40) * 40, 0, 200).astype(int)
    return dict(zip(COMPETENCIAS, notas.tolist())), cortada
