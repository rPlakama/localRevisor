"""Treino e previsão do BERTimbau para notas C1–C5 do ENEM."""
from __future__ import annotations

import copy
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DEFAULT_MODELS_DIR
from ..eval.metrics import COMPS, qwk_por_competencia

MODELO_BASE = "neuralmind/bert-base-portuguese-cased"
MAX_LEN, BATCH, LR = 512, 8, 2e-5
DEFAULT_BERT_DIR = DEFAULT_MODELS_DIR / "bertimbau_enem"

_tokenizador = None


def _get_tokenizer(model_name_or_path: str = MODELO_BASE):
    global _tokenizador
    if _tokenizador is None or model_name_or_path != MODELO_BASE:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name_or_path)
        if model_name_or_path == MODELO_BASE:
            _tokenizador = tok
        return tok
    return _tokenizador


def train_bert(
    df_treino: pd.DataFrame,
    *,
    semente: int = 42,
    epocas: int = 4,
    df_validacao: pd.DataFrame | None = None,
    output_dir: Path | None = None,
):
    """Treina o BERTimbau e opcionalmente salva em ``output_dir``."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    usar_amp = dispositivo == "cuda"
    tok = _get_tokenizer()

    class Redacoes(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.enc = tok(df["texto"].tolist(), truncation=True, max_length=MAX_LEN)
            self.y = (df[COMPS].values / 200.0).astype("float32")

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, i: int):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    def agrupar(lote):
        entradas = tok.pad(
            [{k: v for k, v in x.items() if k != "labels"} for x in lote],
            return_tensors="pt",
        )
        entradas["labels"] = torch.tensor(np.stack([x["labels"] for x in lote]))
        return entradas

    torch.manual_seed(semente)
    np.random.seed(semente)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE, num_labels=5, problem_type="regression", use_safetensors=False
    ).to(dispositivo)
    dl = DataLoader(Redacoes(df_treino), batch_size=BATCH, shuffle=True, collate_fn=agrupar)
    total = len(dl) * epocas
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)
    agenda = get_linear_schedule_with_warmup(otimizador, int(0.1 * total), total)
    escala = torch.amp.GradScaler("cuda", enabled=usar_amp)

    melhor, melhores_pesos, historico = -1.0, None, []
    for epoca in range(epocas):
        modelo.train()
        soma = 0.0
        for lote in tqdm(dl, desc=f"Época {epoca + 1}/{epocas}", leave=False):
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            otimizador.zero_grad()
            with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=usar_amp):
                perda = modelo(**lote).loss
            escala.scale(perda).backward()
            escala.unscale_(otimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escala.step(otimizador)
            escala.update()
            agenda.step()
            soma += perda.item()
        linha = {"epoca": epoca + 1, "perda_treino": round(soma / max(len(dl), 1), 4)}
        if df_validacao is not None:
            prev = predict_bert_df(modelo, df_validacao)
            qwk_val = qwk_por_competencia(df_validacao[COMPS].values, prev).mean()
            linha["qwk_medio_validacao"] = round(float(qwk_val), 3)
            if qwk_val > melhor:
                melhor = qwk_val
                melhores_pesos = copy.deepcopy(
                    {k: v.cpu() for k, v in modelo.state_dict().items()}
                )
        historico.append(linha)
        print(linha)

    if melhores_pesos is not None:
        modelo.load_state_dict(melhores_pesos)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        modelo.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)

    return modelo, historico


def predict_bert_df(modelo, df: pd.DataFrame) -> np.ndarray:
    """Notas previstas (n x 5, múltiplos de 40 entre 0 e 200) para um DataFrame."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    dispositivo = next(modelo.parameters()).device.type
    usar_amp = dispositivo == "cuda"
    tok = _get_tokenizer()

    class RedacoesSemLabel(Dataset):
        def __init__(self, textos: list[str]):
            self.enc = tok(textos, truncation=True, max_length=MAX_LEN)

        def __len__(self) -> int:
            return len(self.enc["input_ids"])

        def __getitem__(self, i: int):
            return {k: v[i] for k, v in self.enc.items()}

    def agrupar(lote):
        return tok.pad(lote, return_tensors="pt")

    modelo.eval()
    saidas = []
    with torch.no_grad():
        for lote in DataLoader(
            RedacoesSemLabel(df["texto"].tolist()), batch_size=32, collate_fn=agrupar
        ):
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=usar_amp):
                saidas.append(modelo(**lote).logits.float().cpu().numpy())
    continuo = np.concatenate(saidas) * 200.0
    return np.clip(np.round(continuo / 40.0) * 40.0, 0, 200).astype(int)


def predict_bert_texts(
    texts: list[str], model_dir: Path | None = None
) -> np.ndarray:
    """Carrega o checkpoint salvo em ``model_dir`` e prevê C1–C5 para ``texts``."""
    import torch
    from transformers import AutoModelForSequenceClassification

    source = model_dir or DEFAULT_BERT_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint BERT não encontrado em {source}")
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = AutoModelForSequenceClassification.from_pretrained(source).to(dispositivo)
    global _tokenizador
    _tokenizador = _get_tokenizer(str(source))
    return predict_bert_df(modelo, pd.DataFrame({"texto": texts}))


def liberar() -> None:
    """Libera memória da GPU entre treinos."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
