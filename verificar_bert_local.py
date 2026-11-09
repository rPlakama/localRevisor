"""Confere se o modelo exportado dá as mesmas notas que as previsões do Colab.
Uso: python verificar_bert_local.py CAMINHO\valid.csv CAMINHO\bert_valid_previsoes.csv [N]
"""
import os
import sys

import numpy as np
import pandas as pd

os.environ["USE_BERT"] = "1"
from src import bert_notas

COMPS = ["C1", "C2", "C3", "C4", "C5"]


def notas(texto):
    resultado = bert_notas.estimar(texto)[0]
    return [resultado[c] for c in COMPS]


valid = pd.read_csv(sys.argv[1]).set_index("id")
colab = pd.read_csv(sys.argv[2]).set_index("id")
n = int(sys.argv[3]) if len(sys.argv) > 3 else 50

amostra = valid.sample(n=n, random_state=42)
local = np.array([notas(t) for t in amostra["texto"]])
colab_notas = colab.loc[amostra.index, COMPS].values
professor = amostra[COMPS].values

print("Notas idênticas ao Colab:", round((local == colab_notas).mean() * 100, 1), "%")
print("Diferença média (pontos):", round(np.abs(local - colab_notas).mean(), 2))
print("Erro médio vs professor, local:", round(np.abs(local - professor).mean(), 1))
print("Erro médio vs professor, Colab:", round(np.abs(colab_notas - professor).mean(), 1))