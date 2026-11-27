from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src import bert_notas


class BertValidationTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("BERT_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        os.environ["BERT_DIR"] = str(self.dir)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("BERT_DIR", None)
        else:
            os.environ["BERT_DIR"] = self._old
        self.tmp.cleanup()

    def write_config(self, **extra):
        config = {"architectures": ["BertForMaskedLM"], **extra}
        (self.dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_missing_dir(self):
        os.environ["BERT_DIR"] = str(self.dir / "nao-existe")
        self.assertIn("não foi encontrada", bert_notas.motivo_indisponivel())

    def test_base_model_is_rejected(self):
        self.write_config()
        self.assertFalse(bert_notas._ajustado())
        self.assertIn("BERTimbau base", bert_notas.motivo_indisponivel())

    def test_two_outputs_rejected(self):
        self.write_config(
            architectures=["BertForSequenceClassification"], num_labels=2
        )
        self.assertFalse(bert_notas._ajustado())
        self.assertIn("5 saídas", bert_notas.motivo_indisponivel())

    def test_five_outputs_accepted(self):
        self.write_config(
            architectures=["BertForSequenceClassification"], num_labels=5
        )
        self.assertTrue(bert_notas._ajustado())
        reason = bert_notas.motivo_indisponivel()
        if reason:
            # Em ambientes sem torch/transformers, o motivo correto é dependência.
            self.assertIn("dependências", reason)

    def test_id2label_accepts_five_competencies(self):
        self.write_config(
            architectures=["BertForSequenceClassification"],
            id2label={"0": "C1", "1": "C2", "2": "C3", "3": "C4", "4": "C5"},
        )
        self.assertTrue(bert_notas._ajustado())

    def test_wrong_id2label_rejected(self):
        self.write_config(
            architectures=["BertForSequenceClassification"],
            num_labels=5,
            id2label={"0": "A", "1": "B", "2": "C", "3": "D", "4": "E"},
        )
        self.assertFalse(bert_notas._ajustado())
        self.assertFalse(bert_notas._rotulos_validos(bert_notas._config()))
        self.assertIn("5 saídas", bert_notas.motivo_indisponivel())

    def test_default_labels_accepted(self):
        self.write_config(
            architectures=["BertForSequenceClassification"],
            id2label={
                "0": "LABEL_0",
                "1": "LABEL_1",
                "2": "LABEL_2",
                "3": "LABEL_3",
                "4": "LABEL_4",
            },
        )
        self.assertTrue(bert_notas._ajustado())

    def test_invalid_json(self):
        (self.dir / "config.json").write_text("{", encoding="utf-8")
        self.assertFalse(bert_notas._ajustado())
        self.assertIn("config.json", bert_notas.motivo_indisponivel())

    def test_baixar_modelo_uses_valid_repo_id(self):
        destino = self.dir / "novo_modelo"
        os.environ["BERT_DIR"] = str(destino)
        download = mock.Mock()
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = download
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            bert_notas.baixar_modelo()
        self.assertTrue(destino.is_dir())
        download.assert_called_once_with(
            repo_id="berghilde/bertimbau-enem-notas",
            local_dir=str(destino),
        )

    def test_construir_janelas_short_text_single_window(self):
        tokens = list(range(300))
        janelas, cortada = bert_notas.construir_janelas(
            tokens, window_size=512, max_total_tokens=2048
        )
        self.assertFalse(cortada)
        self.assertEqual(len(janelas), 1)
        self.assertEqual(janelas[0], tokens)

    def test_construir_janelas_long_essay_covers_end_without_truncation(self):
        tokens = list(range(700))
        janelas, cortada = bert_notas.construir_janelas(
            tokens, window_size=512, max_total_tokens=2048
        )
        self.assertFalse(cortada)
        self.assertEqual(len(janelas), 2)
        self.assertEqual(len(janelas[0]), 510)
        self.assertEqual(len(janelas[-1]), 510)
        self.assertEqual(janelas[0][0], 0)
        self.assertEqual(janelas[-1][-1], 699)

    def test_construir_janelas_marks_truncated_only_above_max_total_tokens(self):
        tokens = list(range(2500))
        janelas, cortada = bert_notas.construir_janelas(
            tokens, window_size=512, max_total_tokens=2048
        )
        self.assertTrue(cortada)
        self.assertGreaterEqual(len(janelas), 4)
        self.assertEqual(janelas[-1][-1], 2045)

    def test_construir_janelas_com_pesos_weights_by_uncovered_tokens(self):
        # 545 tokens: janela 0 cobre 510 tokens novos; janela 1 (ancorada no fim)
        # acrescenta apenas 35 tokens inéditos e não deve ter peso 510 (50%).
        janelas_545, cortada = bert_notas.construir_janelas_com_pesos(
            list(range(545)), window_size=512, max_total_tokens=2048
        )
        self.assertFalse(cortada)
        self.assertEqual([peso for _, peso in janelas_545], [510, 35])

        # 1000 tokens com effective_overlap = 510 // 4 = 127 e stride = 383:
        # [0:510], [383:893], [490:1000] -> pesos novos [510, 383, 107] (soma 1000).
        janelas_1000, cortada = bert_notas.construir_janelas_com_pesos(
            list(range(1000)), window_size=512, max_total_tokens=2048
        )
        self.assertFalse(cortada)
        self.assertEqual([peso for _, peso in janelas_1000], [510, 383, 107])

    def test_default_max_total_tokens_matches_documented_default(self):
        self.assertEqual(bert_notas.DEFAULT_MAX_TOTAL_TOKENS, 2048)


if __name__ == "__main__":
    unittest.main()
