from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src import grading


def response(content: str, finish: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


class FakeCompletions:
    def __init__(self, *contents: str | None):
        self.contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        if content is None:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None), finish_reason="stop"
                    )
                ],
                usage=None,
            )
        return response(content)


class FakeClient:
    def __init__(self, *contents: str | None):
        self.completions = FakeCompletions(*contents)
        self.chat = SimpleNamespace(completions=self.completions)


SETTINGS = {
    "base_url": "http://localhost:11434/v1",
    "correction_model": "fake",
    "image_model": "fake",
    "use_bert": "0",
}


class ParsingTests(unittest.TestCase):
    def test_review_valid(self):
        content = (
            '{"scores": {"C1": 120, "C2": 160, "C3": 120, "C4": 120, "C5": 80},'
            ' "note": 600, "explanation": "ok", "errors": ["um erro"]}'
        )
        note, scores, explanation, errors = grading.parse_review(content)
        self.assertEqual(note, 600)
        self.assertEqual(scores["C2"], 160)
        self.assertEqual(explanation, "ok")
        self.assertIn("um erro", errors)

    def test_review_fenced_json(self):
        content = (
            "```json\n"
            '{"scores": {"C1": 40, "C2": 40, "C3": 40, "C4": 40, "C5": 40},'
            ' "explanation": "ok", "errors": []}\n'
            "```"
        )
        note, scores, _, _ = grading.parse_review(content)
        self.assertEqual(note, 200)
        self.assertEqual(set(scores), set(grading.COMPETENCIAS))

    def test_review_rejects_out_of_range(self):
        content = (
            '{"scores": {"C1": 999, "C2": 40, "C3": 40, "C4": 40, "C5": 40},'
            ' "explanation": "ok", "errors": []}'
        )
        with self.assertRaises(grading.ModelOutputError):
            grading.parse_review(content)

    def test_review_requires_all_competencies(self):
        content = '{"scores": {"C1": 120}, "explanation": "ok", "errors": []}'
        with self.assertRaises(grading.ModelOutputError):
            grading.parse_review(content)

    def test_review_rounds_to_enem_step(self):
        content = (
            '{"scores": {"C1": 119, "C2": 121, "C3": 139, "C4": 141, "C5": 200},'
            ' "explanation": "ok", "errors": []}'
        )
        _, scores, _, _ = grading.parse_review(content)
        self.assertEqual(scores, {"C1": 120, "C2": 120, "C3": 120, "C4": 160, "C5": 200})

    def test_hybrid_uses_bert_scores_and_warns_truncation(self):
        content = (
            '{"explanation": {"C1": "bom", "C2": "bom", "C3": "bom", "C4": "bom",'
            ' "C5": "bom"}, "errors": []}'
        )
        scores = {"C1": 120, "C2": 120, "C3": 120, "C4": 120, "C5": 120}
        note, result_scores, explanation, errors = grading.parse_hybrid(
            content, scores, True, "texto"
        )
        self.assertEqual(note, 600)
        self.assertEqual(result_scores, scores)
        self.assertIn("leu apenas o início", explanation)
        self.assertIn("Aviso", explanation)

    def test_hybrid_allows_llm_to_calibrate_bert_scores(self):
        content = (
            '{"scores": {"C1": 120, "C2": 40, "C3": 80, "C4": 120, "C5": 80},'
            ' "explanation": {"C1": "ok", "C2": "tangenciou o tema", "C3": "raso",'
            ' "C4": "ok", "C5": "faltam elementos"}, "errors": ["tangenciamento"]}'
        )
        bert_scores = {"C1": 120, "C2": 160, "C3": 120, "C4": 120, "C5": 120}
        note, result_scores, explanation, _ = grading.parse_hybrid(
            content, bert_scores, False, "texto"
        )
        self.assertEqual(
            result_scores, {"C1": 120, "C2": 40, "C3": 80, "C4": 120, "C5": 80}
        )
        self.assertEqual(note, 440)
        self.assertIn("C2 (40/200, ref. BERT: 160)", explanation)
        self.assertIn("C1 (120/200):", explanation)

    def test_quote_warning(self):
        errors = grading.verify_quotes(
            'O texto diz "uma frase que não existe no original".',
            "outra coisa completamente diferente",
        )
        self.assertIn("não encontrada", errors)

    def test_extract_json_rejects_empty_and_prose(self):
        with self.assertRaises(grading.ModelOutputError):
            grading.extract_json_object("")
        with self.assertRaises(grading.ModelOutputError):
            grading.extract_json_object("sem json aqui")


class ReviewTests(unittest.TestCase):
    def test_review_with_fake_client(self):
        client = FakeClient(
            '{"scores": {"C1": 120, "C2": 160, "C3": 120, "C4": 120, "C5": 80},'
            ' "note": 600, "explanation": "ok", "errors": []}'
        )
        result = grading.review("Tema", "Redação.", SETTINGS, client=client)
        self.assertEqual(result.note, 600)
        self.assertEqual(result.score_mode, "llm")
        self.assertEqual(result.model, "fake")
        self.assertEqual(result.tokens_in, 10)
        self.assertEqual(result.source_text, "Redação.")

    def test_review_repairs_invalid_json(self):
        client = FakeClient(
            "não é json",
            '{"scores": {"C1": 40, "C2": 40, "C3": 40, "C4": 40, "C5": 40},'
            ' "explanation": "corrigido", "errors": []}',
        )
        result = grading.review("Tema", "Redação.", SETTINGS, client=client)
        self.assertEqual(result.note, 200)
        self.assertEqual(result.explanation, "corrigido")
        self.assertEqual(len(client.completions.calls), 2)

    def test_review_rejects_empty_content(self):
        client = FakeClient(None)
        with self.assertRaises(grading.ModelOutputError):
            grading.review("Tema", "Redação.", SETTINGS, client=client)

    def test_review_requires_model(self):
        settings = dict(SETTINGS, correction_model="")
        with self.assertRaises(grading.GradingError):
            grading.review("Tema", "Redação.", settings, client=FakeClient("{}"))

    def test_bert_fails_closed(self):
        settings = dict(SETTINGS, use_bert="1")
        with mock.patch.object(
            grading.bert_notas, "motivo_indisponivel", return_value="sem checkpoint"
        ):
            with self.assertRaises(grading.GradingError):
                grading.review("Tema", "Redação.", settings, client=FakeClient("{}"))

    def test_review_retries_when_truncated_by_length(self):
        valid = (
            '{"scores": {"C1": 120, "C2": 120, "C3": 120, "C4": 120, "C5": 120},'
            ' "note": 600, "explanation": "ok", "errors": []}'
        )
        responses = [response("", finish="length"), response(valid, finish="stop")]
        calls: list[dict] = []

        class LengthClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        calls.append(kwargs)
                        return responses.pop(0)

        result = grading.review("Tema", "Redação.", SETTINGS, client=LengthClient())
        self.assertEqual(result.note, 600)
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1]["max_tokens"], calls[0]["max_tokens"])

    def test_review_raises_clear_error_when_length_persists(self):
        class AlwaysLengthClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return response("", finish="length")

        with self.assertRaises(grading.ModelOutputError) as ctx:
            grading.review("Tema", "Redação.", SETTINGS, client=AlwaysLengthClient())
        self.assertIn("limite de tokens", str(ctx.exception))

    def test_review_bert_only_without_ai_model(self):
        settings = {"base_url": "", "correction_model": "", "use_bert": "0"}
        fake_scores = {"C1": 160, "C2": 120, "C3": 160, "C4": 160, "C5": 120}
        with mock.patch.object(
            grading.bert_notas, "motivo_indisponivel", return_value=None
        ):
            with mock.patch.object(
                grading.bert_notas, "estimar", return_value=(fake_scores, False)
            ):
                result = grading.review("Tema", "Redação.", settings, bert_only=True)
        self.assertEqual(result.score_mode, "bert")
        self.assertEqual(result.note, 720)
        self.assertEqual(result.scores, fake_scores)
        self.assertEqual(result.tokens_in, 0)
        self.assertEqual(result.tokens_out, 0)

    def test_review_hybrid_passes_reference_scores_and_applies_calibration(self):
        settings = dict(SETTINGS, use_bert="1")
        bert_scores = {"C1": 160, "C2": 160, "C3": 160, "C4": 160, "C5": 120}
        client = FakeClient(
            '{"scores": {"C1": 120, "C2": 40, "C3": 80, "C4": 120, "C5": 80},'
            ' "explanation": {"C1": "desvios", "C2": "tangenciou", "C3": "lacunas",'
            ' "C4": "ok", "C5": "incompleta"}, "errors": []}'
        )
        with mock.patch.object(
            grading.bert_notas, "motivo_indisponivel", return_value=None
        ):
            with mock.patch.object(
                grading.bert_notas, "estimar", return_value=(bert_scores, False)
            ):
                result = grading.review("Tema", "Redação.", settings, client=client)
        self.assertEqual(result.score_mode, "bert+llm")
        self.assertEqual(result.note, 440)
        self.assertEqual(result.scores["C2"], 40)
        self.assertIn("ref. BERT: 160", result.explanation)
        user_msg = client.completions.calls[0]["messages"][1]["content"]
        self.assertIn("sem leitura do tema", user_msg)


class FetchModelsTests(unittest.TestCase):
    def _fake_response(self, status_code: int, payload=None, json_error: bool = False):
        resp = mock.Mock()
        resp.status_code = status_code
        if json_error:
            resp.json.side_effect = ValueError("not json")
        else:
            resp.json.return_value = payload
        return resp

    def test_fetch_models_uses_http_get_openai_schema(self):
        fake = self._fake_response(
            200,
            {"data": [{"id": "model-b"}, {"id": "model-a"}, {"id": "model-a"}]},
        )
        with mock.patch("src.grading.provider.httpx.get", return_value=fake) as get:
            models = grading.fetch_models("https://opencode.ai/zen/go/v1", "sk-test")
        self.assertEqual(models, ["model-a", "model-b"])
        get.assert_called_once()
        called_url = get.call_args.args[0]
        headers = get.call_args.kwargs["headers"]
        self.assertEqual(called_url, "https://opencode.ai/zen/go/v1/models")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertIn("x-opencode-session", headers)

    def test_fetch_models_supports_models_key(self):
        fake = self._fake_response(
            200,
            {"models": [{"name": "llama3.2"}, {"id": "qwen2.5"}]},
        )
        with mock.patch("src.grading.provider.httpx.get", return_value=fake):
            models = grading.fetch_models("http://localhost:11434/v1")
        self.assertEqual(models, ["llama3.2", "qwen2.5"])

    def test_fetch_models_404_mentions_url(self):
        fake = self._fake_response(404, {"error": "not found"})
        with mock.patch("src.grading.provider.httpx.get", return_value=fake):
            with self.assertRaises(grading.GradingError) as ctx:
                grading.fetch_models("https://opencode.ai/inference/go", "sk-test")
        self.assertIn("404", str(ctx.exception))
        self.assertIn("https://opencode.ai/inference/go/models", str(ctx.exception))

    def test_fetch_models_rejects_non_json_response(self):
        fake = self._fake_response(200, json_error=True)
        with mock.patch("src.grading.provider.httpx.get", return_value=fake):
            with self.assertRaises(grading.GradingError) as ctx:
                grading.fetch_models("https://example.com/api", "sk-test")
        self.assertIn("JSON válido", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
