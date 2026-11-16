import os
import smtplib
import unittest
from unittest.mock import patch

from src.main import app


class ProblemReportTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_sends_report_to_fixed_recipient_with_reply_to(self):
        smtp_settings = {
            "SMTP_HOST": "smtp.example.test",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "sender@example.test",
            "SMTP_PASSWORD": "test-password",
        }
        with (
            patch.dict(os.environ, smtp_settings, clear=True),
            patch("src.main.smtplib.SMTP") as smtp,
            patch("src.main.ssl.create_default_context") as ssl_context,
        ):
            server = smtp.return_value.__enter__.return_value
            response = self.client.post(
                "/report-problem",
                data={
                    "name": "Ana Silva",
                    "email": "ana@example.test",
                    "category": "Erro no sistema",
                    "description": "A tela não carrega.",
                },
            )

        self.assertEqual(response.status_code, 302)
        server.starttls.assert_called_once_with(context=ssl_context.return_value)
        message = server.send_message.call_args.args[0]
        self.assertEqual(message["To"], "marcusvssasis@gmail.com")
        self.assertEqual(message["Reply-To"], "ana@example.test")
        self.assertEqual(message["Subject"], "Relato recebido pelo LocalRevisor")
        for value in (
            "Ana Silva",
            "ana@example.test",
            "Erro no sistema",
            "A tela não carrega.",
        ):
            self.assertIn(value, message.get_content())

    def test_rejects_missing_required_description_without_sending(self):
        with patch("src.main.smtplib.SMTP") as smtp:
            response = self.client.post(
                "/report-problem",
                data={"category": "Outros", "description": "  "},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Descreva o problema", response.get_data(as_text=True))
        smtp.assert_not_called()

    def test_reports_missing_smtp_configuration(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "src.main.smtplib.SMTP"
        ) as smtp:
            response = self.client.post(
                "/report-problem",
                data={"category": "Outros", "description": "Teste."},
            )

        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Não foi possível enviar", page)
        self.assertIn("SMTP_HOST", page)
        smtp.assert_not_called()

    def test_reports_smtp_failure_and_keeps_form_data_for_retry(self):
        smtp_settings = {
            "SMTP_HOST": "smtp.example.test",
            "SMTP_USERNAME": "sender@example.test",
            "SMTP_PASSWORD": "test-password",
        }
        with (
            patch.dict(os.environ, smtp_settings, clear=True),
            patch("src.main.smtplib.SMTP") as smtp,
            patch("src.main.ssl.create_default_context"),
        ):
            server = smtp.return_value.__enter__.return_value
            server.send_message.side_effect = smtplib.SMTPException("relay unavailable")
            response = self.client.post(
                "/report-problem",
                data={
                    "email": "ana@example.test",
                    "category": "Outros",
                    "description": "Não consigo salvar.",
                },
            )

        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Não foi possível enviar o relato", page)
        self.assertIn("Não consigo salvar.", page)
        self.assertIn('id="report-submit" type="submit">Enviar relato</button>', page)


if __name__ == "__main__":
    unittest.main()
