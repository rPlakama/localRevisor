"""Proteção CSRF, cabeçalhos de segurança e páginas de erro HTTP."""
from __future__ import annotations

import hmac
import logging
import secrets

from flask import Flask, abort, render_template, request, session

from .. import paths

LOGGER = logging.getLogger(__name__)


def csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def register(app: Flask) -> None:
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def _csrf_protect():
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            expected = session.get("_csrf_token", "")
            sent = request.form.get("_csrf_token") or request.headers.get(
                "X-CSRF-Token", ""
            )
            if not expected or not hmac.compare_digest(str(expected), str(sent)):
                abort(400, description="Sessão inválida ou formulário expirado.")

    @app.after_request
    def _security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self' 'wasm-unsafe-eval'; "
            "connect-src 'self' http://localhost:* http://127.0.0.1:*; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )
        return response

    @app.errorhandler(400)
    def bad_request(error):
        return (
            render_template(
                "error.html",
                code=400,
                message=getattr(error, "description", "Requisição inválida."),
            ),
            400,
        )

    @app.errorhandler(404)
    def not_found(_error):
        return (
            render_template("error.html", code=404, message="Página não encontrada."),
            404,
        )

    @app.errorhandler(413)
    def too_large(_error):
        limit = paths.MAX_UPLOAD_BYTES / (1024 * 1024)
        return (
            render_template(
                "error.html",
                code=413,
                message=f"O arquivo é maior que o limite de {limit:.0f} MB.",
            ),
            413,
        )

    @app.errorhandler(500)
    def server_error(error):
        LOGGER.exception("Erro interno: %s", error)
        return (
            render_template(
                "error.html",
                code=500,
                message="Algo deu errado no servidor. Tente novamente.",
            ),
            500,
        )
