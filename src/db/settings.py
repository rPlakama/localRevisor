"""Leitura e gravação de configurações no SQLite."""
from __future__ import annotations

from contextlib import closing

from .. import paths
from .connection import connect, query

DEFAULT_SETTINGS = {
    "base_url": "https://api.deepseek.com",
    "correction_model": "",
    "image_model": "",
    "models": "[]",
    "use_bert": "0",
    "client_wasm": "0",
    "bert_max_tokens": str(paths.BERT_MAX_TOKENS),
}


def get_settings() -> dict[str, str]:
    settings = dict(DEFAULT_SETTINGS)
    for row in query("SELECT key, value FROM settings"):
        settings[row["key"]] = row["value"]
    return settings


def save_settings(values: dict[str, str]) -> None:
    with closing(connect()) as conn, conn:
        conn.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(key, str(value)) for key, value in values.items()],
        )
