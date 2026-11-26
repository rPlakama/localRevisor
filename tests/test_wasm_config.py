"""Testes da opção WebAssembly no cliente ('Usar hardware (seu) para trabalhar com modelos')."""
from __future__ import annotations

import json
import time

from src import db, paths, worker
from src.main import app
from tests import support


def _setup_client(tmp_path):
    paths.configure(tmp_path)
    paths.ensure_dirs()
    support.reset()
    app.config["TESTING"] = True
    return app.test_client()


def test_config_saves_client_wasm_option(tmp_path) -> None:
    client = _setup_client(tmp_path)
    page = client.get("/config").get_data(as_text=True)
    assert "Usar hardware (seu) para trabalhar com modelos" in page

    token = support.csrf(client, "/config")
    resp = client.post(
        "/config",
        data={
            "_csrf_token": token,
            "action": "save",
            "base_url": "http://localhost:11434/v1",
            "correction_model": "",
            "image_model": "",
            "client_wasm": "1",
            "use_bert": "1",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    settings = db.get_settings()
    assert settings["client_wasm"] == "1"
    assert settings["use_bert"] == "1"


def test_wasm_assets_and_weights_endpoint(tmp_path) -> None:
    client = _setup_client(tmp_path)
    wasm_resp = client.get("/static/model_score.wasm")
    assert wasm_resp.status_code == 200
    assert wasm_resp.data[:4] == b"\x00asm"

    weights_resp = client.get("/wasm-model/weights.json")
    assert weights_resp.status_code == 200
    payload = weights_resp.get_json()
    assert payload["comps"] == ["C1", "C2", "C3", "C4", "C5"]
    assert len(payload["scaler_mean"]) == 8


def test_upload_with_client_wasm_scores_without_server_bert(tmp_path) -> None:
    client = _setup_client(tmp_path)
    db.save_settings(
        {
            "base_url": "http://localhost:11434/v1",
            "correction_model": "",
            "image_model": "",
            "models": "[]",
            "use_bert": "1",
            "client_wasm": "1",
        }
    )

    upload_page = client.get("/upload")
    assert upload_page.status_code == 200
    assert 'data-client-wasm="1"' in upload_page.get_data(as_text=True)

    token = support.csrf(client, "/upload")
    scores = {"C1": 160, "C2": 120, "C3": 160, "C4": 120, "C5": 160}
    resp = client.post(
        "/upload",
        data={
            "_csrf_token": token,
            "source": "text",
            "text": "Redação de teste para inferência WebAssembly no cliente.",
            "theme": "Educação digital",
            "bert_only": "1",
            "client_scores": json.dumps(scores),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302

    # Aguarda o worker concluir
    for _ in range(50):
        rows, _ = db.recent_corrections(1, 10)
        if rows and rows[0]["status"] != "corrigindo" and not worker.is_active(rows[0]["id"]):
            break
        time.sleep(0.05)

    rows, _ = db.recent_corrections(1, 10)
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["status"] == "corrigido"
    assert row["score_mode"] == "bert_wasm"
    assert row["note"] == "720"
    assert json.loads(row["scores"]) == scores
