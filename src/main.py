import base64
import hashlib
import json
import mimetypes
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pypdf
from flask import Flask, abort, redirect, render_template, request, url_for
from openai import OpenAI, OpenAIError
from werkzeug.utils import secure_filename

app = Flask(__name__)

DATA_DIR = Path(app.root_path).parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"

DEFAULT_SETTINGS = {
    "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
    "correction_model": os.environ.get("OPENAI_MODEL", ""),
    "image_model": os.environ.get("OPENAI_IMAGE_MODEL", ""),
    "models": "[]",
}

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
TEXT_EXTENSIONS = {".md", ".txt"}

REVIEW_PROMPT = """Você é um corretor de redações em português.
Avalie a redação considerando o tema informado.
Responda somente com JSON válido, sem markdown, no formato:
{"note": "nota de 0 a 1000", "explanation": "explicação detalhada", "errors": ["erro 1", "erro 2"]}
"""


def query(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


def execute(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        return conn.execute(sql, params).lastrowid


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    execute(
        "CREATE TABLE IF NOT EXISTS settings ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    execute(
        "CREATE TABLE IF NOT EXISTS corrections ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "filename TEXT NOT NULL,"
        "stored_name TEXT NOT NULL,"
        "theme TEXT NOT NULL,"
        "created_at TEXT NOT NULL,"
        "status TEXT NOT NULL DEFAULT 'pendente',"
        "note TEXT NOT NULL DEFAULT '',"
        "explanation TEXT NOT NULL DEFAULT '',"
        "errors TEXT NOT NULL DEFAULT '')"
    )
    for key, value in DEFAULT_SETTINGS.items():
        execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value)
        )
    legacy = query("SELECT value FROM settings WHERE key = 'api_key'")
    if legacy and legacy[0]["value"]:
        execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('api_key_hash', ?)",
            (hash_key(legacy[0]["value"]),),
        )
    execute("DELETE FROM settings WHERE key = 'api_key'")
    columns = {row["name"] for row in query("PRAGMA table_info(corrections)")}
    for name, definition in (
        ("status", "TEXT NOT NULL DEFAULT 'pendente'"),
        ("note", "TEXT NOT NULL DEFAULT ''"),
        ("explanation", "TEXT NOT NULL DEFAULT ''"),
        ("errors", "TEXT NOT NULL DEFAULT ''"),
    ):
        if name not in columns:
            execute(f"ALTER TABLE corrections ADD COLUMN {name} {definition}")


def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest()


def get_settings():
    rows = query("SELECT key, value FROM settings")
    return {row["key"]: row["value"] for row in rows}


def save_settings(values):
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.executemany(
            "UPDATE settings SET value = ? WHERE key = ?",
            [(value, key) for key, value in values.items()],
        )


def recent_corrections(limit=20):
    return query(
        "SELECT id, theme, filename, created_at, status, note"
        " FROM corrections ORDER BY id DESC LIMIT ?",
        (limit,),
    )


def fetch_models(base_url, api_key):
    client = OpenAI(api_key=api_key or "local", base_url=base_url)
    return sorted(model.id for model in client.models.list())


def extract_text(path):
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return path.read_text(errors="replace").strip()


def parse_review(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("Formato inesperado na resposta do modelo.")
    errors = data.get("errors", "")
    if isinstance(errors, list):
        errors = "\n".join(f"- {item}" for item in errors)
    return (
        str(data.get("note", "")).strip(),
        str(data.get("explanation", "")).strip(),
        errors.strip(),
    )


def request_review(theme, path):
    settings = get_settings()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        app.logger.warning("OPENAI_API_KEY não configurada.")
    client = OpenAI(api_key=api_key or "local", base_url=settings["base_url"])
    if path.suffix.lower() in TEXT_EXTENSIONS or path.suffix.lower() == ".pdf":
        model = settings["correction_model"]
        if not model:
            raise ValueError("Nenhum modelo de correção configurado.")
        text = extract_text(path)
        if not text:
            raise ValueError("Não foi possível extrair texto do arquivo.")
        messages = [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": f"Tema: {theme}\n\nRedação:\n{text}"},
        ]
    else:
        model = settings["image_model"]
        if not model:
            raise ValueError("Nenhum modelo de imagem configurado.")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        messages = [
            {"role": "system", "content": REVIEW_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Tema: {theme}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            },
        ]
    response = client.chat.completions.create(model=model, messages=messages)
    return parse_review(response.choices[0].message.content)


def process_correction(correction_id, theme, path):
    try:
        note, explanation, errors = request_review(theme, path)
    except Exception as exc:  # never leave the row stuck in "corrigindo"
        app.logger.exception("Falha na correção %s", correction_id)
        execute(
            "UPDATE corrections SET status = 'erro', errors = ? WHERE id = ?",
            (f"Falha na correção: {exc}", correction_id),
        )
    else:
        execute(
            "UPDATE corrections SET status = 'corrigido', note = ?,"
            " explanation = ?, errors = ? WHERE id = ?",
            (note, explanation, errors, correction_id),
        )


init_db()


@app.route("/")
def index():
    return render_template("index.html", corrections=recent_corrections())


@app.route("/review/<int:correction_id>")
def review(correction_id):
    rows = query("SELECT * FROM corrections WHERE id = ?", (correction_id,))
    if not rows:
        abort(404)
    if rows[0]["status"] == "corrigindo":
        return redirect(url_for("index"))
    return render_template("review.html", correction=rows[0])


@app.route("/correction/<int:correction_id>/retry", methods=["POST"])
def retry(correction_id):
    rows = query("SELECT * FROM corrections WHERE id = ?", (correction_id,))
    if not rows:
        abort(404)
    correction = rows[0]
    if correction["status"] == "corrigindo":
        return redirect(url_for("index"))
    execute(
        "UPDATE corrections SET status = 'corrigindo', note = '',"
        " explanation = '', errors = '' WHERE id = ?",
        (correction_id,),
    )
    threading.Thread(
        target=process_correction,
        args=(
            correction_id,
            correction["theme"],
            UPLOAD_DIR / correction["stored_name"],
        ),
        daemon=True,
    ).start()
    return redirect(url_for("index"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        theme = request.form.get("theme", "").strip()
        filename = file.filename if file else None
        if not filename:
            error = "Selecione um arquivo."
        elif (
            os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS
            and not file.mimetype.startswith("image/")
        ):
            error = "Formato não suportado. Envie PDF, imagem, .md ou .txt."
        elif not theme:
            error = "Informe o tema."
        else:
            safe_name = secure_filename(filename) or "arquivo"
            stored_name = f"{uuid4().hex}_{safe_name}"
            file.save(UPLOAD_DIR / stored_name)
            correction_id = execute(
                "INSERT INTO corrections"
                " (filename, stored_name, theme, created_at, status)"
                " VALUES (?, ?, ?, ?, 'corrigindo')",
                (
                    filename,
                    stored_name,
                    theme,
                    datetime.now().strftime("%d/%m/%Y %H:%M"),
                ),
            )
            threading.Thread(
                target=process_correction,
                args=(correction_id, theme, UPLOAD_DIR / stored_name),
                daemon=True,
            ).start()
            return redirect(url_for("index"))
    return render_template("upload.html", error=error)


@app.route("/config", methods=["GET", "POST"])
def config():
    message = None
    settings = get_settings()
    if request.method == "POST":
        settings["base_url"] = request.form.get("base_url", "").strip()
        settings["correction_model"] = request.form.get(
            "correction_model", ""
        ).strip()
        settings["image_model"] = request.form.get("image_model", "").strip()
        entered_key = request.form.get("api_key", "").strip()
        if entered_key:
            settings["api_key_hash"] = hash_key(entered_key)

        if request.form.get("action") == "connect":
            if not settings["base_url"]:
                message = "Informe a URL base."
            else:
                try:
                    key = entered_key or os.environ.get("OPENAI_API_KEY", "")
                    models = fetch_models(settings["base_url"], key)
                except OpenAIError as exc:
                    message = f"Não foi possível conectar: {exc}"
                else:
                    settings["models"] = json.dumps(models)
                    for field in ("correction_model", "image_model"):
                        if settings[field] not in models:
                            settings[field] = models[0] if models else ""
                    message = f"Conectado. {len(models)} modelo(s) encontrado(s)."
        else:
            message = "Configurações salvas."
        save_settings(settings)

    model_options = json.loads(settings["models"])
    for field in ("correction_model", "image_model"):
        if settings[field] and settings[field] not in model_options:
            model_options.insert(0, settings[field])
    return render_template(
        "config.html",
        base_url=settings["base_url"],
        key_configured=bool(os.environ.get("OPENAI_API_KEY")),
        key_entered=bool(settings.get("api_key_hash")),
        model_options=model_options,
        correction_model=settings["correction_model"],
        image_model=settings["image_model"],
        message=message,
    )
