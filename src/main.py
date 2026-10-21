import base64
import hashlib
import json
import mimetypes
import os
import sqlite3
import subprocess
import tempfile
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pypdf
from dotenv import load_dotenv
from flask import Flask, abort, redirect, render_template, request, url_for
from openai import OpenAI, OpenAIError
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

DATA_DIR = Path(app.root_path).parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
ENV_PATH = Path(app.root_path).parent / ".env"

DEFAULT_SETTINGS = {
    "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
    "correction_model": os.environ.get("OPENAI_MODEL", ""),
    "image_model": os.environ.get("OPENAI_IMAGE_MODEL", ""),
    "models": "[]",
}

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
TEXT_EXTENSIONS = {".md", ".txt"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

VISION_MODEL = "deepseek-v4-flash-vision-exp"

TRANSCRIBE_PROMPT = """Transcreva fielmente o conteúdo da imagem ou documento,
em português, sem comentários, correções ou formatação extra.
Responda apenas com a transcrição."""

REVIEW_PROMPT = """Você é um corretor de redações do ENEM em português.
Avalie a redação considerando o tema informado e regras do ENEM, quais são:
A prova de redação exigirá de você a produção de um texto dissertativo-argumentativo, em modalidade escrita formal da língua portuguesa, sobre um tema de ordem social, científica, cultural ou política. Os aspectos a serem avaliados relacionam-se às competências que devem ter sido desenvolvidas ao longo de sua formação, ou seja, ao final do ensino médio. Nessa redação, você deverá defender um ponto de vista — uma opinião a respeito do tema proposto —, apoiada em argumentos consistentes,  estruturados com coerência e coesão, formando uma unidade textual. Para tanto, deverá selecionar, organizar e relacionar, também de forma coerente e coesa, argumentos e fatos para a defesa do seu ponto de vista. Você também deverá e laborar uma proposta de intervenção social para o problema apresentado no desenvolvimento do texto. Por fim, essa proposta deve respeitar os direitos humanos. [...]
Tal que ignore a quantidade de linha, já que por ser um produto em texto digital, é ignorado a sua extensão (pela caligrafia humana variar). Além disso, a nota é uma junção de cinco competências, sendo elas: • Competência 1: Domínio da norma culta da língua escrita (gramática, ortografia e sintaxe).
• Competência 2: Compreensão do tema e aplicação de repertório sociocultural produtivo.
• Competência 3: Capacidade de interpretar, selecionar, relacionar e organizar informações em defesa de um ponto de vista.
• Competência 4: Uso de recursos coesivos (conectivos) para articular as ideias do texto.
• Competência 5: Elaboração de uma proposta de intervenção para o problema abordado, respeitando os direitos humanos.
Responda somente com JSON válido, sem markdown, no formato:
{"note": "nota de 0 a 1000", "explanation": "explicação detalhada em topicos", "errors": ["erro 1", "erro 2"]}
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


def save_api_key(key):
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    lines = [
        line
        for line in lines
        if not line.lstrip().removeprefix("export ").startswith("OPENAI_API_KEY=")
    ]
    lines.append(f"OPENAI_API_KEY={key}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    ENV_PATH.chmod(0o600)
    os.environ["OPENAI_API_KEY"] = key


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


def api_client():
    base_url = get_settings()["base_url"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        if urlparse(base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                "OPENAI_API_KEY não configurada. Defina-a no .env e reinicie."
            )
        app.logger.warning("OPENAI_API_KEY não configurada; usando endpoint local.")
    return OpenAI(api_key=api_key or "local", base_url=base_url)


def convert_to_png(path):
    with tempfile.NamedTemporaryFile(
        suffix=".png", dir=UPLOAD_DIR, delete=False
    ) as handle:
        target = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), str(target)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg não encontrado para converter a imagem.") from exc
    except subprocess.CalledProcessError as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError("Falha ao converter a imagem para PNG.") from exc
    return target


def transcribe(path):
    model = get_settings()["image_model"] or VISION_MODEL
    converted = None
    if path.suffix.lower() == ".pdf":
        source = path
        mime = "application/pdf"
    elif path.suffix.lower() in IMAGE_EXTENSIONS:
        source = path
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
    else:
        source = converted = convert_to_png(path)
        mime = "image/png"
    encoded = base64.b64encode(source.read_bytes()).decode()
    if converted:
        converted.unlink(missing_ok=True)
    response = api_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSCRIBE_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    }
                ],
            },
        ],
    )
    return response.choices[0].message.content.strip()


def obtain_text(path):
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        text = path.read_text(errors="replace").strip()
    else:
        text = ""
        if suffix == ".pdf":
            try:
                text = extract_text(path)
            except Exception:
                app.logger.exception("Falha ao extrair texto de %s", path.name)
        if not text:
            app.logger.warning("Transcrevendo %s com %s", path.name, VISION_MODEL)
            text = transcribe(path).strip()
    if not text:
        raise ValueError("Não foi possível extrair texto do arquivo.")
    return text


def request_review(theme, path):
    settings = get_settings()
    model = settings["correction_model"]
    if not model:
        raise ValueError("Nenhum modelo de correção configurado.")
    messages = [
        {"role": "system", "content": REVIEW_PROMPT},
        {"role": "user", "content": f"Tema: {theme}\n\nRedação:\n{obtain_text(path)}"},
    ]
    response = api_client().chat.completions.create(model=model, messages=messages)
    return parse_review(response.choices[0].message.content)


def start_correction(filename, stored_name, theme):
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
    corrections = recent_corrections()
    pending = any(c["status"] == "corrigindo" for c in corrections)
    return render_template(
        "index.html", corrections=corrections, pending=pending
    )


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


@app.route("/correction/<int:correction_id>/delete", methods=["POST"])
def delete_correction(correction_id):
    rows = query("SELECT stored_name FROM corrections WHERE id = ?", (correction_id,))
    if not rows:
        abort(404)
    execute("DELETE FROM corrections WHERE id = ?", (correction_id,))
    (UPLOAD_DIR / rows[0]["stored_name"]).unlink(missing_ok=True)
    return redirect(url_for("index"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    error = None
    if request.method == "POST":
        theme = request.form.get("theme", "").strip()
        if request.form.get("source") == "text":
            text = request.form.get("text", "").strip()
            if not text:
                error = "Cole o texto."
            elif not theme:
                error = "Informe o tema."
            else:
                stored_name = f"{uuid4().hex}_texto.txt"
                (UPLOAD_DIR / stored_name).write_text(text)
                start_correction("Texto colado.txt", stored_name, theme)
                return redirect(url_for("index"))
        else:
            file = request.files.get("file")
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
                start_correction(filename, stored_name, theme)
                return redirect(url_for("index"))
    return render_template("upload.html", error=error)


@app.route("/config", methods=["GET", "POST"])
def config():
    message = None
    key_error = False
    settings = get_settings()
    if request.method == "POST":
        settings["base_url"] = request.form.get("base_url", "").strip()
        settings["correction_model"] = request.form.get("correction_model", "").strip()
        settings["image_model"] = request.form.get("image_model", "").strip()
        entered_key = request.form.get("api_key", "").strip()
        if any(char.isspace() for char in entered_key):
            key_error = True
            entered_key = ""
        if entered_key:
            settings["api_key_hash"] = hash_key(entered_key)
            if request.remote_addr in ("127.0.0.1", "::1"):
                save_api_key(entered_key)

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
                    deepseek = "deepseek" in settings["base_url"].lower()
                    if settings["correction_model"] not in models:
                        settings["correction_model"] = models[0] if models else ""
                    if settings["image_model"] not in models:
                        if deepseek:
                            settings["image_model"] = VISION_MODEL
                        else:
                            settings["image_model"] = models[0] if models else ""
                    message = f"Conectado. {len(models)} modelo(s) encontrado(s)."
        else:
            message = "Configurações salvas."
        save_settings(settings)
        if key_error:
            message = "Chave da API inválida."

    models = json.loads(settings["models"])
    deepseek = "deepseek" in settings["base_url"].lower()
    correction_options = list(models)
    image_options = list(models)
    if deepseek and VISION_MODEL not in image_options:
        image_options.insert(0, VISION_MODEL)
    if (
        settings["correction_model"]
        and settings["correction_model"] not in correction_options
    ):
        correction_options.insert(0, settings["correction_model"])
    if settings["image_model"] and settings["image_model"] not in image_options:
        image_options.insert(0, settings["image_model"])
    return render_template(
        "config.html",
        base_url=settings["base_url"],
        key_configured=bool(os.environ.get("OPENAI_API_KEY")),
        key_entered=bool(settings.get("api_key_hash")),
        correction_options=correction_options,
        image_options=image_options,
        correction_model=settings["correction_model"],
        image_model=settings["image_model"],
        message=message,
    )
