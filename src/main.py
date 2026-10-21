import os

from flask import Flask, render_template, request

app = Flask(__name__)

app.config["OPENAI_BASE_URL"] = os.environ.get(
    "OPENAI_BASE_URL", "https://api.deepseek.com"
)
app.config["OPENAI_MODEL"] = os.environ.get("OPENAI_MODEL", "deepseek-flash")
app.config["OPENAI_IMAGE_MODEL"] = os.environ.get("OPENAI_IMAGE_MODEL", "")
app.config["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

MODEL_PLACEHOLDER = ["deepseek-flash", "deepseek-chat"]


def model_options(model):
    options = list(MODEL_PLACEHOLDER)
    if model and model not in options:
        options.insert(0, model)
    return options


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    error = None
    filename = None
    if request.method == "POST":
        file = request.files.get("file")
        filename = file.filename if file else None
        if not filename:
            error = "Selecione um arquivo."
    return render_template("upload.html", error=error, filename=filename)


@app.route("/config", methods=["GET", "POST"])
def config():
    message = None
    if request.method == "POST":
        app.config["OPENAI_BASE_URL"] = request.form.get("base_url", "").strip()
        app.config["OPENAI_MODEL"] = request.form.get("correction_model", "").strip()
        app.config["OPENAI_IMAGE_MODEL"] = request.form.get("image_model", "").strip()
        app.config["OPENAI_API_KEY"] = request.form.get("api_key", "").strip()
        if request.form.get("action") == "connect":
            message = "Conexão com a API ainda não implementada."
        else:
            message = "Configurações salvas nesta sessão."
    correction_model = app.config["OPENAI_MODEL"]
    image_model = app.config["OPENAI_IMAGE_MODEL"]
    return render_template(
        "config.html",
        base_url=app.config["OPENAI_BASE_URL"],
        api_key=app.config["OPENAI_API_KEY"],
        correction_model=correction_model,
        correction_options=model_options(correction_model),
        image_model=image_model,
        image_options=model_options(image_model),
        message=message,
    )
