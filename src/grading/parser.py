"""Parsing, normalização de notas ENEM e validação de citações."""
from __future__ import annotations

import json
import math
import re

from .models import COMPETENCIAS, ModelOutputError


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(content: str | None) -> dict:
    if not isinstance(content, str) or not content.strip():
        raise ModelOutputError("O modelo não devolveu conteúdo.")
    text = _strip_fence(content)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ModelOutputError("O modelo não devolveu um objeto JSON.")
    candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"JSON inválido na resposta do modelo: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelOutputError("Formato inesperado na resposta do modelo.")
    return data


def normalize_errors(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {str(item).strip()}" for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _number(value) -> float:
    if isinstance(value, bool):
        raise ModelOutputError("Nota inválida.")
    try:
        number = float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"Nota inválida: {value!r}") from exc
    if not math.isfinite(number):
        raise ModelOutputError("Nota inválida.")
    return number


def normalize_competency(value) -> int:
    number = _number(value)
    if not 0 <= number <= 200:
        raise ModelOutputError(f"Nota de competência fora de 0–200: {value!r}")
    return max(0, min(200, int(math.floor((number + 20) / 40) * 40)))


def normalize_note(value) -> int:
    number = _number(value)
    if not 0 <= number <= 1000:
        raise ModelOutputError(f"Nota final fora de 0–1000: {value!r}")
    return max(0, min(1000, int(math.floor((number + 20) / 40) * 40)))


def parse_scores(data: dict, required: bool) -> dict[str, int]:
    raw = data.get("scores")
    if raw is None:
        raw = {name: data[name] for name in COMPETENCIAS if name in data}
    if not isinstance(raw, dict):
        if required:
            raise ModelOutputError("A resposta não trouxe o objeto de competências.")
        return {}
    scores: dict[str, int] = {}
    for name in COMPETENCIAS:
        if name not in raw:
            if required:
                raise ModelOutputError(f"A resposta não trouxe a competência {name}.")
            continue
        scores[name] = normalize_competency(raw[name])
    if required and len(scores) != 5:
        raise ModelOutputError("A resposta não trouxe as cinco competências.")
    return scores


def parse_review(
    content: str, *, require_scores: bool = True
) -> tuple[int, dict[str, int], str, str]:
    data = extract_json_object(content)
    scores = parse_scores(data, require_scores)
    if scores:
        note = sum(scores.values())
    else:
        note = normalize_note(data.get("note"))
    explanation = str(data.get("explanation", "")).strip()
    if not explanation:
        raise ModelOutputError("A resposta não trouxe explicação.")
    return note, scores, explanation, normalize_errors(data.get("errors"))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quotes(explanation: str, source_text: str) -> str:
    """Sinaliza citações entre aspas que não aparecem literalmente na redação."""
    if not explanation or not source_text:
        return ""
    fonte = _normalize_text(source_text)
    faltando: list[str] = []
    for quote in re.findall(r"[“\"]([^”\"]{12,})[”\"]", explanation):
        trecho = _normalize_text(quote).strip(" .,;:!?")
        if len(trecho) < 12:
            continue
        if trecho not in fonte and trecho not in faltando:
            faltando.append(trecho)
        if len(faltando) >= 2:
            break
    if not faltando:
        return ""
    resumo = "; ".join(f"“{item[:80]}”" for item in faltando)
    return f"- Aviso: citação não encontrada literalmente na redação: {resumo}"


def join_errors(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def parse_hybrid(
    content: str, bert_scores: dict[str, int], truncated: bool, source_text: str = ""
) -> tuple[int, dict[str, int], str, str]:
    data = extract_json_object(content)
    parsed_scores = parse_scores(data, required=False)
    final_scores = {
        name: parsed_scores.get(name, bert_scores[name]) for name in COMPETENCIAS
    }
    explanations = data.get("explanation", {})
    if not isinstance(explanations, dict):
        raise ModelOutputError("A resposta não trouxe a explicação por competência.")
    linhas = []
    ausentes = []
    for name in COMPETENCIAS:
        texto = str(explanations.get(name, "")).strip()
        if not texto:
            ausentes.append(name)
        nota_final = final_scores[name]
        nota_bert = bert_scores[name]
        cabecalho = (
            f"{name} ({nota_final}/200, ref. BERT: {nota_bert})"
            if nota_final != nota_bert
            else f"{name} ({nota_final}/200)"
        )
        linhas.append(f"{cabecalho}: {texto or 'Sem explicação.'}")
    errors = normalize_errors(data.get("errors"))
    if ausentes:
        errors = join_errors(
            errors,
            "- Aviso: o modelo não explicou " + ", ".join(ausentes) + ".",
        )
    if truncated:
        linhas.append(
            "Aviso: a redação é longa e o modelo de notas leu apenas o início do texto."
        )
    quote_warning = verify_quotes("\n".join(linhas), source_text)
    errors = join_errors(errors, quote_warning)
    return sum(final_scores.values()), final_scores, "\n\n".join(linhas), errors
