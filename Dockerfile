FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requiriments.txt .
RUN pip install --no-cache-dir -r requiriments.txt

ARG INSTALAR_BERT=0
RUN if [ "$INSTALAR_BERT" = "1" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir transformers numpy; \
    fi

COPY src ./src

EXPOSE 5000

CMD ["python", "-m", "flask", "--app", "src.main", "run", "--host", "0.0.0.0", "--port", "5000"]
