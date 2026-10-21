FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requeriments.txt .
RUN pip install --no-cache-dir -r requeriments.txt

COPY src ./src

EXPOSE 5000

CMD ["python", "-m", "flask", "--app", "src.main", "run", "--host", "0.0.0.0", "--port", "5000"]
