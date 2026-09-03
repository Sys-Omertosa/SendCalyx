FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install --upgrade pip
# CPU wheels keep the image small; CUDA is still used automatically when a
# GPU-enabled torch build is present at runtime.
RUN .venv/bin/pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# --- Runtime image
FROM python:3.12-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

COPY --from=builder /app/.venv /app/.venv
COPY app.py app_utils.py analysis.py architectures.py ./
COPY models/ ./models/

# The app binds to $PORT, which container platforms such as Cloud Run inject
# (commonly 8080). The default above applies when nothing is provided.
EXPOSE 8080

CMD ["python", "app.py"]
