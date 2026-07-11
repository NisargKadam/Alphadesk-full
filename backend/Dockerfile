# AlphaDesk backend — Session 10. One image, one process, state on a volume.
# Honest caveat: sentence-transformers pulls torch, so this image is multi-GB.
# RERANK_ENABLED=false keeps the model unloaded at runtime; slimming the
# install itself is the Session 10 stretch.
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# curl exists only for the compose healthcheck probe.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Requirements first: code edits must not re-download torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
