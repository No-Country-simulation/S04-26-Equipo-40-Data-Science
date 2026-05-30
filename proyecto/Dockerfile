FROM python:3.11-slim

WORKDIR /app

# ── Dependencias ──
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# ── Aplicación + modelos entrenados ──
# Sentimiento: Rosela/xlm-r-sentiment-espt (~556MB)
# Intención:   Rosela/xlm-r-intent-espt (~556MB)
COPY . .

# ── Limpiar artefactos de build ──
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python3 -c 'import urllib.request; urllib.request.urlopen("http://localhost:8501/_stcore/health")'

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
