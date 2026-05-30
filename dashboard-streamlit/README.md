# ConversaAI — Dashboard Streamlit (Docker)

Este dashboard es el **nuestro** (versión con `ConversaAIPipeline`).

## Uso con Docker

```bash
docker build -t conversaai-dashboard proyecto/
docker run -p 8501:8501 conversaai-dashboard
```

O localmente:

```bash
cd proyecto/
pip install -r requirements.txt
streamlit run dashboard_app.py
```
