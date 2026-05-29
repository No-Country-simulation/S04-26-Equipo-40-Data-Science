# ConversaAI — Sistema de Análisis de Conversaciones de Soporte

Pipeline de NLP para clasificar sentimiento (3 clases), intención (9 clases) y riesgo de churn en conversaciones de soporte al cliente en español y portugués.

## Enlaces

| Recurso | URL |
|---------|-----|
| Repositorio GitHub | [No-Country-simulation/S04-26-Equipo-40-Data-Science](https://github.com/No-Country-simulation/S04-26-Equipo-40-Data-Science) |
| Modelo Sentimiento (HF Hub) | [Rosela/xlm-r-sentiment-espt](https://huggingface.co/Rosela/xlm-r-sentiment-espt) |
| Modelo Intención (HF Hub) | [Rosela/xlm-r-intent-espt](https://huggingface.co/Rosela/xlm-r-intent-espt) |
| Perfil HF Hub | [huggingface.co/Rosela](https://huggingface.co/Rosela) |

## Arquitectura

```
Mensaje
  ├──► SentimentClassifier (XLM-R fine-tuneado)
  │     3 clases: negative / neutral / positive
  ├──► IntentClassifier (XLM-R fine-tuneado)
  │     9 clases: cancelacion, consulta_general, facturacion_pago,
  │               feedback, gestion_cuenta, modificacion_pedido,
  │               queja, reembolso, seguimiento
  └──► ChurnScorer (heurístico)
        w1 * sent_prob_negative + w2 * frust_flag + w3 * max(intent_risk)
        Pesos default: w1=0.3, w2=0.35, w3=0.35
```

### Fallback automático

Si los modelos fine-tuneados no están disponibles (sin GPU/local), el pipeline cae a:
- **Sentimiento**: `nlptown/bert-base-multilingual-uncased-sentiment` (5 estrellas → 3 clases)
- **Intención**: `MoritzLaurer/DeBERTa-v3-base-mnli` (zero-shot sobre las 9 clases)
- **Demo 100% offline**: `scripts/pipeline_fallback.py` — rule-based por keywords + regex (sin dependencias ML)

## Stack

| Componente | Tecnología |
|------------|-----------|
| Modelos | XLM-R (`xlm-roberta-base`) fine-tuneado ES/PT |
| Entrenamiento | Kaggle (GPU T4x2) + Hugging Face Hub |
| Dashboard | Streamlit + Plotly |
| Infraestructura | Docker (Python 3.11-slim, CPU) |
| Lenguaje | Python 3.11 |

## Herramientas y Tecnologías

### Lenguaje y Entorno

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| Python 3.11 | Lenguaje principal |
| pip | Gestor de dependencias |
| venv | Entorno virtual aislado |
| Docker | Containerización para deploy |

### Machine Learning / NLP

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| PyTorch 2.x | Backend de deep learning (CPU/GPU) |
| Hugging Face Transformers | Modelos XLM-R, tokenización, pipeline HF |
| Hugging Face Datasets | Carga y procesamiento de datasets |
| Hugging Face Hub | Registro y distribución de modelos fine-tuneados |
| scikit-learn | Split estratificado, métricas de evaluación |
| evaluate | Métricas estandarizadas (accuracy, F1) |
| accelerate | Entrenamiento distribuido en GPU |
| XLM-R (xlm-roberta-base) | Modelo base multilingual para fine-tuning |
| Tokenizers (HF) | Tokenización rápida en Rust |

### Entrenamiento

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| Kaggle (GPU T4x2) | Entrenamiento de modelos en la nube |
| Google Colab | Alternativa para fine-tuning |
| Trainer (HF) | Loop de entrenamiento con early stopping, weight decay, label smoothing |

### Dashboard y Visualización

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| Streamlit | Dashboard interactivo web |
| Plotly | Gráficos interactivos (pie, bar, histograma, stacked bar) |
| Matplotlib | Visualización en notebooks |
| Seaborn | Visualización estadística en notebooks |

### Procesamiento de Datos

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| Pandas | Manipulación y análisis de datos |
| NumPy | Operaciones numéricas y arrays |
| PyArrow | Lectura/escritura de formato Parquet |
| regex (re) | Patrones de búsqueda para detección de intenciones en fallback |

### Desarrollo y Colaboración

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| Git | Control de versiones |
| GitHub | Repositorio remoto y colaboración |
| GitHub CLI (gh) | Interacción con GitHub desde CLI |
| opencode (Spec-Driven Dev) | Flujo de trabajo: propuesta → spec → diseño → tareas → implementación → verificación |
| Engram | Memoria persistente entre sesiones de desarrollo |

### DevOps

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| Docker | Imagen con Python 3.11-slim + dependencias + precarga de modelos |
| Docker Hub / Registry | Distribución de imágenes (futuro) |
| Healthcheck Docker | Monitoreo de salud del contenedor |

### Productividad

| Herramienta | Uso en el proyecto |
|-------------|-------------------|
| tqdm | Barras de progreso en procesamiento por lotes |
| logging | Logging estructurado en todo el pipeline |
| threading | Lock thread-safe para singletons de fallback |

## Dashboards

Actualmente el pipeline se ejecuta desde código (Python/notebooks) y los resultados se visualizan en los dashboards.

### Dashboard principal (rama `dashboard`)

El equipo creó un dashboard alternativo en la rama [`dashboard`](https://github.com/No-Country-simulation/S04-26-Equipo-40-Data-Science/tree/dashboard):

- `app.py` — Dashboard Streamlit con clasificación de intenciones y sentimiento
- `docs/` — Dashboard web con HTML/CSS/JS (charts interactivos)
- `tests/` — Tests unitarios con pytest

Para verlo:

```bash
git checkout dashboard
streamlit run app.py
```

## Docker

```bash
docker build -t conversaai .
docker run -p 8501:8501 conversaai
```

Los modelos fine-tuneados se descargan de HF Hub al arrancar (~556MB c/u).

> ⚠️ El Dockerfile actual apunta a `dashboard_app.py` que fue eliminado de `main`.
> Si querés usar Docker, actualizá el `CMD` en el Dockerfile o usá la rama `dashboard`.

## Dataset demo

`data/corpus_demo_bilingue.csv` — 109 mensajes bilingües ES/PT etiquetados con sentimiento + intención para prueba.

## Notebooks Kaggle

- `notebooks/sentiment_xlmr.ipynb.py` — Fine-tuning XLM-R para sentimiento (3 clases, ~92.2% accuracy)
- `notebooks/intent_xlmr.ipynb.py` — Fine-tuning XLM-R para intención (9 clases, ~99.6% F1)
- `notebooks/sentiment_xlmr_colab.ipynb` — Versión Colab del notebook de sentimiento
- `notebooks/intent_xlmr_colab.ipynb` — Versión Colab del notebook de intención

## Scripts

| Script | Propósito |
|--------|-----------|
| `src/pipeline.py` | Orquestador ConversaAIPipeline |
| `src/data/processor.py` | DataProcessor: normalización ES/PT + tokenización XLM-R |
| `src/models/sentiment.py` | SentimentClassifier (XLM-R + fallback nlptown) |
| `src/models/intent.py` | IntentClassifier (XLM-R + fallback zero-shot) |
| `src/features/churn.py` | ChurnScorer heurístico |
| `scripts/pipeline_fallback.py` | Pipeline rule-based 100% offline (demo) |
| `scripts/archive/` | Scripts one-off legacy (preparar datos, push model, etc.) |

## Dataset fine-tuning (Kaggle)

Dataset en Kaggle: [rosaiselagonzalez/conversaai-sentiment-data](https://www.kaggle.com/datasets/rosaiselagonzalez/conversaai-sentiment-data)

Compuesto por:
- `amazon_reviews_multi` ES/PT (sentimiento 5 estrellas → 3 clases)
- Bitext intent dataset (27 intents originales → 9 clases limpias)
- `corpus_ecommerce.csv` (110k registros PT, sentimiento positive/negative)
