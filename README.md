# ConversaAI — Sistema de Análisis de Sentimiento e Intención para Soporte al Cliente

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-XLM--R-orange?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/No--Country-S04--26-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>Sistema NLP multilenguaje (ES/PT) para detectar frustración, clasificar intenciones y predecir abandono en conversaciones de soporte al cliente.</b>
</p>

---

## 📋 Descripción del Proyecto

**ConversaAI** es un pipeline de análisis de conversaciones de soporte al cliente que identifica:

- 😤 **Frustración** de usuarios en tiempo real
- 🎯 **Intenciones** no resueltas en las conversaciones
- 📉 **Patrones de abandono y escalación** (churn prediction)
- 📊 **Insights accionables** para el equipo de producto

El sistema procesa **2M+ mensajes/mes** en Español y Portugués, utilizando modelos Transformer fine-tuneados sobre XLM-RoBERTa base.

---

## 🏗️ Arquitectura del Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                      ConversaAI Pipeline                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │      N1      │ ─► │      N2      │ ─► │      N3      │        │
│  │ Preparación  │    │ Entrenamiento│    │  Dashboard   │        │
│  │   Datos      │    │   Modelos    │    │   Insights   │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                   │                   │                  │
│         ▼                   ▼                   ▼                  │
│  corpus_bilingue      XLM-R Sentiment     reporte.json            │
│  (328k ES+PT)         XLM-R Intent        dashboard.png           │
│  Bitext (27 intents)  ChurnScorer                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

```
                    ┌─────────────────────────┐
                    │   ConversaAIPipeline     │
                    │  (Orquestador unificado) │
                    └─────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ DataProcessor│     │ DataProcessor│     │ DataProcessor│
  │ (preproc+tok)│     │ (preproc+tok)│     │ (preproc+tok)│
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         ▼                    ▼                     │
  ┌──────────────┐     ┌──────────────┐             │
  │ XLM-R        │     │ XLM-R        │             │
  │ Sentiment    │     │ Intent       │             │
  │ (3 clases)   │     │ (9 clases)   │             │
  └──────┬───────┘     └──────┬───────┘             │
         └──────────┬──────────┘                    │
                    ▼                               │
             ┌──────────────┐                       │
             │ ChurnScorer  │◄──────────────────────┘
             │ (heurístico) │   (frustration flags)
             └──────┬───────┘
                    ▼
             ┌──────────────┐
             │  Dashboard   │
             │ (Streamlit)  │
             └──────────────┘
```

---

## 📊 Datasets Utilizados

| Dataset | Uso | Registros | Idioma |
|---------|-----|-----------|--------|
| `amazon_reviews_multi` (ES) | Entrenamiento sentimiento | ~200K | 🇪🇸 Español |
| `amazon_reviews_multi` (PT) | Entrenamiento sentimiento | ~128K | 🇧🇷 Portugués |
| `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | Entrenamiento intenciones | ~27K (27 intents) | 🌐 EN (traducido) |
| `dair-ai/emotion` | Complemento emocional | ~20K | 🌐 Multilenguaje |
| `corpus_demo_bilingue.csv` | Demo y evaluación | ~1K | 🇪🇸🇧🇷 ES+PT |
| `data/raw/bitext_train.parquet` | Pipeline de intenciones | 27K (binario) | EN→ES/PT |
| `data/raw/emotions_train.parquet` | Pipeline de emociones | ~16K (binario) | Multilenguaje |

> **Nota:** Los datasets de entrenamiento se descargan automáticamente desde HuggingFace Hub durante el proceso de fine-tuning. Solo `corpus_demo_bilingue.csv` está incluido en el repositorio para fines de demostración.

---

## 🤖 Modelos

### Clasificador de Sentimiento — `XLM-R Sentiment`

| Propiedad | Valor |
|-----------|-------|
| **Base checkpoint** | `xlm-roberta-base` |
| **HF Hub** | `Rosela/xlmr-sentiment-3clases` |
| **Clases** | `negative`, `neutral`, `positive` |
| **Idiomas** | Español + Portugués (sin routing) |
| **Precisión objetivo** | > 92% en `amazon_reviews_multi` ES test split |
| **API** | `predict_sentiment(text: str) -> {"label": str, "probability": float, "probabilities": {str: float}}` |

### Detector de Intenciones — `XLM-R Intent`

| Propiedad | Valor |
|-----------|-------|
| **Base checkpoint** | `xlm-roberta-base` |
| **HF Hub** | `Rosela/xlmr-intent-6clases` |
| **Clases (9)** | `cancelacion`, `consulta_general`, `facturacion_pago`, `feedback`, `gestion_cuenta`, `modificacion_pedido`, `queja`, `reembolso`, `seguimiento` |
| **Idiomas** | Español + Portugués |
| **Métrica objetivo** | Macro F1 > 0.80 |
| **API** | `predict_intent(text: str) -> {"intent": str, "confidence": float}` |

### Churn Scorer — `Heurístico Calibrado`

Score de riesgo de abandono (0.0–1.0) calculado mediante la fórmula:

```
churn = w1*(1 - sent_prob_negative) + w2*frust_flag + w3*max(intent_prob ∈ {queja, cancelacion, reembolso})
```

Donde `w1`, `w2`, `w3` son pesos configurables calibrados desde datos de validación.

| Propiedad | Valor |
|-----------|-------|
| **Tipo** | Heurístico (sin datos etiquetados de churn) |
| **Salida** | `sentiment_contribution`, `frustration_contribution`, `intent_contribution`, `aggregate_score` |
| **Umbrales** | Bajo < 0.3 · Medio 0.3–0.6 · Alto 0.6–0.8 · Crítico > 0.8 |

---

## 🗂️ Estructura del Proyecto

```
S04-26-Equipo-40-Data-Science/
│
├── proyecto/                          # 📦 Código principal
│   ├── src/
│   │   ├── models/
│   │   │   ├── sentiment.py          # Clasificador XLM-R sentimiento
│   │   │   └── intent.py             # Clasificador XLM-R intenciones
│   │   ├── features/
│   │   │   └── churn.py              # ChurnScorer heurístico
│   │   ├── data/
│   │   │   └── processor.py          # Preprocesamiento y tokenización
│   │   └── pipeline.py               # Orquestador ConversaAIPipeline
│   │
│   ├── scripts/
│   │   └── pipeline_fallback.py      # Pipeline con fallback TF-IDF/BART
│   │
│   ├── notebooks/
│   │   ├── sentiment_xlmr_colab.ipynb   # Fine-tuning sentimiento (Colab)
│   │   ├── intent_xlmr_colab.ipynb      # Fine-tuning intenciones (Colab)
│   │   ├── sentiment_xlmr.ipynb.py      # Versión local sentimiento
│   │   └── intent_xlmr.ipynb.py         # Versión local intenciones
│   │
│   ├── data/
│   │   ├── corpus_demo_bilingue.csv      # Corpus demo ES+PT
│   │   └── raw/
│   │       ├── bitext_train.parquet      # Dataset intenciones (binario)
│   │       └── emotions_train.parquet    # Dataset emociones (binario)
│   │
│   ├── outputs/
│   │   ├── dashboard.png                 # Captura del dashboard
│   │   └── evaluacion_modelo.png         # Gráfico de evaluación
│   │
│   ├── dashboard_app.py              # Dashboard Streamlit (completo)
│   ├── demo_dashboard.py             # Dashboard Streamlit (demo)
│   ├── Dockerfile                    # Imagen Docker lista para deploy
│   ├── requirements.txt              # Dependencias Python
│   └── README.md                     # Documentación del proyecto
│
├── openspec/                          # 📐 Especificaciones técnicas
│   ├── config.yaml                   # Configuración global
│   ├── specs/
│   │   ├── sentiment-classifier/spec.md
│   │   ├── intent-detector/spec.md
│   │   └── churn-scorer/spec.md
│   └── changes/
│       └── finetuning-sentimiento-intencion-churn/
│           ├── design.md             # Diseño de arquitectura
│           ├── proposal.md           # Propuesta técnica
│           └── tasks.md              # Lista de tareas
│
└── .gitignore
```

---

## 🚀 Instalación y Uso

### Requisitos previos

- Python 3.11+
- 8GB RAM mínimo (16GB recomendado para modelos completos)
- GPU opcional (CPU funciona con mayor latencia)

### Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/No-Country-simulation/S04-26-Equipo-40-Data-Science.git
cd S04-26-Equipo-40-Data-Science/proyecto

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Ejecutar el Dashboard (Demo)

```bash
# Dashboard de demostración (sin modelos locales, usa HF Hub)
streamlit run demo_dashboard.py

# Dashboard completo con modelos fine-tuneados
streamlit run dashboard_app.py
```

El dashboard quedará disponible en: **http://localhost:8501**

### Ejecutar con Docker

```bash
# Construir imagen
docker build -t conversaai .

# Ejecutar contenedor
docker run -p 8501:8501 conversaai
```

---

## 🔬 Fine-tuning de Modelos

### Sentimiento (Google Colab)

```bash
# Abrir en Colab
notebooks/sentiment_xlmr_colab.ipynb
```

El notebook realiza:
1. Descarga y preprocesamiento del dataset `amazon_reviews_multi` (ES+PT)
2. Fine-tuning de `xlm-roberta-base` con 3 épocas, batch size 32
3. Evaluación y upload a HuggingFace Hub (`Rosela/xlmr-sentiment-3clases`)

### Intenciones (Google Colab)

```bash
# Abrir en Colab
notebooks/intent_xlmr_colab.ipynb
```

El notebook realiza:
1. Descarga del dataset Bitext (27K muestras, 27 intenciones)
2. Fine-tuning de `xlm-roberta-base` para 9 clases de intención
3. Evaluación con Macro F1 y upload a HuggingFace Hub

---

## 📈 Métricas Objetivo

| Modelo | Métrica | Objetivo |
|--------|---------|----------|
| XLM-R Sentimiento | Accuracy en test split ES | > 92% |
| XLM-R Sentimiento | Accuracy en conversaciones reales | > 85% |
| XLM-R Intenciones | Macro F1 | > 0.80 |
| ChurnScorer | AUC-ROC (validación heurística) | > 0.75 |
| Pipeline completo | Latencia por mensaje | < 500ms |

---

## 🛠️ Decisiones de Arquitectura

| Decisión | Elegido | Razón |
|----------|---------|-------|
| Base model | `xlm-roberta-base` | Control total del fine-tuning vs modelos pre-fine-tuneados |
| Tokenizer | Compartido (XLM-R) | Menos memoria, una sola configuración |
| Churn | Heurístico calibrado | Sin datos etiquetados de churn disponibles |
| Carga de modelos | HuggingFace Hub | Versionado, fácil rollback, fallback a TF-IDF |
| Batch size | 32 | Equilibrio memoria/velocidad en T4 16GB |
| Épocas | 3 + early stopping | Evitar overfitting |
| HF Hub org | `Rosela/` | Cuenta real del equipo |

---

## 📦 Dependencias Principales

```txt
# Core
pandas>=1.5.0
numpy>=1.23.0

# ML/NLP
transformers>=4.30.0
torch>=2.0.0         # CPU-only instalado por separado
scikit-learn>=1.3.0

# Dashboard
streamlit
plotly

# Data
datasets             # HuggingFace datasets
```

---

## 👥 Equipo — No Country S04-26 Data Science

| Rol | Responsabilidad |
|-----|----------------|
| Data Scientist | Fine-tuning de modelos XLM-R, pipeline NLP |
| ML Engineer | Arquitectura del pipeline, Docker, HF Hub |
| Data Analyst | Dashboard Streamlit, insights y visualizaciones |
| Project Lead | OpenSpec, diseño de arquitectura, coordinación |

> **Simulación:** No Country — Sprint S04 · Equipo 40 · Data Science

---

## 🔗 Referencias

| Recurso | URL |
|---------|-----|
| Repositorio | [GitHub](https://github.com/No-Country-simulation/S04-26-Equipo-40-Data-Science) |
| Modelo Sentimiento | [Rosela/xlmr-sentiment-3clases](https://huggingface.co/Rosela/xlmr-sentiment-3clases) |
| Modelo Intenciones | [Rosela/xlmr-intent-6clases](https://huggingface.co/Rosela/xlmr-intent-6clases) |
| Dataset Base | [amazon_reviews_multi](https://huggingface.co/datasets/amazon_reviews_multi) |
| Dataset Intenciones | [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| XLM-RoBERTa | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) |

---

<p align="center">
  Hecho con ❤️ por el Equipo 40 · No Country Simulation S04
</p>
