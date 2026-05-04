# ConversaAI - Support Conversation Analytics

Pipeline de análisis de sentimiento e intención para **ConversaAI** que procesa **2M mensajes/mes** en **Español** y **Portugués**, detectando frustración y predicción de abandono (churn).

## Características

- 🇪🇸 **Español** y 🇧🇷 **Portugués** soportados
- 🧹 Preprocesamiento de texto con spaCy (limpieza, normalización)
- 🤖 **XLM-RoBERTa** para clasificación multi-clase (sentimiento + intención)
- 😤 Detección de frustración basada en patrones + sentimiento + historial
- 📉 Predicción de churn (abandono) con análisis de conversaciones
- 📊 Dashboard interactivo con Streamlit + Plotly
- 📝 Generador de informes HTML con recomendaciones prácticas

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd sentiment_analysis

# Crear entorno virtual (Python 3.11 recomendado)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelos de spaCy
python -m spacy download es_core_news_sm
python -m spacy download pt_core_news_sm
```

## Uso

### 1. Se requieren datos de conversaciones de soporte

`
### 2. Fine-tuning de modelos XLM-R

```bash
# Fine-tuning para intención (6 clases: reclamo, consulta, cancelacion, etc.)
python finetune_intent_xlmr.py

# Fine-tuning para sentimiento en conversaciones
python finetune_xlmr.py
```

**Nota**: En CPU tardan ~5-8 horas cada uno. Recomendado usar **Google Colab** (30-60 min).

### 3. Ejecutar el Dashboard

```bash
streamlit run src/dashboard/app.py
```

El dashboard muestra:
- 📈 Resumen general (métricas, distribución de intenciones/sentimiento)
- 🚨 Intenciones no resueltas por conversación
- 😤 Análisis de frustración (picos, tendencias, niveles)
- 💡 Recomendaciones prácticas y plan de acción

### 4. Generar informe de análisis

```bash
# Generar informe HTML (últimos 30 días)
python generate_report.py --format html --days 30

# Generar informe JSON
python generate_report.py --format json --days 30

# Filtrar por idioma
python generate_report.py --format html --lang es
```

Los informes se guardan en `data/report_YYYYMMDD_HHMMSS.html`.

## Estructura del Proyecto

```
sentiment_analysis/
├── requirements.txt                    # Dependencias Python
├── generate_support_data.py           # Generador de datos de soporte
├── finetune_xlmr.py                 # Fine-tuning XLM-R para sentimiento
├── finetune_intent_xlmr.py          # Fine-tuning XLM-R para intención
├── generate_report.py                # Generador de informes HTML/JSON
├── test_frustration.py               # Test del módulo de frustración
├── data/                             # Datos procesados
│   ├── support_conversations.csv     # Datos sintéticos de soporte (22k msg)
│   ├── support_conversations_es.csv  # Conversaciones en español
│   ├── support_conversations_pt.csv # Conversaciones en portugués
│   └── models/                      # Modelos fine-tuneados
│       ├── intent_xlmr/             # Modelo de intención
│       └── xlmr-sentiment/         # Modelo de sentimiento
├── src/
│   ├── utils/
│   │   └── lang_detect.py          # Detección de idioma (ES/PT)
│   ├── models/
│   │   ├── classifier.py            # Clasificador TF-IDF + LogReg
│   │   ├── bert_classifier.py       # Clasificador BERT (sentimiento)
│   │   ├── intent_classifier.py    # Clasificador de intención
│   │   └── frustration_churn.py   # Detección de frustración + churn
│   └── dashboard/
│       └── app.py                  # Dashboard Streamlit
└── tests/                           # Tests unitarios y E2E
```

## Modelos Utilizados

| Idioma | Modelo Base | Fine-tuned para | Clases |
|--------|--------------|----------------|-------|
| Español + Portugués | `xlm-roberta-base` | Sentimiento en soporte | positive, negative, neutral |
| Español + Portugués | `xlm-roberta-base` | Intención en soporte | reclamo, consulta, cancelacion, problema_tecnico, facturacion, resuelto |

## Datasets

- **Datos sintéticos**: Generados con `generate_support_data.py` (~22k mensajes)
- **Español**: Datos de soporte al cliente simulados
- **Portugués**: Datos de soporte ao cliente simulados

## Funcionalidades Implementadas

### ✅ Completadas
1. **Procesamiento de texto** (ES/PT) con spaCy ✅
2. **Clasificación de sentimiento** con métricas ✅
3. **Clasificación de intención** (6 clases) ✅
4. **Detección de frustración** con análisis de patrones ✅
5. **Predicción de churn** basada en frustración acumulada ✅
6. **Dashboard interactivo** con intenciones no resueltas y picos de frustración ✅
7. **Generador de informes** con recomendaciones prácticas ✅

### 🔄 En Progreso
- Fine-tuning XLM-R para intención (scripts listos, pendiente ejecución)
- Fine-tuning XLM-R para sentimiento en conversaciones (scripts listos, pendiente ejecución)

## Métricas de Rendimiento (Datos sintéticos)

- **Intención**: ~85-90% accuracy esperado (después de fine-tuning)
- **Sentimiento**: ~90-95% accuracy esperado (después de fine-tuning)
- **Detección de frustración**: Basada en reglas + ML
- **Predicción de churn**: Basada en análisis de ventanas de tiempo

## Requisitos

- Python 3.11+ (recomendado, evitar 3.14 muy nuevo o 3.8 muy viejo)
- 8GB RAM (para modelos transformers)
- 5GB espacio en disco
- Conexión a internet (descarga de modelos la primera vez)

