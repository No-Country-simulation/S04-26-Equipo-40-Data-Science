# ConversaAI - Sistema de Análisis de Sentimiento e Intenciones

## 📋 Descripción del Proyecto

**ConversaAI** es un sistema de análisis de conversaciones de soporte al cliente que identifica:
- **Frustración** de usuarios
- **Intenciones** no resueltas
- **Patrones de abandono y escalación**

Este sistema permite al equipo de producto mejorar los flujos conversacionales basándose en datos concretos.

---

## 🎯 Objetivos del Proyecto

1. **Analizar 2M+ mensajes/mes** de conversaciones de soporte
2. **Detectar frustración** en tiempo real
3. **Identificar intenciones** que no se están resolviendo
4. **Predecir abandono** y escalación
5. **Generar insights accionables** para el equipo de producto

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE CONVERSAAI                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│  │    N1       │ ──► │    N2       │ ──► │    N3       │             │
│  │ Preparación │     │ Entrenamiento│    │ Dashboard   │             │
│  │   Datos     │     │   Modelos   │     │  Insights   │             │
│  └─────────────┘     └─────────────┘     └─────────────┘             │
│       │                   │                   │                        │
│       ▼                   ▼                   ▼                        │
│  corpus_ecommerce    sentiment_model    reporte.json                  │
│  (110k PT)           (cardiffnlp)        dashboard.png                 │
│  Bitext              intent_model                               │
│  (27 intents)       (zero-shot)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets

| Dataset | Uso | Registros | Idioma |
|---------|-----|-----------|--------|
| **corpus_ecommerce** | Sentimiento (fine-tuning) | 110,895 | Portugués |
| **Bitext** | Intenciones (zero-shot candidates) | 26,872 | Inglés |
| **Emotions** | Análisis emocional (referencia) | 16,000 | Inglés |

### corpus_ecommerce.csv (Dataset Principal)
- `text`: Texto del mensaje
- `sentiment`: positive/negative (pre-etiquetado)
- `lang`: pt (portugués)
- `split`: train/val/test
- `industry`: e-commerce

---

## 🤖 Modelos

### Sentimiento
- **Modelo base**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Tipo**: Fine-tuning binary (positive/negative)
- **Razón**: Pre-entrenado en tweets multilingual, mejor para lenguaje informal

### Intenciones
- **Modelo base**: `facebook/bart-large-mnli`
- **Tipo**: Zero-shot classification
- **Razón**: No requiere entrenamiento, usa Bitext como candidatos

---

## 📁 Estructura del Proyecto

```
proyecto/
├── data/
│   ├── raw/
│   │   ├── corpus_ecommerce.csv      # Datos principales
│   │   ├── bitext_train.parquet       # Intenciones (EN)
│   │   └── emotions_train.parquet     # Emociones (EN)
│   └── processed/
│       ├── sentiment/                  # Dataset sentiment procesados
│       ├── intent_candidates.json     # Lista de intenciones
│       └── metadatos.json
├── notebooks/
│   ├── N1PreparacionDatos.py          # Preparación de datos
│   ├── N2Entrenamiento.py              # Entrenamiento modelos
│   └── N3EvaluacionDashboard.py       # Dashboard y análisis
├── models/
│   ├── sentiment_model/                # Modelo de sentimiento
│   └── intent_model/                   # Modelo de intenciones
├── checkpoints/                       # Checkpoints entrenamiento
├── logs/                              # Logs de entrenamiento
├── dashboard_app.py                   # Dashboard Streamlit interactivo
├── requirements.txt                   # Dependencias Python
└── README.md                          # Este archivo
```

---

## 🚀 Cómo Ejecutar

### Opción 1: Google Colab (Entrenamiento)

1. **Subir proyecto a Drive**
   ```
   /content/drive/MyDrive/ConversaAI/proyecto/
   ```

2. **Ejecutar N1 - Preparación de Datos**
   - Montar Drive
   - Ejecutar celdas
   - Descarga y procesa datasets

3. **Ejecutar N2 - Entrenamiento**
   - Requiere GPU (Colab Pro recomendado)
   - Entrena modelo de sentimiento (~30 min)
   - Configura zero-shot para intenciones (~5 min)

4. **Ejecutar N3 - Dashboard**
   - Sin GPU requerido
   - Analiza corpus completo
   - Genera visualizaciones y reporte

### Opción 2: Streamlit Dashboard (Visualización Interactiva)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar dashboard
streamlit run dashboard_app.py
```

El dashboard incluye:
- 📊 Métricas principales en tiempo real
- 📈 Distribución de sentimiento
- 🎯 Intenciones con más frustración
- ⚠️ Señales de abandono
- 🔍 Explorador de mensajes
- 💡 Recomendaciones accionables

**Despliegue:**
- Local: `streamlit run dashboard_app.py`
- Streamlit Cloud: Conectar repo de GitHub
- Docker: Crear container con Dockerfile

---

### 📋 Flujo de Trabajo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUJO MENSUAL                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Data Analyst carga nuevo corpus                            │
│  2. Ejecuta N1 → N2 → N3 (Colab con GPU)                      │
│  3. Genera reporte.json + dashboard.png                        │
│  4. Equipo de Producto revisa dashboard (Streamlit)           │
│  5. Identifica prioridades del sprint                         │
│  6. Implementa mejoras en flujos                              │
│  7. Repetir monthly                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📈 Métricas del Sistema

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **Tasa de Frustración** | % de mensajes con frustración | < 15% |
| **Accuracy Sentimiento** | Precisión del clasificador de sentimiento | > 85% |
| **Intenciones No Resueltas** | Top 5 intenciones con más frustración | Identificar y reducir |
| **Señales de Abandono** | Mensajes con intención de abandonar | Detectar y intervenir |

---

## 🎯 Outputs del Dashboard

1. **Distribución de sentimiento** (pie chart)
2. **Scores de confianza** (histograma)
3. **Top intenciones con frustración** (bar chart)
4. **Mapa de calor** sentimiento por idioma
5. **Señales de abandono detectadas**
6. **Reporte JSON** con recomendaciones accionables

---

## 👥 Usuarios del Sistema

- **Equipo de Producto**: Identifica flujos con más frustración
- **Data Analyst**: Procesa corpus mensual, genera reportes
- **Customer Success**: Interviene en casos de escalación

---

## 📝 Licencia

Este proyecto es para uso interno de ConversaAI.

---

## 🔄 Changelog

### v1.0 (2025-05)
- Pipeline completo de 3 notebooks
- Modelo de sentimiento con corpus_ecommerce
- Zero-shot intent classification
- Dashboard con métricas de frustración

---

## ⚠️ Notas Importantes

1. **GPU requerido** para N2 (entrenamiento)
2. **Colab Pro** recomendado para mayor velocidad
3. **Token HF** requerido en los notebooks
4. Los modelos se guardan en Drive para persistencia

---

## 📚 Recursos

- [Hugging Face Datasets](https://huggingface.co/datasets)
- [CardiffNLP Models](https://huggingface.co/cardiffnlp)
- [Transformers Documentation](https://huggingface.co/docs/transformers)