# Design: Fine-tuning XLM-R para Sentimiento e Intención + Churn

## 1. Arquitectura General

```
                         ┌─────────────────────────────────────┐
                         │         ConversaAIPipeline          │
                         │        (Orquestador unificado)      │
                         └─────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │ DataProcessor   │ │ DataProcessor   │ │ DataProcessor   │
          │ (preproc+tok)   │ │ (preproc+tok)   │ │ (preproc+tok)   │
          └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                   ▼                   ▼                   │
          ┌─────────────────┐ ┌─────────────────┐          │
          │ XLM-R Sentiment │ │ XLM-R Intent    │          │
          │ (3 classes)     │ │ (9 classes)     │          │
          └────────┬────────┘ └────────┬────────┘          │
                   │                   │                   │
                   └───────┬───────────┘                   │
                           │                               │
                           ▼                               │
                    ┌─────────────────┐                    │
                    │   ChurnScorer   │◄───────────────────┘
                    │ (heurístico)    │  (frustration flags)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ConversAIDash-  │
                    │ board (Streamlit)│
                    └─────────────────┘
```

## 2. Decisiones de Arquitectura

| Decisión | Opciones | Tradeoff | Elegido |
|----------|----------|----------|---------|
| Tokenizer compartido | 1 tokenizer vs 2 separados | Menos memoria, 1 cfg menos | **Compartido**: XLM-R tokenizer para ambos modelos (spec) |
| Churn: ML vs heurístico | XGBoost/LR vs fórmula | Sin datos etiquetados de churn → heurístico | **Heurístico** con pesos calibrables (spec) |
| Carga de modelos | `from_pretrained()` vs `torch.load()` | HF Hub versionado, fácil rollback | **HF Hub** con fallback a legacy TF-IDF/BART |
| Base model sentimiento | `cardiffnlp/...` vs `xlm-roberta-base` | Cardiff pre-fine-tuneado en sentiment vs empezar de cero | **`xlm-roberta-base`** — decide usar base para tener control total |
| HF Hub org | `conversaai/` vs `Rosela/` | Organización vs cuenta personal | **`Rosela/`** — usuario real de HF |
| Batch size | 16 vs 32 | Memoria vs velocidad | **32** — cabe en T4 16GB |
| Épocas | 5 vs 3 | Más épocas puede overfittear | **3** — suficiente con early stopping |
| Dataset sentimiento | Full (328k) vs reducido (178k) | Velocidad vs precisión | **Full (328k)** — priorizar precisión sobre tiempo |

## 3. Componentes de Software

### a) DataProcessor
- **Responsabilidad**: Preprocesamiento unificado ES/PT (lowercase, NFKC, expandir contracciones) + tokenización XLM-R (padding, truncation a 128 tokens)
- **API**: `process(text: str) -> dict` | `process_batch(texts: list[str], batch_size=32) -> list[dict]`
- **Dependencias**: `transformers.AutoTokenizer` (xlm-roberta-base)
- **Estado**: Script existe (`scripts/data_processor.py`), verificar API match

### b) SentimentClassifier
- **Responsabilidad**: Wrapper sobre XLM-R fine-tuneado (3 clases: negative/neutral/positive)
- **API**: `predict_sentiment(text: str) -> {"label": str, "probability": float, "probabilities": dict}`
- **Dependencias**: `transformers.AutoModelForSequenceClassification` + DataProcessor.tokenizer
- **Modelo HF**: `Rosela/xlm-r-sentiment-espt`
- **Estado**: En training (Kaggle, epoch 2.43/3)

### c) IntentClassifier
- **Responsabilidad**: Wrapper sobre XLM-R fine-tuneado (9 clases taxonomía)
- **API**: `predict_intent(text: str) -> {"intent": str, "probability": float, "probabilities": dict}`
- **Dependencias**: `transformers.AutoModelForSequenceClassification` + DataProcessor.tokenizer
- **Modelo HF**: `Rosela/xlm-r-intent-espt`
- **Dataset**: `intent_limpio` (31k, 9 clases, clean, balanced)
- **Estado**: Notebook listo, pendiente de ejecutar

### d) ChurnScorer
- **Responsabilidad**: Score heurístico combinando sentimiento + frustración + intención
- **API**: `score(sentiment: dict, intent: dict, text: str) -> {"aggregate_score": float, "sentiment_contribution": float, "frustration_contribution": float, "intent_contribution": float}`
- **Fórmula**: `churn = w1 * sent_prob_negative + w2 * frust_flag + w3 * max(intent_prob for {queja, cancelacion, reembolso})`
- **Dependencias**: Keyword lists ES/PT para frustración, pesos configurables (w1=0.3, w2=0.35, w3=0.35 default)
- **Estado**: Script existe (`scripts/churn_scorer.py`)

### e) ConversaAIPipeline
- **Responsabilidad**: Orquestador unificado que carga modelos, coordina predict, y maneja fallback
- **API**: `predict(text: str) -> dict` | `batch_predict(texts: list[str], batch_size=32) -> list[dict]` | `predict_conversation(messages: list[str]) -> dict` (agregación worst-message-wins)
- **Dependencias**: Todos los componentes anteriores
- **Fallback**: Si HF Hub caído → log warning + llama legacy pipeline TF-IDF/BART

### f) ConversaAIDashboard (modificado)
- **Responsabilidad**: Dashboard Streamlit con nuevas visualizaciones
- **API**: Carga `predict.py` via `ConversaAIPipeline`, reemplaza métricas TF-IDF legacy
- **Nuevas secciones**: histograma churn con slider de threshold, stacked bar intent distribution filtrable por sentimiento, tabla high-risk sortable con desglose de componentes, toggle side-by-side XLM-R vs legacy (MAY)

## 4. Flujo de Inferencia

```
Usuario/App
    │
    ▼
batch_predict(["Excelente servicio", "Péssimo atendimento"])
    │
    ▼
ConversaAIPipeline
    │
    ├── 1. DataProcessor.process_batch()
    │      ↓ lower, NFKC, expand, tokenize, pad (128)
    │
    ├── 2. SentimentClassifier.predict()  ──→ {"label":"positive","prob":0.97}
    │      ↓ XLM-R forward pass (3 clases)
    │
    ├── 3. IntentClassifier.predict()     ──→ {"intent":"feedback","prob":0.89}
    │      ↓ XLM-R forward pass (9 clases)
    │
    ├── 4. ChurnScorer.score()            ──→ {"aggregate":0.12, ...}
    │      ↓ w1*sent_prob_neg + w2*frust + w3*max(intents_riesgo)
    │
    └── 5. Output: 
         [{"text":"Excelente servicio","sentiment":"positive","intent":"feedback","churn":0.12},
          {"text":"Péssimo...","sentiment":"negative","intent":"queja","churn":0.87}]
```

## 5. Estrategia de Entrenamiento

### Modelos (sentimiento 3 clases, intención 9 clases)

| Parámetro | Sentimiento | Intención |
|-----------|-------------|-----------|
| Base model | `xlm-roberta-base` | `xlm-roberta-base` |
| Classification head | 3 labels (neg/neu/pos) | 9 labels |
| Learning rate | 2e-5 | 2e-5 |
| Batch size | 32 | 32 |
| Épocas máximas | 3 (con early stopping patience=2) | 3 (early stopping patience=2) |
| Optimizer | AdamW (weight_decay=0.01) | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10% steps) | Linear warmup (10% steps) |
| FP16 | Sí | Sí |
| Split | 80/10/10 train/val/test | 80/10/10 train/val/test |
| Métrica best | accuracy | macro F1 |
| Push to Hub | `Rosela/xlm-r-sentiment-espt` | `Rosela/xlm-r-intent-espt` |
| Checkpointing | Cada epoch, keep top 2 via HF Hub | Idem |
| Early stopping | Patience=2, eval cada epoch | Idem |

### Anti-Overfitting Measures
- **Early stopping** patience=2
- **Weight decay** 0.01
- **Dropout** — XLM-R base incluye dropout 0.1
- **Stratified split** 80/10/10
- **Monitor train vs val loss**
- **Anti-leak split** — en intent, split por texto único antes de tokenizar
- **Label smoothing** (sugerido: 0.1) si se observa overfitting

### Datos
- **Sentimiento**: 328k muestras (corpus_ecommerce PT + amazon_reviews_multi ES). Split 80/10/10.
- **Intención**: `intent_limpio` — 31,809 muestras (25,447 train, 3,181 val, 3,181 test). 9 clases. Limpio (sin duplicados, sin placeholders, balanceado).

## 6. Estrategia de Evaluación

| Etapa | Método | Métrica target |
|-------|--------|----------------|
| Offline test set | Evaluar en test split retenido (10%) | Sent: accuracy >92% / Intent: macro F1 >0.85 |
| Comparación baseline | Ejecutar `evaluacion_modelo.py` en mismo test split | XLM-R accuracy vs TF-IDF+LR (~94.49%) |
| Online dashboard | Dashboard con tabla de scores por conversación | Correlación churn >0.5 con etiquetas manuales |

## 7. Estado Actual (26 Mayo 2026)

| Componente | Estado | Notas |
|-----------|--------|-------|
| Sentiment training | ⏳ En Kaggle (epoch 2.43/3, ~92% acc) | Usando full dataset 328k, base xlm-roberta |
| Intent 9 clases | 📋 Notebook listo, pendiente Kaggle | Dataset intent_limpio 31k listo |
| Churn Scorer | ✅ Script existe | `scripts/churn_scorer.py` |
| DataProcessor | ✅ Script existe | `scripts/data_processor.py` |
| Dashboard | ❌ Pendiente | Requiere modelos trained + pusheados a HF |
| N1, N2, N3 | ✅ Eliminados | Legacy v1 |
| Limpieza repo | ✅ Parcial | archive/, xlmr-sentiment/, notebooks v1 borrados |

## 8. Archivos Afectados

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `notebooks/sentiment_xlmr.ipynb.py` | Crear | Fine-tuning XLM-R sentimiento 3 clases (Kaggle) |
| `notebooks/intent_xlmr.ipynb.py` | Crear | Fine-tuning XLM-R intención 9 clases (Kaggle) |
| `notebooks/intent_xlmr_colab.ipynb` | Crear | Versión Colab del intent 9 clases |
| `notebooks/sentiment_xlmr_colab.ipynb` | Crear | Versión Colab del sentiment 3 clases |
| `scripts/limpiar_intent.py` | Crear | Limpieza de dataset intent (intent_limpio) |
| `scripts/predict.py` | Modificar | `ConversaAIPipeline` + `SentimentClassifier` + `IntentClassifier` |
| `scripts/data_processor.py` | Crear | `DataProcessor` con preprocesamiento unificado |
| `scripts/churn_scorer.py` | Crear | `ChurnScorer` con fórmula heurística |
| `scripts/evaluacion_modelo.py` | Modificar | Añadir evaluación comparativa XLM-R vs TF-IDF |
| `dashboard_app.py` | Modificar | Reemplazar TF-IDF por scores XLM-R |
| `notebooks/N1PreparacionDatos.py` | ✅ Eliminado | Legacy |
| `notebooks/N2Entrenamiento.py` | ✅ Eliminado | Legacy |
| `notebooks/N3EvaluacionDashboard.py` | ✅ Eliminado | Legacy |

## 9. Decisiones Tomadas

| Pregunta | Decisión | Razón |
|----------|----------|-------|
| Base model sentimiento | `xlm-roberta-base` (no Cardiff) | Control total, mismo base que intent |
| HF Hub org | `Rosela/` | Cuenta real del usuario |
| Dataset sentimiento | Full (328k) no reducido (178k) | Máxima precisión posible |
| Batch size | 32 | Cabe en T4 16GB, acelera training |
| Épocas | 3 | Suficiente, early stopping si estanca |
| Notebooks v1 | Eliminados | Reemplazados por Kaggle/Colab notebooks |
