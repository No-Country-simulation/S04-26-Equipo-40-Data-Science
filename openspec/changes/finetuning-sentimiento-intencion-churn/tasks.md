# Tasks: finetuning-sentimiento-intencion-churn

Checklist de implementación para el pipeline unificado XLM-R (sentimiento 3 clases + intención 9 clases + churn heurístico).

---

## Fase 1: Infrastructure

### 1.1 [x] Preparar datos de sentimiento — `data/processed/`
Combinar corpus_ecommerce (mapear 0→neg, 2→pos) con amazon_reviews_multi ES (1-2★→neg, 3★→neu, 4-5★→pos). Split 80/10/10. Guardar en `data/processed/sentiment/`.
- **Resultado**: 328k total (294k train, 16.8k val, 16.7k test). Dataset completo, no reducido.
- **Dependencias**: corpus_ecommerce.csv, amazon_reviews_multi (HF)

### 1.2 [x] Preparar datos de intención — `data/processed/`
Consolidar datasets traducidos ES/PT en un dataset unificado con taxonomía de 9 clases. Limpiar: eliminar duplicados, placeholders `{{...}}`, balancear gestion_cuenta. Split 80/10/10.
- **Resultado**: `intent_limpio/` — 31,809 total (25,447 train, 3,181 val, 3,181 test). ~16k PT, ~15k ES.
- **Script**: `scripts/limpiar_intent.py`
- **Dependencias**: archivos en `data/translated/` (intents_9clases_*.csv)

### 1.3 [x] Configurar entorno Kaggle + HF Hub
Crear secrets de Kaggle y verificar acceso de escritura a HF Hub para `Rosela/`.
- **Nota**: Cambiado de `conversaai/` a `Rosela/` (usuario HF real)
- **Dependencias**: cuenta Kaggle con T4x2, HF token con permisos write

---

## Fase 2: Implementation

### 2.1 [ ] Crear `DataProcessor` — `scripts/data_processor.py`
Clase con:
- Preprocesamiento unificado ES/PT: lowercase, normalización NFKC, expansión de contracciones comunes
- Tokenización XLM-R (`xlm-roberta-base`) con padding y truncation a 128 tokens
- API: `process(text: str) -> dict` y `process_batch(texts: list[str], batch_size=32) -> list[dict]`
- **Dependencias**: `transformers.AutoTokenizer`
- **Depende de**: — (independiente)
- **Nota**: Existe `scripts/data_processor.py`, verificar que cumple spec

### 2.2 [ ] Crear `SentimentClassifier` — `scripts/predict.py`
Wrapper sobre XLM-R fine-tuneado (3 clases: negative/neutral/positive) que carga `Rosela/xlm-r-sentiment-espt` desde HF Hub con fallback a pipeline TF-IDF legacy.
- API: `predict_sentiment(text: str) -> {"label": str, "probability": float, "probabilities": dict}`
- Manejo de edge case: input vacío → `neutral` con probability 1.0
- **Dependencias**: `transformers.AutoModelForSequenceClassification`, `DataProcessor.tokenizer`
- **Depende de**: 2.1, 3.1
- **Nota**: Pendiente de que termine training sentiment en Kaggle

### 2.3 [ ] Crear `IntentClassifier` — `scripts/predict.py`
Wrapper sobre XLM-R fine-tuneado (9 clases taxonomía) que carga `Rosela/xlm-r-intent-espt` desde HF Hub con fallback a BART zero-shot legacy.
- API: `predict_intent(text: str) -> {"intent": str, "probability": float, "probabilities": dict}`
- Manejo de edge case: input sin sentido → retorna una de las 9 intents con probability >= 0.2
- **Dependencias**: `transformers.AutoModelForSequenceClassification`, `DataProcessor.tokenizer`
- **Depende de**: 2.1, 3.2

### 2.4 [x] Crear `ChurnScorer` — `scripts/churn_scorer.py`
Clase con score heurístico combinando sentimiento + frustración + intención.
- Fórmula: `churn = w1*sent_prob_negative + w2*frust_flag + w3*max(intent_prob for {queja, cancelacion, reembolso})`
- Pesos configurables (default: w1=0.3, w2=0.35, w3=0.35)
- Keyword lists ES/PT para detección de frustración
- API: `score(sentiment: dict, intent: dict, text: str) -> dict` con `aggregate_score`, `sentiment_contribution`, `frustration_contribution`, `intent_contribution`
- **Dependencias**: — (solo reglas + keywords)
- **Depende de**: — (independiente)
- **Estado**: Script existe, verificar API match con spec

### 2.5 [ ] Crear `ConversaAIPipeline` — `scripts/predict.py`
Orquestador unificado que carga todos los componentes y coordina el flujo de inferencia.
- API: `predict(text: str) -> dict` | `batch_predict(texts: list[str], batch_size=32) -> list[dict]` | `predict_conversation(messages: list[str]) -> dict` (worst-message-wins)
- Fallback: si HF Hub caído → log warning + llama legacy pipeline TF-IDF/BART
- Flujo: DataProcessor → SentimentClassifier → IntentClassifier → ChurnScorer
- **Dependencias**: 2.1, 2.2, 2.3, 2.4
- **Depende de**: 2.2, 2.3, 2.4 (todos listos)

### 2.6 [ ] Verificar tests inline — `scripts/predict.py`, `scripts/churn_scorer.py`, `scripts/data_processor.py`
Agregar bloques `if __name__ == "__main__"` con asserts que cubran los escenarios definidos en specs.
- **Depende de**: 2.5

---

## Fase 3: Training (Kaggle)

### 3.1 [~] Notebook fine-tuning Sentimiento XLM-R — `notebooks/sentiment_xlmr.ipynb`
Notebook Kaggle que:
- [x] Carga datos desde `data/processed/sentiment_*` (parquets)
- [x] Fine-tunea `xlm-roberta-base` (no `cardiffnlp/twitter-xlm-roberta-base-sentiment`) con head de 3 labels
- [x] Hiperparámetros: lr=2e-5, batch_size=32, epochs=3 (no 5), early stopping patience=2, AdamW, linear warmup 10%, FP16
- [~] Evalúa en test split: accuracy ~92% en epoch 2.43 (en progreso)
- [ ] Pushea a HF Hub: `Rosela/xlm-r-sentiment-espt`
- [ ] Checkpointing cada epoch
- **Dependencias**: `transformers>=4.30`, `datasets`, `torch>=2.0`, `accelerate`, `evaluate`
- **Depende de**: 1.1, 1.3
- **Estado**: ENTRENANDO en Kaggle (epoch 2.43/3, ~92% accuracy)

### 3.2 [~] Notebook fine-tuning Intención XLM-R — `notebooks/intent_xlmr.ipynb`
Notebook Kaggle que:
- [x] Notebook creado: `intent_xlmr_colab.ipynb` (Colab) y `intent_xlmr.ipynb.py` (Kaggle)
- [x] Dataset listo: `intent_limpio` (31k, 9 clases, clean)
- [ ] Fine-tunea `xlm-roberta-base` con head de 9 labels
- [ ] Hiperparámetros: lr=2e-5, batch_size=32, epochs=5, early stopping patience=2, FP16
- [ ] Evalúa en test set: target macro F1 > 0.85
- [ ] Pushea a HF Hub: `Rosela/xlm-r-intent-espt`
- **Dependencias**: `transformers>=4.30`, `datasets`, `torch>=2.0`, `accelerate`, `evaluate`
- **Depende de**: 1.2, 1.3
- **Estado**: Notebook listo, pendiente de ejecutar en Kaggle

### 3.3 [ ] Ejecutar notebooks en Kaggle
- [~] Sentiment corriendo (epoch 2.43/3)
- [ ] Intent pendiente
- [ ] Verificar métricas cumplen targets
- [ ] Modelos pusheados correctamente a HF Hub
- **Depende de**: 3.1, 3.2

---

## Fase 4: Integration

### 4.1 [ ] Extender evaluación comparativa — `scripts/evaluacion_modelo.py`
Modificar el script existente para cargar pipeline XLM-R y comparar contra TF-IDF+LR (~94.49%).
- **Dependencias**: 2.5, 3.3

### 4.2 [ ] Actualizar Dashboard — `dashboard_app.py`
Reemplazar métricas TF-IDF legacy por scores XLM-R. Nuevas secciones: histograma churn, stacked bar intents, tabla high-risk, toggle comparativo.
- **Dependencias**: 2.5 (ConversaAIPipeline listo)
- **Depende de**: 2.5, 3.3

### 4.3 [ ] Verificar integración completa
- `python scripts/predict.py` (carga de modelos, inferencia batch)
- `streamlit run dashboard_app.py` (visualizaciones cargan)
- `python scripts/evaluacion_modelo.py` (comparativa)
- **Depende de**: 4.1, 4.2

---

## Fase 5: Cleanup

### 5.1 [x] Eliminar `notebooks/N1PreparacionDatos.py`, `N2Entrenamiento.py`, `N3EvaluacionDashboard.py`
Archivos legacy de la v1 del pipeline. Reemplazados por notebooks Kaggle/Colab.
- **Depende de**: —

### 5.2 [ ] (BLOQUEADO) Eliminar `reports/reporte_conversaAI.py`
Verificar que `dashboard_app.py` ya no referencia este archivo.
- **Depende de**: 4.2

### 5.3 [ ] Verificación post-cleanup
- **Depende de**: 5.1, 5.2
