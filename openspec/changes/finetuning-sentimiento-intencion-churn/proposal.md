# Propuesta: Fine-tuning XLM-R para Sentimiento e Intención + Churn

## Intent

Pipeline actual (TF-IDF+LR sentimiento ~94.49%, BART zero-shot intención ~12s/inferencia, keywords churn) no escala. Necesitamos un pipeline fine-tuneado unificado con XLM-R base (277M params) que cubra 3 clases sentimiento, 9 clases intención, y score heurístico de churn. Entrenamiento en Kaggle T4x2, almacenamiento en HF Hub (`Rosela/`).

## Scope

### In Scope
- Fine-tune XLM-R sentimiento 3 clases (328k muestras: corpus_ecommerce PT + amazon_reviews_multi ES + neutros)
- Fine-tune XLM-R intención 9 clases (dataset limpio 31k, dedup, balanceado)
- Churn score: heurístico combinado sentimiento + intención + frustración
- Entrenamiento Kaggle, push_to_hub, dashboard actualizado, reporte recomendaciones

### Out of Scope
- Modelo supervisado de churn (sin datos etiquetados)
- Modelos >2B params (sin GPU suficiente)
- Despliegue producción
- Nuevos idiomas (solo PT/ES)

## Approach

1. **Sentimiento**: XLM-R base (`xlm-roberta-base`, no `cardiffnlp/twitter-xlm-roberta-base-sentiment`). corpus_ecommerce mapeado (0→neg, 2→pos) + amazon_reviews_multi ES (1-2★→neg, 3★→neu, 4-5★→pos). Split 80/10/10. Full dataset 328k (no reducido).
2. **Intención**: XLM-R base (`xlm-roberta-base`) fine-tuneado desde checkpoint pre-entrenado. Taxonomía 9 clases (intent_limpio: 31k limpio, sin duplicados ni placeholders, gestion_cuenta sampleada a ~4k para balancear). Dataset: 25k train, 3.1k val, 3.1k test.
3. **Churn**: `churn = w1*(1-sent_prob_negative) + w2*frust_flag + w3*max(intent_prob queja|cancelacion|reembolso)`. Pesos: w1=0.3, w2=0.35, w3=0.35.
4. **Plataforma**: Kaggle Trainer + push_to_hub a `Rosela/`. Batch_size=32 en T4 16GB.

## Affected Areas

| Area | Impact |
|------|--------|
| `notebooks/intent_xlmr.ipynb` | New (Kaggle, 9 clases) |
| `notebooks/sentiment_xlmr.ipynb` | New (Kaggle, 3 clases) |
| `scripts/predict.py` | Modified |
| `scripts/churn_scorer.py` | Existing |
| `dashboard_app.py` | Modified |
| `data/processed/` | Ready (intent_limpio, sentiment full) |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|------------|
| corpus_ecommerce sin neutro → bias clase | Resuelto | amazon_reviews_multi 3-star mapping + neutros sintéticos |
| Kaggle timeout 12h | Medium | Checkpoints parciales + resume |
| Dominio reseñas ≠ soporte real | Medium | Dataset incluye corpus_ecommerce conversacional; accuracy target >85% soporte real |
| XLM-R 1.1GB RAM en T4 | Low | Batch_size=32 cabe en 16GB |

## Rollback Plan

Modelos versionados en HF Hub (`Rosela/`). Dashboard anterior en git. Kaggle notebooks preservados como fallback.

## Dependencies

Kaggle account (T4x2), HF token write (`Rosela`), `transformers>=4.30`, `datasets`, `torch>=2.0`.

## Success Criteria

- [ ] Sentimiento: accuracy >92% en test amazon_reviews (real: ~92% en epoch 2.43)
- [ ] Sentimiento: accuracy >85% en subset soporte real
- [ ] Intención: F1 macro >0.85 en test 9 clases
- [ ] Churn score: correlación >0.5 con etiquetas manuales si existen
- [ ] Pipeline completo corre en Kaggle <6h
- [ ] Dashboard muestra los 3 scores por conversación
