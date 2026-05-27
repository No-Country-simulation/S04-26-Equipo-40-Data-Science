# Propuesta: Fine-tuning XLM-R para Sentimiento e Intención + Churn

## Intent

Pipeline actual (TF-IDF+LR sentimiento ~94.49%, BART zero-shot intención ~12s/inferencia, keywords churn) no escala. Necesitamos un pipeline fine-tuneado unificado con XLM-R base (277M params) que cubra 3 clases sentimiento, 9 clases intención, y score heurístico de churn. Entrenamiento en Kaggle T4x2, almacenamiento en HF Hub.

## Scope

### In Scope
- Fine-tune XLM-R sentimiento 3 clases (PT corpus_ecommerce + neutros sintéticos, ES amazon_reviews_multi)
- Fine-tune XLM-R intención 9 clases (datasets traducidos ES/PT ~27k c/u)
- Churn score: heurístico combinado sentimiento + intención + frustración
- Entrenamiento Kaggle, push_to_hub, dashboard actualizado, reporte recomendaciones

### Out of Scope
- Modelo supervisado de churn (sin datos etiquetados)
- Modelos >2B params (sin GPU suficiente)
- Despliegue producción
- Nuevos idiomas (solo PT/ES)

## Approach

1. **Sentimiento**: XLM-R fine-tuneado desde `cardiffnlp/twitter-xlm-roberta-base-sentiment`. corpus_ecommerce mapeado (0→neg, 1→pos) + neutros sintéticos (backtranslation). amazon_reviews_multi ES mapeado estrellas 1-2→0, 3→1, 4-5→2. Split 80/10/10.
2. **Intención**: XLM-R base (`xlm-roberta-base`) fine-tuneado desde el checkpoint pre-entrenado en lenguaje. Taxonomía: cancelacion, consulta_general, facturacion_pago, feedback, gestion_cuenta, modificacion_pedido, queja, reembolso, seguimiento. NOTA: fine-tuning (no entrenamiento desde cero) — arranca de pesos pre-entrenados en 100+ idiomas.
3. **Churn**: `churn = w1*(1-sent_prob) + w2*frust_flag + w3*(intent_prob queja|cancelacion|reembolso)`. Pesos por calibrar.
4. **Plataforma**: Kaggle `Trainer` + `push_to_hub`. Sin almacenamiento local. Batch_size=16 cabe en T4 16GB.

## Affected Areas

| Area | Impact |
|------|--------|
| `notebooks/sentiment_xlmr.ipynb` | New |
| `notebooks/intent_xlmr.ipynb` | New |
| `scripts/predict.py` | Modified |
| `dashboard_app.py` | Modified |
| `data/processed/` | Modified |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|------------|
| corpus_ecommerce sin neutro → bias clase | High | Neutros sintéticos con aumentación |
| Kaggle timeout 12h | Medium | Checkpoints parciales + resume desde Hub |
| Dominio reseñas ≠ soporte real | Medium | Evaluar en soporte real; accuracy target >85% |
| XLM-R 1.1GB RAM en T4 | Low | Batch_size=16, cabe en 16GB |

## Rollback Plan

Modelos versionados en HF Hub. Dashboard anterior en git. Kaggle notebooks preservados como fallback. Volver al pipeline TF-IDF + BART es revertir `predict.py` y `dashboard_app.py` por git.

## Dependencies

Kaggle account (T4x2), HF token write, `transformers>=4.30`, `datasets`, `torch>=2.0`.

## Success Criteria

- [ ] Sentimiento: accuracy >95% en test amazon_reviews
- [ ] Sentimiento: accuracy >85% en subset soporte real
- [ ] Intención: F1 macro >0.85 en test 9 clases
- [ ] Churn score: correlación >0.5 con etiquetas manuales si existen
- [ ] Pipeline completo corre en Kaggle <6h
- [ ] Dashboard muestra los 3 scores por conversación
