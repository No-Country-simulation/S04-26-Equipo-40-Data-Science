from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_HAS_HF = False
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification
    _HAS_HF = True
except ImportError:
    logger.warning("transformers/torch no disponible — solo modo fallback")


class SentimentClassifier:
    LABELS = ["negative", "neutral", "positive"]
    _FALLBACK_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
    _fallback_pipeline = None
    _fallback_lock = __import__("threading").Lock()

    def __init__(
        self,
        data_processor,
        model_name: str = "Rosela/xlm-r-sentiment-espt",
        fallback_to_legacy: bool = True,
    ):
        self.data_processor = data_processor
        self.model_name = model_name
        self.fallback_to_legacy = fallback_to_legacy
        self._model = None

        if _HAS_HF:
            try:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self._model.eval()
                logger.info("SentimentClassifier: Modelo XLM-R cargado desde HF Hub")
            except Exception as exc:
                logger.warning(
                    "SentimentClassifier: No se pudo cargar %s: %s", model_name, exc
                )
                if self.fallback_to_legacy:
                    logger.info("SentimentClassifier: Usando fallback (nlptown/zero-shot)")
                else:
                    raise RuntimeError(
                        f"No se pudo cargar {model_name} y fallback desactivado"
                    ) from exc
        elif fallback_to_legacy:
            logger.info("SentimentClassifier: transformers no disponible, usando fallback (nlptown/zero-shot)")
        else:
            raise RuntimeError(
                "transformers no disponible y fallback desactivado"
            )

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def predict_sentiment(self, text: str) -> dict:
        if self._model is not None:
            return self._predict_xlmr(text)
        return self._predict_fallback(text)

    def _predict_xlmr(self, text: str) -> dict:
        tokenized = self.data_processor.process(text)
        input_ids = torch.tensor([tokenized["input_ids"]])
        attention_mask = torch.tensor([tokenized["attention_mask"]])

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)

        probs_list = probabilities[0].tolist()
        confidence = max(probs_list)
        label_idx = probs_list.index(confidence)
        label = self.LABELS[label_idx]

        return {
            "label": label,
            "probability": confidence,
            "probabilities": {
                lbl: float(p) for lbl, p in zip(self.LABELS, probs_list)
            },
        }

    def _predict_fallback(self, text: str) -> dict:
        if self._fallback_pipeline is None:
            with self._fallback_lock:
                if self._fallback_pipeline is not None:
                    return self._predict_fallback(text)
                try:
                    from transformers import pipeline
                    self._fallback_pipeline = pipeline(
                        "sentiment-analysis",
                        model=self._FALLBACK_MODEL,
                        device=-1,
                    )
                except Exception as exc:
                    logger.warning(
                        "SentimentClassifier: No se pudo cargar nlptown: %s", exc
                    )
                    return {
                        "label": "neutral",
                        "probability": 1.0,
                        "probabilities": {
                            "negative": 0.0, "neutral": 1.0, "positive": 0.0,
                        },
                    }

        try:
            result = self._fallback_pipeline(text, truncation=True)[0]
            label_raw = result["label"]
            confidence = float(result["score"])

            if "LABEL" in label_raw:
                idx = int(label_raw.split("_")[1])
                if idx <= 1:
                    label = "negative"
                elif idx == 2:
                    label = "neutral"
                else:
                    label = "positive"
            elif "star" in label_raw:
                star = int(label_raw.split()[0])
                if star <= 2:
                    label = "negative"
                elif star == 3:
                    label = "neutral"
                else:
                    label = "positive"
            else:
                label = label_raw.lower()

            probs = {lbl: 0.0 for lbl in self.LABELS}
            probs[label] = confidence
            remainder = 1.0 - confidence
            for lbl in self.LABELS:
                if lbl != label:
                    probs[lbl] = remainder / (len(self.LABELS) - 1)

            return {
                "label": label,
                "probability": confidence,
                "probabilities": probs,
            }
        except Exception as exc:
            logger.warning(
                "SentimentClassifier: nlptown fallback error: %s", exc
            )
            return {
                "label": "neutral",
                "probability": 1.0,
                "probabilities": {
                    "negative": 0.0, "neutral": 1.0, "positive": 0.0,
                },
            }
