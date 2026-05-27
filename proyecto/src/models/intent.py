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


class IntentClassifier:
    LABELS = [
        "cancelacion", "consulta_general", "facturacion_pago", "feedback",
        "gestion_cuenta", "modificacion_pedido", "queja", "reembolso", "seguimiento",
    ]
    _FALLBACK_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli"
    _fallback_pipeline = None
    _fallback_lock = __import__("threading").Lock()

    def __init__(
        self,
        data_processor,
        model_name: str = "Rosela/xlm-r-intent-espt",
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
                logger.info("IntentClassifier: Modelo XLM-R cargado desde HF Hub")
            except Exception as exc:
                logger.warning(
                    "IntentClassifier: No se pudo cargar %s: %s", model_name, exc
                )
                if self.fallback_to_legacy:
                    logger.info("IntentClassifier: Usando fallback (nlptown/zero-shot)")
                else:
                    raise RuntimeError(
                        f"No se pudo cargar {model_name} y fallback desactivado"
                    ) from exc
        elif fallback_to_legacy:
            logger.info("IntentClassifier: transformers no disponible, usando fallback (nlptown/zero-shot)")
        else:
            raise RuntimeError(
                "transformers no disponible y fallback desactivado"
            )

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def predict_intent(self, text: str) -> dict:
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
        intent = self.LABELS[label_idx]

        return {
            "intent": intent,
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
                        "zero-shot-classification",
                        model=self._FALLBACK_MODEL,
                        device=-1,
                    )
                except Exception as exc:
                    logger.warning(
                        "IntentClassifier: No se pudo cargar zero-shot: %s", exc
                    )
                    probs = {intent: 0.0 for intent in self.LABELS}
                    probs["consulta_general"] = 1.0
                    return {
                        "intent": "consulta_general",
                        "probability": 1.0,
                        "probabilities": probs,
                    }

        try:
            result = self._fallback_pipeline(text, self.LABELS)
            intent = result["labels"][0]
            confidence = float(result["scores"][0])
            probabilities = {
                lbl: float(score)
                for lbl, score in zip(result["labels"], result["scores"])
            }

            return {
                "intent": intent,
                "probability": confidence,
                "probabilities": probabilities,
            }
        except Exception as exc:
            logger.warning(
                "IntentClassifier: zero-shot fallback error: %s", exc
            )
            probs = {intent: 0.0 for intent in self.LABELS}
            probs["consulta_general"] = 1.0
            return {
                "intent": "consulta_general",
                "probability": 1.0,
                "probabilities": probs,
            }
