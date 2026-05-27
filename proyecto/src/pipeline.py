from __future__ import annotations
import logging
import os
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)

from src.data.processor import DataProcessor
from src.features.churn import ChurnScorer
from src.models.sentiment import SentimentClassifier
from src.models.intent import IntentClassifier


_BASE = os.path.dirname(os.path.abspath(__file__))


class ConversaAIPipeline:
    def __init__(self, sentiment_model: str = None,
                 intent_model: str = None,
                 tokenizer_model: str = None):
        sentiment_model = sentiment_model or os.path.join(_BASE, "models", "sentiment")
        intent_model = intent_model or os.path.join(_BASE, "models", "intent")
        tokenizer_model = tokenizer_model or os.path.join(_BASE, "models", "sentiment")
        self.data_processor = DataProcessor(model_name=tokenizer_model)
        self.sentiment_classifier = SentimentClassifier(self.data_processor, model_name=sentiment_model)
        self.intent_classifier = IntentClassifier(self.data_processor, model_name=intent_model)
        self.churn_scorer = ChurnScorer()

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "text": text,
                "sentiment": {"label": "neutral", "probability": 1.0, "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}},
                "intent": {"intent": "consulta_general", "probability": 1.0, "probabilities": {lbl: 0.0 for lbl in IntentClassifier.LABELS}},
                "churn": {"aggregate_score": 0.0, "sentiment_contribution": 0.0, "frustration_contribution": 0.0, "intent_contribution": 0.0},
                "model_status": self.get_model_status(),
            }
        sentiment = self.sentiment_classifier.predict_sentiment(text)
        intent = self.intent_classifier.predict_intent(text)
        churn = self.churn_scorer.score(sentiment, intent, text)

        return {
            "text": text,
            "sentiment": sentiment,
            "intent": intent,
            "churn": churn,
            "model_status": self.get_model_status(),
        }

    def batch_predict(self, texts: List[str], batch_size: int = 16) -> List[dict]:
        results: List[dict] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for text in batch:
                results.append(self.predict(text))
        return results

    def predict_conversation(self, messages: List[str]) -> dict:
        processed = [self.predict(msg) for msg in messages]

        message_results = [
            {
                "text": p["text"],
                "sentiment": p["sentiment"],
                "intent": p["intent"],
                "churn": p["churn"],
            }
            for p in processed
        ]

        churn_scores = [p["churn"] for p in processed]
        aggregate_churn = self.churn_scorer.score_conversation(churn_scores)

        sentiments = [p["sentiment"]["label"] for p in processed]
        intents = [p["intent"]["intent"] for p in processed]

        dominant_sentiment = (
            Counter(sentiments).most_common(1)[0][0] if sentiments else "neutral"
        )
        dominant_intent = (
            Counter(intents).most_common(1)[0][0] if intents else "consulta_general"
        )

        return {
            "messages": message_results,
            "aggregate_churn": aggregate_churn,
            "dominant_sentiment": dominant_sentiment,
            "dominant_intent": dominant_intent,
        }

    def get_model_status(self) -> dict:
        return {
            "sentiment": (
                "xlm-r"
                if self.sentiment_classifier.is_model_loaded()
                else "fallback"
            ),
            "intent": (
                "xlm-r"
                if self.intent_classifier.is_model_loaded()
                else "fallback"
            ),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🧪 Running pipeline tests...\n")

    pipeline = ConversaAIPipeline()

    result_pos = pipeline.predict("Excelente servicio, muy bueno y fantástico")
    assert result_pos["sentiment"]["label"] == "positive"
    print(
        f"✅ Test 1 — Sentimiento positivo: "
        f"label={result_pos['sentiment']['label']}, "
        f"prob={result_pos['sentiment']['probability']:.4f}"
    )

    try:
        result_empty = pipeline.predict("")
        assert result_empty["sentiment"]["label"] == "neutral"
        assert result_empty["sentiment"]["probability"] == 1.0
        print(
            f"✅ Test 2 — Texto vacío: "
            f"sentiment={result_empty['sentiment']['label']}, "
            f"intent={result_empty['intent']['intent']}"
        )
    except Exception as exc:
        raise AssertionError(f"Test 2 failed: empty text raised {exc}")

    status = pipeline.get_model_status()
    assert isinstance(status, dict)
    assert "sentiment" in status
    assert "intent" in status
    assert status["sentiment"] in ("xlm-r", "fallback")
    assert status["intent"] in ("xlm-r", "fallback")
    print(
        f"✅ Test 3 — Model status: "
        f"sentiment={status['sentiment']}, intent={status['intent']}"
    )

    batch_results = pipeline.batch_predict(
        ["Excelente servicio", "Pésimo atención, no sirve"]
    )
    assert isinstance(batch_results, list)
    assert len(batch_results) == 2
    for i, res in enumerate(batch_results):
        assert "text" in res
        assert "sentiment" in res
        assert "intent" in res
        assert "churn" in res
        assert "model_status" in res
    print(f"✅ Test 4 — batch_predict: {len(batch_results)} resultados correctos")

    conv = pipeline.predict_conversation([
        "Excelente servicio, muchas gracias",
        "Pésimo, no sirve para nada, quiero cancelar",
        "Buen producto en general",
    ])
    assert isinstance(conv, dict)
    assert "messages" in conv
    assert "aggregate_churn" in conv
    assert "dominant_sentiment" in conv
    assert "dominant_intent" in conv
    assert len(conv["messages"]) == 3
    assert "aggregate_score" in conv["aggregate_churn"]
    assert isinstance(conv["dominant_sentiment"], str)
    assert isinstance(conv["dominant_intent"], str)
    print(
        f"✅ Test 5 — predict_conversation: {len(conv['messages'])} mensajes, "
        f"sentimiento dominante={conv['dominant_sentiment']}, "
        f"intent dominante={conv['dominant_intent']}, "
        f"churn agregado={conv['aggregate_churn']['aggregate_score']:.4f}"
    )

    print("\n🎉 Todos los tests pasaron.")
