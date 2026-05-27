#!/usr/bin/env python3
from __future__ import annotations
"""
ChurnScorer — Score heurístico de riesgo de abandono (churn).

Combina señales de:
  - Sentimiento negativo (probabilidad de clase "negative")
  - Frustración detectada por keywords ES/PT
  - Intención de riesgo (queja, cancelacion, reembolso)

Fórmula:
  churn = w1 * sent_prob_negative + w2 * frust_flag + w3 * max(intent_risk_probs)

Pesos default: w1=0.3, w2=0.35, w3=0.35
"""


# ---------------------------------------------------------------------------
# Keywords de frustración — combinación ES + PT
# ---------------------------------------------------------------------------
FRUSTRATION_KEYWORDS = frozenset({
    # Español
    "desisto", "nunca más", "pésimo", "terrible", "horrible",
    "inaceptable", "pésimo servicio", "malísimo", "deficiente",
    "no sirve", "no funciona", "queja", "reclamo", "protesta",
    "abandono", "cancelación", "procon",
    # Portugués
    "nunca mais", "péssimo", "terrível", "horrível", "inaceitável",
    "péssimo serviço", "péssimo atendimento", "não serve",
    "não funciona", "reclamação", "reclamo", "protesto",
    "abandono", "cancelamento", "procon",
})

# Intents consideradas de alto riesgo de churn
RISK_INTENTS = frozenset({"queja", "cancelacion", "reembolso"})


class ChurnScorer:
    """Evalúa riesgo de churn combinando sentimiento, frustración e intención."""

    def __init__(self, w1: float = 0.3, w2: float = 0.35, w3: float = 0.35):
        """
        Args:
            w1: peso para contribución de sentimiento negativo
            w2: peso para contribución de frustración (keywords)
            w3: peso para contribución de intención de riesgo
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    # -- Pesos configurables ------------------------------------------------

    def get_weights(self) -> dict:
        """Retorna pesos actuales."""
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3}

    def set_weights(self, w1: float, w2: float, w3: float) -> None:
        """Actualiza pesos en runtime."""
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    # -- Detección de frustración por keywords ------------------------------

    @staticmethod
    def _detect_frustration(text: str) -> float:
        """Retorna 1.0 si el texto contiene keywords de frustración, 0.0 si no.

        La comparación es case-insensitive y busca la keyword como substring
        (para capturar variaciones como "queja" dentro de "quejé", etc.).
        """
        text_lower = text.lower()
        for kw in FRUSTRATION_KEYWORDS:
            if kw in text_lower:
                return 1.0
        return 0.0

    # -- Score individual ---------------------------------------------------

    def score(self, sentiment: dict, intent: dict, text: str) -> dict:
        """Calcula churn score para un mensaje individual.

        Args:
            sentiment: {"label": str, "probability": float,
                        "probabilities": {str: float}}
            intent:    {"intent": str, "probability": float,
                        "probabilities": {str: float}}
            text:      Texto crudo del mensaje.

        Returns:
            dict con:
              - aggregate_score: float (churn compuesto en [0, 1])
              - sentiment_contribution: float
              - frustration_contribution: float
              - intent_contribution: float
        """
        # --- Sentimiento negativo ---
        sent_prob_negative = sentiment.get("probabilities", {}).get("negative", 0.0)
        sentiment_contribution = self.w1 * sent_prob_negative

        # --- Frustración por keywords ---
        frust_flag = self._detect_frustration(text)
        frustration_contribution = self.w2 * frust_flag

        # --- Intención de riesgo ---
        intent_probs = intent.get("probabilities", {})
        risk_values = [
            intent_probs.get(intent_name, 0.0)
            for intent_name in RISK_INTENTS
        ]
        max_risk_prob = max(risk_values) if risk_values else 0.0
        intent_contribution = self.w3 * max_risk_prob

        # --- Agregado ---
        aggregate_score = sentiment_contribution + frustration_contribution + intent_contribution

        # Acotar a [0, 1] por seguridad (puede exceder si pesos suman > 1)
        aggregate_score = min(max(aggregate_score, 0.0), 1.0)

        return {
            "aggregate_score": aggregate_score,
            "sentiment_contribution": sentiment_contribution,
            "frustration_contribution": frustration_contribution,
            "intent_contribution": intent_contribution,
        }

    # -- Score multi-mensaje (conversación) ---------------------------------

    def score_conversation(self, messages: list[dict]) -> dict:
        """Agrega scores individuales vía worst-message-wins (max).

        Args:
            messages: Lista de resultados de score().

        Returns:
            dict con aggregate_score siendo el máximo de los mensajes,
            y cada contribución del mensaje más riesgoso.
        """
        if not messages:
            return {
                "aggregate_score": 0.0,
                "sentiment_contribution": 0.0,
                "frustration_contribution": 0.0,
                "intent_contribution": 0.0,
            }

        worst = max(messages, key=lambda m: m["aggregate_score"])
        return dict(worst)


# ===========================================================================
# Tests inline
# ===========================================================================

if __name__ == "__main__":
    scorer = ChurnScorer()

    # --- Test 1: Churn alto ---
    # Cliente muy negativo, con intención de cancelación y keywords de frustración
    sentiment_high = {
        "label": "negative",
        "probability": 0.95,
        "probabilities": {"negative": 0.95, "neutral": 0.03, "positive": 0.02},
    }
    intent_high = {
        "intent": "cancelacion",
        "probability": 0.9,
        "probabilities": {
            "cancelacion": 0.9,
            "queja": 0.04,
            "reembolso": 0.01,
            "feedback": 0.03,
            "consulta": 0.01,
            "despedida": 0.005,
            "acuerdo": 0.003,
            "compromiso": 0.001,
            "solicitud": 0.001,
        },
    }
    text_high = "Desisto, quiero cancelar mi cuenta"

    result_high = scorer.score(sentiment_high, intent_high, text_high)
    assert result_high["aggregate_score"] >= 0.7, (
        f"Test 1 failed: aggregate_score={result_high['aggregate_score']:.4f} < 0.7"
    )
    assert "sentiment_contribution" in result_high
    assert "frustration_contribution" in result_high
    assert "intent_contribution" in result_high
    print(f"✅ Test 1 (churn alto): aggregate={result_high['aggregate_score']:.4f}  "
          f"(sent={result_high['sentiment_contribution']:.4f}, "
          f"frust={result_high['frustration_contribution']:.4f}, "
          f"intent={result_high['intent_contribution']:.4f})")

    # --- Test 2: Churn bajo ---
    # Cliente positivo, intención de feedback, sin frustración
    sentiment_low = {
        "label": "positive",
        "probability": 0.95,
        "probabilities": {"positive": 0.95, "neutral": 0.03, "negative": 0.02},
    }
    intent_low = {
        "intent": "feedback",
        "probability": 0.9,
        "probabilities": {
            "feedback": 0.9,
            "consulta": 0.04,
            "despedida": 0.02,
            "acuerdo": 0.01,
            "compromiso": 0.01,
            "solicitud": 0.01,
            "queja": 0.005,
            "cancelacion": 0.003,
            "reembolso": 0.002,
        },
    }
    text_low = "Excelente servicio"

    result_low = scorer.score(sentiment_low, intent_low, text_low)
    assert result_low["aggregate_score"] <= 0.2, (
        f"Test 2 failed: aggregate_score={result_low['aggregate_score']:.4f} > 0.2"
    )
    print(f"✅ Test 2 (churn bajo):  aggregate={result_low['aggregate_score']:.4f}  "
          f"(sent={result_low['sentiment_contribution']:.4f}, "
          f"frust={result_low['frustration_contribution']:.4f}, "
          f"intent={result_low['intent_contribution']:.4f})")

    # --- Test 3: Output incluye los 3 componentes + aggregate ---
    required_keys = {"aggregate_score", "sentiment_contribution",
                     "frustration_contribution", "intent_contribution"}
    assert required_keys.issubset(result_high.keys()), (
        f"Test 3 failed: missing keys {required_keys - result_high.keys()}"
    )
    assert required_keys.issubset(result_low.keys()), (
        f"Test 3 failed: missing keys {required_keys - result_low.keys()}"
    )
    print(f"✅ Test 3 (componentes): output contiene los 4 campos requeridos")

    # --- Test 4: score_conversation con worst-message-wins ---
    conv_result = scorer.score_conversation([result_low, result_high, result_low])
    assert conv_result["aggregate_score"] == result_high["aggregate_score"], (
        f"Test 4 failed: conv aggregate {conv_result['aggregate_score']} "
        f"!= max {result_high['aggregate_score']}"
    )
    print(f"✅ Test 4 (conversación): worst-message-wins = {conv_result['aggregate_score']:.4f}")

    # --- Test 5: get/set weights ---
    assert scorer.get_weights() == {"w1": 0.3, "w2": 0.35, "w3": 0.35}
    scorer.set_weights(0.5, 0.3, 0.2)
    assert scorer.get_weights() == {"w1": 0.5, "w2": 0.3, "w3": 0.2}
    scorer.set_weights(0.3, 0.35, 0.35)  # restaurar
    print(f"✅ Test 5 (pesos): get/set_weights funciona correctamente")

    # --- Test 6: Frustration keywords matching ---
    # PT keyword
    result_pt = scorer.score(sentiment_high, intent_low, "Péssimo atendimento, não funciona")
    assert result_pt["frustration_contribution"] > 0, (
        f"Test 6a failed: PT keyword not detected"
    )
    print(f"✅ Test 6a (keywords PT): frust={result_pt['frustration_contribution']:.4f}")

    # Sin keywords
    result_no_frust = scorer.score(sentiment_low, intent_low, "Todo está perfecto")
    assert result_no_frust["frustration_contribution"] == 0.0, (
        f"Test 6b failed: false positive frustration"
    )
    print(f"✅ Test 6b (sin keywords): frust={result_no_frust['frustration_contribution']:.4f}")

    # --- Test 7: Escenario intermedio ---
    sentiment_mid = {
        "label": "neutral",
        "probability": 0.7,
        "probabilities": {"negative": 0.3, "neutral": 0.7, "positive": 0.0},
    }
    intent_mid = {
        "intent": "queja",
        "probability": 0.6,
        "probabilities": {
            "queja": 0.6,
            "cancelacion": 0.1,
            "reembolso": 0.05,
            "feedback": 0.1,
            "consulta": 0.1,
            "despedida": 0.02,
            "acuerdo": 0.01,
            "compromiso": 0.01,
            "solicitud": 0.01,
        },
    }
    result_mid = scorer.score(sentiment_mid, intent_mid, "El servicio es malísimo")
    assert 0.2 < result_mid["aggregate_score"] < 0.8, (
        f"Test 7 failed: aggregate_score={result_mid['aggregate_score']:.4f} not in (0.2, 0.8)"
    )
    print(f"✅ Test 7 (intermedio): aggregate={result_mid['aggregate_score']:.4f}")

    print("\n🎯 Todos los tests pasaron.")
