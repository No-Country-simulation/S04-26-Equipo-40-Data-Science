# Sentiment Classifier Specification

## Purpose

Clasificar mensajes de soporte al cliente en 3 clases de sentimiento (negativo, neutro, positivo) para ES y PT, usando XLM-R fine-tuneado.

## Requirements

### Requirement: Multi-class Classification

The system MUST output exactly one label per message: `negative`, `neutral`, or `positive`.

The system MUST support both ES and PT inputs without language routing.

The trained model MUST achieve accuracy > 92% on the `amazon_reviews_multi` ES test split.

The model SHOULD achieve accuracy > 85% on a held-out subset of real support conversations.

The system SHALL expose a `predict_sentiment(text: str) -> dict` returning `{"label": str, "probability": float, "probabilities": {str: float}}`.

The model SHALL use `xlm-roberta-base` as the base checkpoint (NOT `cardiffnlp/twitter-xlm-roberta-base-sentiment`).

The training data SHALL use 3 classes mapped as:
- `negative` (0): 1-2 star reviews, negative labeled conversations
- `neutral` (1): 3 star reviews, synthetic neutral generation
- `positive` (2): 4-5 star reviews, positive labeled conversations

#### Scenario: Positive sentiment in ES

- GIVEN the message "Excelente servicio, muy rápido y eficiente"
- WHEN `predict_sentiment` is called
- THEN the returned label MUST be `positive` with probability > 0.85

#### Scenario: Negative sentiment in PT

- GIVEN the message "Péssimo atendimento, nunca mais compro aqui"
- WHEN `predict_sentiment` is called
- THEN the returned label MUST be `negative` with probability > 0.85

#### Scenario: Neutral detection for factual query

- GIVEN the message "Gostaria de saber o prazo de entrega do pedido 12345"
- WHEN `predict_sentiment` is called
- THEN the returned label MUST be `neutral`

#### Scenario: Empty input edge case

- GIVEN an empty string or whitespace-only input
- WHEN `predict_sentiment` is called
- THEN the system MUST NOT raise an exception
- AND MUST return `neutral` with probability 1.0
