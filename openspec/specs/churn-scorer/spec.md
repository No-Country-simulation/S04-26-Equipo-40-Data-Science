# Churn Scorer Specification

## Purpose

Calcular un score heurístico de riesgo de abandono (churn) por conversación, combinando sentimiento, intención y señales de frustración.

## Requirements

### Requirement: Heuristic Churn Score

The system MUST compute a churn score in [0.0, 1.0] using the formula:

`churn = w1*(1 - sent_prob_negative) + w2*frust_flag + w3*max(intent_prob for {queja, cancelacion, reembolso})`

Weights w1, w2, w3 MUST be configurable (default calibrated from validation data).

The system MUST expose each component separately: `sentiment_contribution`, `frustration_contribution`, `intent_contribution`, and `aggregate_score`.

The system MUST aggregate multi-message conversations using max score (worst-message-wins).

The frustration signal MUST be detected via keyword matching in ES and PT combined with the probability of negative sentiment.

#### Scenario: High churn risk conversation

- GIVEN a conversation with strong negative sentiment, cancelacion intent, and frustration keywords
- WHEN the churn scorer computes the score
- THEN the aggregate_score MUST be >= 0.7

#### Scenario: Low churn risk conversation

- GIVEN a conversation with neutral/positive sentiment and no frustration signals
- WHEN the churn scorer computes the score
- THEN the aggregate_score MUST be <= 0.2

#### Scenario: Component interpretability

- GIVEN any conversation
- WHEN the churn scorer returns a score
- THEN the output MUST include all three component values AND the aggregate
