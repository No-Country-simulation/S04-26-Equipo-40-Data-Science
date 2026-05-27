# Intent Detector Specification

## Purpose

Clasificar mensajes de clientes en 9 categorías de intención para ES y PT, usando XLM-R fine-tuneado desde el checkpoint pre-entrenado multilingüe.

## Requirements

### Requirement: 9-Class Intent Classification

The system MUST assign exactly one of these 9 intents per message:

| # | Intent | Description |
|---|--------|-------------|
| 1 | `cancelacion` | Solicitud de cancelación |
| 2 | `consulta_general` | Consulta informativa general |
| 3 | `facturacion_pago` | Problemas de facturación o pago |
| 4 | `feedback` | Opinión o retroalimentación |
| 5 | `gestion_cuenta` | Cambios en cuenta/datos personales |
| 6 | `modificacion_pedido` | Cambios en pedido existente |
| 7 | `queja` | Reclamo o queja formal |
| 8 | `reembolso` | Solicitud de reembolso |
| 9 | `seguimiento` | Consulta de estado/seguimiento |

The model MUST achieve macro F1 > 0.85 on the held-out test set (ES + PT combined).

The system MUST handle ES and PT messages with the same model instance.

The system SHALL expose `predict_intent(text: str) -> dict` returning `{"intent": str, "probability": float, "probabilities": {str: float}}`.

#### Scenario: Cancelacion intent detection

- GIVEN the message "Quiero cancelar mi suscripción premium ya mismo"
- WHEN `predict_intent` is called
- THEN the returned intent MUST be `cancelacion` with probability > 0.7

#### Scenario: Reembolso in PT

- GIVEN the message "Quero meu dinheiro de volta, fiz o pagamento errado"
- WHEN `predict_intent` is called
- THEN the returned intent MUST be `reembolso`

#### Scenario: Low-coherence edge case

- GIVEN a meaningless input ("asdfgh qwerty 12345")
- WHEN `predict_intent` is called
- THEN the system MUST NOT raise an exception
- AND MUST return one of the 9 intents with probability >= 0.2
