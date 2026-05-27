# Delta for Text Processing Pipeline

## ADDED Requirements

### Requirement: Text Preprocessing for ES/PT

The pipeline MUST apply these preprocessing steps before inference: lowercase, strip whitespace, normalize unicode (NFKC), and expand common ES/PT contractions.

### Requirement: Unified XLM-R Tokenization

The pipeline MUST use the XLM-R tokenizer (`xlm-roberta-base`) for BOTH sentiment and intent models — MUST NOT use separate tokenizers per model.

### Requirement: Batch Inference Support

The pipeline SHOULD support batch inference with configurable batch size (default 16) for GPU-efficient processing. Results MUST preserve input ordering.

### Requirement: HF Hub Model Loading

The pipeline MUST load both fine-tuned models from Hugging Face Hub using `from_pretrained()`. The Hub repository names MUST be configurable via environment variables or a config file.

### Requirement: Graceful Fallback

If a model fails to load (Hub unavailable, corrupted checkpoint), the pipeline MUST log a warning: "XLM-R model unavailable, falling back to legacy pipeline" and continue with TF-IDF/BART without crashing.

#### Scenario: Batch inference ordering

- GIVEN 100 messages in a list
- WHEN `batch_predict(messages)` is called with default batch_size=16
- THEN the pipeline MUST process in at most 7 batches (ceil(100/16))
- AND return results in the same order as input

#### Scenario: Hub model unavailable

- GIVEN HF Hub returns 503 or network is down
- WHEN the pipeline initializes the models
- THEN it MUST log the fallback warning message
- AND continue processing using legacy TF-IDF/BART pipeline
- AND NOT raise an unhandled exception

#### Scenario: Preprocessing of ES special chars

- GIVEN the input "¡HOLA! ¿Cómo estás?"
- WHEN the text enters the pipeline
- THEN it MUST be normalized to "¡hola! ¿cómo estás?" (lowercase, NFKC)
