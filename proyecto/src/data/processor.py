"""
DataProcessor — Preprocesamiento unificado ES/PT + tokenización XLM-R.

Responsabilidad:
  - Normalización de texto (lowercase, NFKC, espacios múltiples)
  - Tokenización con XLM-R (padding, truncation a max_length)
  - Procesamiento individual y por batches

Dependencias:
  - transformers.AutoTokenizer
"""

from typing import List
import re
import unicodedata

from transformers import AutoTokenizer


class DataProcessor:
    """Procesador de texto que normaliza y tokeniza usando XLM-R."""

    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 128):
        """
        Inicializa el tokenizer de XLM-R.

        Args:
            model_name: Nombre del modelo HF para el tokenizer.
            max_length: Longitud máxima de tokenización (padding/truncation).
        """
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # XLM-R no tiene pad_token por defecto; usamos eos_token como fallback
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normaliza un texto: lowercase, NFKC, elimina espacios múltiples, strip.

        Args:
            text: Texto de entrada.

        Returns:
            Texto normalizado.
        """
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process(self, text: str) -> dict:
        """
        Procesa un texto individual: normaliza y tokeniza.

        Args:
            text: Texto de entrada.

        Returns:
            Dict tokenizado con input_ids, attention_mask, etc.
        """
        normalized = self.normalize_text(text)
        tokenized = self._tokenizer(
            normalized,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # return Python lists/dicts
        )
        return tokenized

    def process_batch(self, texts: List[str], batch_size: int = 16) -> List[dict]:
        """
        Procesa una lista de textos en batches: normaliza cada texto y tokeniza por batch.

        Args:
            texts: Lista de textos a procesar.
            batch_size: Tamaño del batch para tokenización.

        Returns:
            Lista de dicts tokenizados.
        """
        results: List[dict] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            normalized_batch = [self.normalize_text(t) for t in batch_texts]

            tokenized = self._tokenizer(
                normalized_batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )

            # Convertir de dict of lists a list of dicts
            for j in range(len(normalized_batch)):
                results.append(
                    {k: tokenized[k][j] for k in tokenized.keys()}
                )

        return results

    def get_tokenizer(self) -> AutoTokenizer:
        """Retorna el tokenizer interno."""
        return self._tokenizer


if __name__ == "__main__":
    print("🧪 Running DataProcessor inline tests...\n")

    # Test 1: Normalización
    dp = DataProcessor()
    result = dp.normalize_text("  HOLA   MUNDO!!  ")
    expected = "hola mundo!!"
    assert result == expected, f"normalize_text failed: {result!r} != {expected!r}"
    print(f"✅ Test 1 — normalize_text: {result!r}")

    # Test 2: Procesar texto simple no debe fallar
    tokenized = dp.process("Excelente servicio, muchas gracias")
    assert "input_ids" in tokenized, "process() debe retornar 'input_ids'"
    assert "attention_mask" in tokenized, "process() debe retornar 'attention_mask'"
    assert len(tokenized["input_ids"]) == 128, (
        f"input_ids debe tener largo {dp.max_length}, "
        f"tiene {len(tokenized['input_ids'])}"
    )
    print(f"✅ Test 2 — process texto simple: input_ids length = {len(tokenized['input_ids'])}")

    # Test 3: Procesar texto vacío no debe fallar
    empty_tokenized = dp.process("")
    assert "input_ids" in empty_tokenized, "process('') debe retornar 'input_ids'"
    assert "attention_mask" in empty_tokenized, "process('') debe retornar 'attention_mask'"
    assert len(empty_tokenized["input_ids"]) == 128, (
        f"input_ids de texto vacío debe tener largo {dp.max_length}, "
        f"tiene {len(empty_tokenized['input_ids'])}"
    )
    print(f"✅ Test 3 — process texto vacío: input_ids length = {len(empty_tokenized['input_ids'])}")

    # Test 4: process_batch
    texts = ["Primer mensaje", "Segundo mensaje", "Tercer mensaje"]
    batch_results = dp.process_batch(texts, batch_size=2)
    assert len(batch_results) == 3, f"process_batch debe retornar 3 resultados, obtuvo {len(batch_results)}"
    for i, res in enumerate(batch_results):
        assert "input_ids" in res, f"Resultado {i} debe tener 'input_ids'"
        assert "attention_mask" in res, f"Resultado {i} debe tener 'attention_mask'"
        assert len(res["input_ids"]) == 128, (
            f"Resultado {i} input_ids debe tener largo {dp.max_length}, "
            f"tiene {len(res['input_ids'])}"
        )
    print(f"✅ Test 4 — process_batch: {len(batch_results)} resultados correctos")

    # Test 5: get_tokenizer
    tokenizer = dp.get_tokenizer()
    assert tokenizer is not None, "get_tokenizer() no debe retornar None"
    assert tokenizer.pad_token is not None, "pad_token no debe ser None"
    print(f"✅ Test 5 — get_tokenizer: {type(tokenizer).__name__}, pad_token={tokenizer.pad_token!r}")

    print("\n🎉 All tests passed!")
