# ============================================
# CONVERSAI - NOTEBOOK 2: ENTRENAMIENTO
# ============================================
# Entrena los modelos de sentimiento e intenciones
# usando los datasets procesados del Notebook 1
#
# Ejecutar en Google Colab con GPU

# 1. MONTAR DRIVE
from google.colab import drive
drive.mount('/content/drive')

# 2. INSTALAR DEPENDENCIAS
!pip install -q datasets transformers torch scikit-learn pandas numpy tqdm accelerate huggingface_hub

# 3. CONFIGURAR
import os
import json
import torch
import numpy as np
from datetime import datetime
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForZeroShotClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from huggingface_hub import login

# ⚠️ IMPORTANTE: Usa tu token personal de HF
# Obtén uno en: https://huggingface.co/settings/tokens
token_hf = "TU_TOKEN_AQUI"
login(token=token_hf)

# Paths
BASE_PATH = "/content/drive/MyDrive/ConversaAI/proyecto"
DATA_PATH = os.path.join(BASE_PATH, "data")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
MODELS_PATH = os.path.join(BASE_PATH, "models")
CHECKPOINTS_PATH = os.path.join(BASE_PATH, "checkpoints")
LOGS_PATH = os.path.join(BASE_PATH, "logs")

for path in [MODELS_PATH, CHECKPOINTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)


class ModeloTrainer:
    """Entrenador de modelos para ConversaAI"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Dispositivo: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ==================== SENTIMIENTO ====================

    def preparar_datos_sentimiento(self):
        """Prepara datos del corpus_ecommerce"""
        print("\n📊 Preparando datos de sentimiento...")

        try:
            ds_sentiment = load_from_disk(os.path.join(PROCESSED_PATH, "sentiment"))
            print(f"   ✅ Dataset cargado: {len(ds_sentiment['train']) + len(ds_sentiment['test'])} ejemplos")

            # Filtrar ejemplos muy cortos
            def filter_short(examples):
                return [len(text) > 10 for text in examples['text']]

            ds_sentiment = ds_sentiment.filter(filter_short)

            return ds_sentiment

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

    def tokenizar(self, dataset, tokenizer_name, max_length=128):
        """Tokeniza el dataset"""
        print(f"   🔤 Tokenizando con {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )

        tokenized = dataset.map(tokenize_function, batched=True)
        return tokenized, tokenizer

    def compute_metrics(self, eval_pred):
        """Calcula métricas de evaluación"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')

        return {'accuracy': accuracy, 'f1': f1}

    def entrenar_sentimiento(self):
        """Entrena modelo de sentimiento con cardiffnlp"""
        print("\n" + "="*60)
        print("😊 ENTRENANDO MODELO DE SENTIMIENTO")
        print("="*60)

        # Cargar datos
        dataset = self.preparar_datos_sentimiento()
        if dataset is None:
            print("   ❌ No se pudo cargar el dataset")
            return None, None

        # Modelo propuesto: cardiffnlp/twitter-xlm-roberta-base-sentiment
        # Este modelo ya está pre-entrenado para sentiment en múltiples idiomas
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        print(f"\n📦 Modelo base: {model_name}")

        # Tokenizar
        tokenized_dataset, tokenizer = self.tokenizar(dataset, model_name, max_length=128)

        # Cargar modelo (3 etiquetas: negative, neutral, positive)
        # Haremos fine-tuning para binary (negative/positive)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # 0: positive, 1: negative
        ).to(self.device)

        # Configurar entrenamiento
        output_dir = os.path.join(CHECKPOINTS_PATH, "sentiment")
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=os.path.join(LOGS_PATH, "sentiment"),
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # Entrenar
        print("\n🚀 Iniciando entrenamiento...")
        trainer.train()

        # Evaluar
        print("\n📊 Evaluando modelo...")
        eval_results = trainer.evaluate()

        # Predicciones para matriz de confusión
        predictions = trainer.predict(tokenized_dataset['test'])
        preds = np.argmax(predictions.predictions, axis=1)

        print("\n📈 Resultados:")
        print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"   F1 Score: {eval_results['eval_f1']:.4f}")

        # Guardar modelo
        model_path = os.path.join(MODELS_PATH, "sentiment_model")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)

        # Guardar métricas
        metricas = {
            'modelo': model_name,
            'tipo': 'sentiment',
            'fecha': datetime.now().isoformat(),
            'accuracy': eval_results['eval_accuracy'],
            'f1': eval_results['eval_f1'],
            'train_size': len(tokenized_dataset['train']),
            'test_size': len(tokenized_dataset['test'])
        }

        with open(os.path.join(model_path, 'metricas.json'), 'w') as f:
            json.dump(metricas, f, indent=2)

        print(f"\n✅ Modelo guardado en: {model_path}")

        return trainer, eval_results

    # ==================== INTENCIONES ====================

    def preparar_intenciones_candidates(self):
        """Carga las intenciones de Bitext como candidatos para zero-shot"""
        print("\n🎯 Preparando candidatos para intenciones...")

        intents_path = os.path.join(PROCESSED_PATH, "intent_candidates.json")
        if os.path.exists(intents_path):
            with open(intents_path, 'r') as f:
                intents = json.load(f)
            print(f"   ✅ {len(intents)} intenciones candidatas")
            return intents
        else:
            print("   ⚠️ No se encontró archivo de intenciones")
            return []

    def setup_zero_shot_intent(self):
        """Configura modelo zero-shot para intenciones"""
        print("\n" + "="*60)
        print("🎯 CONFIGURANDO ZERO-SHOT INTENT CLASSIFIER")
        print("="*60)

        # Obtener candidatos
        intents = self.preparar_intenciones_candidates()

        if not intents:
            print("❌ No hay candidatos de intenciones")
            return None

        # Usar modelo multilingual para zero-shot
        model_name = "facebook/bart-large-mnli"
        print(f"\n📦 Modelo: {model_name}")

        try:
            classifier = AutoModelForZeroShotClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Guardar configuración
            config = {
                'model_name': model_name,
                'intents': intents,
                'tipo': 'zero-shot-intent'
            }

            model_path = os.path.join(MODELS_PATH, "intent_model")
            os.makedirs(model_path, exist_ok=True)

            with open(os.path.join(model_path, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)

            print(f"✅ Modelo zero-shot configurado")
            print(f"   Candidatos: {len(intents)} intenciones")
            print(f"   Guardado en: {model_path}")

            return classifier, tokenizer

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_intent_classifier(self, classifier, tokenizer, intents):
        """Prueba el clasificador de intenciones"""
        print("\n🧪 Probando clasificador de intenciones...")

        test_queries = [
            "Quiero cancelar mi pedido",
            "Cuánto tiempo demora el envío?",
            "Tengo un problema con mi pago",
            "Quiero hablar con un agente"
        ]

        results = []
        for query in test_queries:
            result = classifier(
                query,
                candidate_labels=intents[:10],  # Usar subset para velocidad
                hypothesis_template="El usuario quiere {}."
            )

            results.append({
                'query': query,
                'intent': result['labels'][0],
                'score': result['scores'][0]
            })

            print(f"   '{query}' → {result['labels'][0]} ({result['scores'][0]:.2f})")

        return results


# ==================== EJECUTAR ====================

print("="*60)
print("🚀 CONVERSAI - ENTRENAMIENTO DE MODELOS")
print("="*60)

trainer_obj = ModeloTrainer()

# 1. Entrenar sentimiento
trainer_sent, metrics_sent = trainer_obj.entrenar_sentimiento()

# 2. Configurar zero-shot para intenciones
intent_model = trainer_obj.setup_zero_shot_intent()

if intent_model:
    classifier, tokenizer = intent_model
    intents = trainer_obj.preparar_intenciones_candidates()

    # Probar el clasificador
    results = trainer_obj.test_intent_classifier(classifier, tokenizer, intents)

print("\n" + "="*60)
print("✅ ENTRENAMIENTO COMPLETO")
print("="*60)
print(f"📁 Modelos guardados en: {MODELS_PATH}")
print(f"📁 Checkpoints en: {CHECKPOINTS_PATH}")
print("\n💡 Listo para ejecutar Notebook 3 (Evaluación y Dashboard)")