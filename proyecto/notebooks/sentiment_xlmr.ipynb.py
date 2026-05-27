# %% [markdown]
# # ConversaAI — Fine-tuning XLM-R Sentimiento (3 clases)
#
# ## Objetivo
# Fine-tunear `xlm-roberta-base` para clasificar sentimiento
# en **3 clases (negative / neutral / positive)** para textos de soporte al cliente en ES y PT.
#
# ## Target
# - **Accuracy > 95%** en test set de amazon_reviews_multi ES
# - **Accuracy > 85%** en subset de conversaciones reales de soporte
#
# ## Entorno
# - **Plataforma**: Kaggle (GPU T4x2, 16 GB VRAM)
# - **Datos**: Parquets subidos como Kaggle Dataset o montados manualmente
# - **Checkpoints**: Pusheados a HF Hub (`Rosela/xlm-r-sentiment-espt`)
# - **Tiempo estimado**: ~2-3 horas (5 épocas, early stopping patience=2)
#
# ## Anti-Overfitting
# - Early stopping patience=2
# - Weight decay=0.01
# - Label smoothing=0.1
# - Stratified split 80/10/10
# - Monitor train vs val loss
# - NO evaluar en test set durante training

# %% [markdown]
# ## 1. Instalar dependencias

# %%
!pip install -q transformers datasets torch scikit-learn pandas numpy tqdm accelerate evaluate huggingface_hub pyarrow

# %% [markdown]
# ## 2. Configurar entorno Kaggle

# %%
import os
import json
import shutil
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# ── Cache de HuggingFace en Kaggle Working ──
# Kaggle tiene /kaggle/working/ con ~20 GB disponibles
# NO usar /root/.cache/ que se borra entre sesiones
os.environ["HF_HOME"] = "/kaggle/working/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/working/.cache/datasets"
os.environ["TORCH_HOME"] = "/kaggle/working/.cache/torch"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/.cache/huggingface"

# Crear estructura
for cache_dir in [
    "/kaggle/working/.cache/huggingface",
    "/kaggle/working/.cache/datasets",
    "/kaggle/working/.cache/torch",
    "/kaggle/working/logs",
    "/kaggle/working/checkpoints",
]:
    os.makedirs(cache_dir, exist_ok=True)

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("   ⚠️  NO GPU DETECTADA — El entrenamiento será MUY lento en CPU")

# %% [markdown]
# ## 3. Importaciones y login HF

# %%
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from huggingface_hub import login, whoami
import evaluate

# ── Seed para reproducibilidad ──
set_seed(42)

# ── Login Hugging Face ──
# ⚠️ Reemplazar con tu token de HF (https://huggingface.co/settings/tokens)
HF_TOKEN = "TU_TOKEN_HF_AQUI"
login(token=HF_TOKEN)
user_info = whoami()
print(f"✅ Login exitoso — {user_info['name']}")

# ── Repo en HF Hub ──
HUB_MODEL_ID = "Rosela/xlm-r-sentiment-espt"
print(f"📦 Hub model ID: {HUB_MODEL_ID}")

# %% [markdown]
# ## 4. Cargar datos
#
# Los datos deben estar en formato parquet en `data/processed/sentiment/` con splits:
# - `train.parquet`
# - `val.parquet`
# - `test.parquet`
#
# En Kaggle, se suben como dataset de Kaggle. Ajustar `DATA_PATH` según corresponda.

# %%
# ── Configurar ruta de datos ──
# En Kaggle: /kaggle/input/<dataset-name>/
# Local:     data/processed/sentiment/
import os
import pandas as pd

DATA_PATH = "/kaggle/input/conversaai-sentiment-data"
print("[+] Dataset Kaggle:", DATA_PATH)

print("[+] Archivos en", DATA_PATH)
for f in os.listdir(DATA_PATH):
    print("   ", f)

def load_parquet_safe(path, name):
    full_path = os.path.join(path, name)
    if not os.path.exists(full_path):
        print("   [!]", name, "no encontrado")
        return None
    df = pd.read_parquet(full_path)
    print("   [+]", name, ":", len(df), "ejemplos")
    return df

df_train = load_parquet_safe(DATA_PATH, "train.parquet")
df_val = load_parquet_safe(DATA_PATH, "val.parquet")
df_test = load_parquet_safe(DATA_PATH, "test.parquet")

if df_train is None:
    print("\n[!] No se encontraron splits. Cargando dataset completo...")
    full_path = os.path.join(DATA_PATH, "sentiment_full.parquet")
    if not os.path.exists(full_path):
        parquet_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(
                "No se encontraron archivos parquet en " + DATA_PATH
            )
        full_path = os.path.join(DATA_PATH, parquet_files[0])
        print("   Usando:", parquet_files[0])
    df_all = pd.read_parquet(full_path)
    print("   Total ejemplos:", len(df_all))
    expected_cols = {"text", "label"}
    if not expected_cols.issubset(df_all.columns):
        raise ValueError(
            "Columnas esperadas: " + str(expected_cols) + ", encontradas: " + str(set(df_all.columns))
        )
    X = df_all["text"].values
    y = df_all["label"].values
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
    )
    df_train = pd.DataFrame({"text": X_train, "label": y_train})
    df_val = pd.DataFrame({"text": X_val, "label": y_val})
    df_test = pd.DataFrame({"text": X_test, "label": y_test})
    print("\n   Split: train=", len(df_train), "val=", len(df_val), "test=", len(df_test))

label_names = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

print("\n[+] Distribucion de clases (train):")
train_dist = df_train["label"].value_counts().sort_index()
for label_id, count in train_dist.items():
    pct = count / len(df_train) * 100
    print("   ", label_id, "(" + label_names.get(label_id, "?") + "):", count, "(" + str(round(pct, 1)) + "%)")

print("\n[+] Distribucion de clases (val):")
val_dist = df_val["label"].value_counts().sort_index()
for label_id, count in val_dist.items():
    pct = count / len(df_val) * 100
    print("   ", label_id, "(" + label_names.get(label_id, "?") + "):", count, "(" + str(round(pct, 1)) + "%)")

print("\n[+] Distribucion de clases (test):")
test_dist = df_test["label"].value_counts().sort_index()
for label_id, count in test_dist.items():
    pct = count / len(df_test) * 100
    print("   ", label_id, "(" + label_names.get(label_id, "?") + "):", count, "(" + str(round(pct, 1)) + "%)")

print("\n[+] Total: train=", len(df_train), "val=", len(df_val), "test=", len(df_test))

dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train[["text", "label"]]),
    "val": Dataset.from_pandas(df_val[["text", "label"]]),
    "test": Dataset.from_pandas(df_test[["text", "label"]]),
})
print("\n[+] DatasetDict creado")

# %% [markdown]
# ## 5. Tokenizer

# %%
# Usamos el tokenizer de xlm-roberta-base (NO el de cardiffnlp específicamente,
# porque XLM-R usa el mismo tokenizer para todos los checkpoints)
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# XLM-R no tiene pad_token por defecto; usamos eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"🔤 Tokenizer: {MODEL_NAME}")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"   Max length: {MAX_LENGTH}")

# ── Función de tokenización ──
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

# Tokenizar datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remover columna 'text' (no la necesita el modelo, solo los tokens)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Renombrar 'label' a 'labels' (convención de Transformers)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Formato PyTorch
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

print(f"✅ Datasets tokenizados:")
print(f"   Train: {len(tokenized_datasets['train'])} ejemplos")
print(f"   Val:   {len(tokenized_datasets['val'])} ejemplos")
print(f"   Test:  {len(tokenized_datasets['test'])} ejemplos")

# %% [markdown]
# ## 6. Modelo
#
# Cargamos `cardiffnlp/twitter-xlm-roberta-base-sentiment` como checkpoint base.
# Este modelo ya fue pre-entrenado para sentimiento en múltiples idiomas,
# por lo que es un excelente punto de partida para ES/PT.
#
# Configuramos el classification head para 3 labels:
# - `0 → NEGATIVE`
# - `1 → NEUTRAL`
# - `2 → POSITIVE`

# %%
MODEL_CHECKPOINT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
NUM_LABELS = 3

id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

print(f"📦 Modelo base: {MODEL_CHECKPOINT}")
print(f"   Num labels: {NUM_LABELS}")
print(f"   Labels: {id2label}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # El checkpoint original tiene 3 labels pero puede diferir
)

# Mover modelo a GPU
model = model.to(device)
print(f"✅ Modelo cargado en {device}")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parámetros totales: {total_params:,}")
print(f"   Parámetros entrenables: {trainable_params:,}")

# %% [markdown]
# ## 7. TrainingArguments
#
# Hiperparámetros definidos en el diseño:
# - lr=2e-5, batch_size=16, epochs=5
# - weight_decay=0.01, warmup_ratio=0.1, fp16=True
# - Push to Hub con checkpoint cada epoch

# %%
training_args = TrainingArguments(
    # ── Output ──
    output_dir="/kaggle/working/checkpoints/sentiment",
    logging_dir="/kaggle/working/logs/sentiment",

    # ── Hiperparámetros ──
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,

    # ── Precisión mixta ──
    fp16=torch.cuda.is_available(),
    fp16_full_eval=torch.cuda.is_available(),

    # ── Estrategia de evaluación ──
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    # ── Label smoothing (anti-overfitting) ──
    label_smoothing_factor=0.1,

    # ── Logging ──
    logging_steps=50,
    logging_first_step=True,
    report_to="none",  # No usar wandb/tensorboard en Kaggle

    # ── Push to Hub ──
    push_to_hub=False,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="end",

    # ── Seed ──
    seed=42,
    data_seed=42,

    # ── Misc ──
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    remove_unused_columns=False,
)

print("✅ TrainingArguments configurados:")
print(f"   learning_rate: {training_args.learning_rate}")
print(f"   batch_size: {training_args.per_device_train_batch_size}")
print(f"   epochs: {training_args.num_train_epochs}")
print(f"   weight_decay: {training_args.weight_decay}")
print(f"   warmup_ratio: {training_args.warmup_ratio}")
print(f"   fp16: {training_args.fp16}")
print(f"   label_smoothing: {training_args.label_smoothing_factor}")
print(f"   push_to_hub: {training_args.push_to_hub}")
print(f"   hub_model_id: {training_args.hub_model_id}")
print(f"   hub_strategy: {training_args.hub_strategy}")
print(f"   metric_for_best_model: {training_args.metric_for_best_model}")

# %% [markdown]
# ## 8. Métricas de evaluación

# %%
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Calcula accuracy + F1-weighted para evaluación."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "f1_weighted": f1_weighted["f1"],
        "f1_macro": f1_macro["f1"],
    }

print("✅ compute_metrics definida: accuracy + f1_weighted + f1_macro")

# %% [markdown]
# ## 9. Trainer con Early Stopping

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],  # SOLO val — NUNCA test
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
        )
    ],
)

print("✅ Trainer creado con early stopping patience=2")
print("   Train dataset: tokenized_datasets['train']")
print("   Eval dataset:  tokenized_datasets['val']  (NUNCA test durante training)")
print("   Callback: EarlyStopping(patience=2, threshold=0.001)")

# %% [markdown]
# ## 10. Entrenar 🚀

# %%
print("=" * 70)
print("🚀 INICIANDO ENTRENAMIENTO")
print("=" * 70)
print(f"   Train: {len(tokenized_datasets['train'])} ejemplos")
print(f"   Val:   {len(tokenized_datasets['val'])} ejemplos")
print(f"   Épocas máximas: {training_args.num_train_epochs}")
print(f"   Early stopping patience: 2")
print(f"   Checkpoints → HF Hub: {HUB_MODEL_ID}")
print(f"   Label smoothing: 0.1")
print("=" * 70)

train_result = trainer.train()

# ── Guardar métricas de entrenamiento ──
train_metrics = train_result.metrics
print(f"\n📊 Métricas finales de entrenamiento:")
for key, value in train_metrics.items():
    print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")

# Guardar métricas a archivo
metrics_path = "/kaggle/working/checkpoints/sentiment/train_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(train_metrics, f, indent=2)
print(f"   📄 Métricas guardadas en: {metrics_path}")

# %% [markdown]
# ## 11. Evaluar en validation set (post-training)

# %%
print("📊 Evaluando en validation set...")
val_results = trainer.evaluate(eval_dataset=tokenized_datasets["val"])
print(f"\n   Val accuracy:    {val_results['eval_accuracy']:.4f}")
print(f"   Val f1_weighted: {val_results['eval_f1_weighted']:.4f}")
print(f"   Val f1_macro:    {val_results['eval_f1_macro']:.4f}")
print(f"   Val loss:        {val_results['eval_loss']:.4f}")

# %% [markdown]
# ## 12. Evaluar en test set (SOLO al final)

# %%
print("=" * 70)
print("🧪 EVALUACIÓN EN TEST SET")
print("=" * 70)
print("⚠️  NO se usó test set durante training — esta es la primera vez")

test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print(f"\n   Test accuracy:    {test_results['eval_accuracy']:.4f}")
print(f"   Test f1_weighted: {test_results['eval_f1_weighted']:.4f}")
print(f"   Test f1_macro:    {test_results['eval_f1_macro']:.4f}")
print(f"   Test loss:        {test_results['eval_loss']:.4f}")

# ── Classification Report ──
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print("\n📋 Classification Report:")
print(classification_report(
    true_labels,
    preds,
    target_names=[id2label[i] for i in range(NUM_LABELS)],
    digits=4,
))

# ── Matriz de confusión ──
cm = confusion_matrix(true_labels, preds)
print("\n📊 Matriz de Confusión:")
print(f"       {'':>10}", end="")
for i in range(NUM_LABELS):
    print(f"{id2label[i]:>10}", end="")
print()
for i in range(NUM_LABELS):
    print(f"{id2label[i]:>10}", end="")
    for j in range(NUM_LABELS):
        print(f"{cm[i, j]:>10}", end="")
    print()

# ── Verificar targets ──
test_accuracy = test_results["eval_accuracy"]
print(f"\n🎯 Target: accuracy > 0.95")
print(f"   Resultado: {test_accuracy:.4f} — {'✅ CUMPLE' if test_accuracy > 0.95 else '❌ NO CUMPLE'}")

if test_accuracy <= 0.95:
    print("\n⚠️  NO se alcanzó el target. Posibles acciones:")
    print("   1. Revisar calidad de datos (ruido, errores de etiquetado)")
    print("   2. Aumentar épocas (si no hay overfitting)")
    print("   3. Ajustar learning rate (probar 1e-5 o 3e-5)")
    print("   4. Aumentar datos sintéticos para clases minoritarias")
    print("   5. Probar frozen encoder + solo fine-tune del head")

# Guardar métricas de test
test_metrics = {
    "accuracy": float(test_results["eval_accuracy"]),
    "f1_weighted": float(test_results["eval_f1_weighted"]),
    "f1_macro": float(test_results["eval_f1_macro"]),
    "loss": float(test_results["eval_loss"]),
    "confusion_matrix": cm.tolist(),
    "timestamp": datetime.now().isoformat(),
}
with open("/kaggle/working/checkpoints/sentiment/test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"\n📄 Métricas de test guardadas en /kaggle/working/checkpoints/sentiment/test_metrics.json")

# %% [markdown]
# ## 13. Push final a Hugging Face Hub

# %%
print(f"📤 Pusheando modelo final a HF Hub: {HUB_MODEL_ID}...")

# Push del modelo y tokenizer
trainer.push_to_hub()
tokenizer.push_to_hub(HUB_MODEL_ID)

print(f"✅ Modelo pusheado exitosamente a https://huggingface.co/{HUB_MODEL_ID}")
print(f"   Tags de commit disponibles en el Hub")

# ── Guardar cartas de modelo (model card automática) ──
model_card = f"""
---
language:
- es
- pt
pipeline_tag: text-classification
tags:
- sentiment
- roberta
- xlm-roberta
- conversation-ai
- conversaai
- finetuned
datasets:
- amazon_reviews_multi
- corpus_ecommerce
metrics:
- accuracy
- f1
---

# ConversaAI — XLM-R Sentiment (ES/PT)

Fine-tuned from `cardiffnlp/twitter-xlm-roberta-base-sentiment` for 3-class sentiment
classification (negative, neutral, positive) on Spanish and Portuguese customer support messages.

## Metrics

| Split | Accuracy | F1 (weighted) | F1 (macro) |
|-------|----------|---------------|------------|
| Test  | {test_metrics['accuracy']:.4f} | {test_metrics['f1_weighted']:.4f} | {test_metrics['f1_macro']:.4f} |

## Labels

- `0`: NEGATIVE
- `1`: NEUTRAL
- `2`: POSITIVE
"""
print("\n📝 Model card generada")

# %% [markdown]
# ## 14. Prueba de inferencia

# %%
print("🧪 Probando el modelo con ejemplos manuales...")

# Cargar modelo desde Hub (simula el uso en producción)
from transformers import pipeline

sentiment_pipeline = pipeline(
    "text-classification",
    model=HUB_MODEL_ID,
    tokenizer=MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1,
)

test_examples = [
    # ES - Positivo
    "Excelente servicio, muy rápido y eficiente",
    "Me encantó la atención, volveré a comprar sin dudas",
    # ES - Neutro
    "¿Cuál es el plazo de entrega para mi pedido?",
    "Necesito saber el horario de atención al cliente",
    # ES - Negativo
    "Pésimo servicio, nunca más compro aquí",
    "Llevo una semana esperando y nadie me responde",
    # PT - Positivo
    "Atendimento maravilhoso, resolvemos tudo rapidinho",
    "Adorei o produto, chegou antes do prazo",
    # PT - Neutro
    "Gostaria de saber o prazo de entrega do pedido 12345",
    "Qual o valor do frete para Minas Gerais?",
    # PT - Negativo
    "Péssimo atendimento, nunca mais compro aqui",
    "Quero meu dinheiro de volta, isso é um absurdo",
    # Edge case: vacío
    "",
    "   ",
]

print(f"\n{'TEXTO':<70} → {'LABEL':<12} {'PROB':<8}")
print("-" * 95)
for text in test_examples:
    result = sentiment_pipeline(text[:512])[0]  # truncar por si acaso
    label = result["label"]
    score = result["score"]
    display_text = text if text else "(vacío)"
    print(f"{display_text:<70} → {label:<12} {score:.4f}")

# Verificar escenarios del spec
print("\n" + "=" * 70)
print("🔍 VERIFICACIÓN DE ESCENARIOS (SPEC)")
print("=" * 70)

scenarios = [
    {
        "name": "ES Positive",
        "text": "Excelente servicio, muy rápido y eficiente",
        "expected_label": "POSITIVE",
        "min_prob": 0.85,
    },
    {
        "name": "PT Negative",
        "text": "Péssimo atendimento, nunca mais compro aqui",
        "expected_label": "NEGATIVE",
        "min_prob": 0.85,
    },
    {
        "name": "Neutral factual (PT)",
        "text": "Gostaria de saber o prazo de entrega do pedido 12345",
        "expected_label": "NEUTRAL",
        "min_prob": 0.0,  # Solo verificar label, no prob mínima
    },
    {
        "name": "Empty input",
        "text": "",
        "expected_label": "NEUTRAL",  # Edge case spec
        "min_prob": 0.0,
    },
]

all_pass = True
for scenario in scenarios:
    result = sentiment_pipeline(scenario["text"])[0]
    label = result["label"]
    score = result["score"]

    label_ok = label == scenario["expected_label"]
    prob_ok = score >= scenario["min_prob"]

    status = "✅" if label_ok and prob_ok else "❌"
    if not label_ok:
        all_pass = False
    if not label_ok and scenario["name"] == "Empty input":
        # El empty input puede clasificar distinto, no es crítico
        status = "⚠️"

    print(f"   {status} {scenario['name']:<20} → {label:<12} ({score:.4f}) "
          f"[esperado: {scenario['expected_label']}]")

if all_pass:
    print("\n🎉 Todos los escenarios del spec CUMPLEN")
else:
    print("\n⚠️  Algunos escenarios no cumplen — revisar")

# %% [markdown]
# ## 15. Limpieza de caché

# %%
print("🧹 Limpiando caché para liberar espacio en Kaggle...")

cache_dirs = [
    "/root/.cache/huggingface/datasets",
    "/root/.cache/huggingface/metrics",
    "/root/.cache/huggingface/modules",
]

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(cache_dir)
            for filename in filenames
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"   🗑️  Eliminado: {cache_dir} ({size / 1e6:.1f} MB)")

print("✅ Limpieza completada")
print(f"\n🏁 Modelo disponible en: https://huggingface.co/{HUB_MODEL_ID}")

# %% [markdown]
# ## Resumen Final
#
# | Item | Estado |
# |------|--------|
# | Modelo base | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
# | Labels | 3 (NEGATIVE, NEUTRAL, POSITIVE) |
# | Test accuracy | `{test_metrics['accuracy']:.4f}` |
# | Test F1 (weighted) | `{test_metrics['f1_weighted']:.4f}` |
# | HF Hub | [`{HUB_MODEL_ID}`](https://huggingface.co/{HUB_MODEL_ID}) |
# | Anti-overfitting | ✅ Early stopping, weight decay, label smoothing, stratified split |
