# %% [markdown]
# # ConversaAI — Fine-tuning XLM-R Intención (9 clases)
#
# ## Objetivo
# Fine-tunear `xlm-roberta-base` para clasificar intención en **9 clases** para textos
# de soporte al cliente en ES y PT.
#
# ## Target
# - **Macro F1 > 0.85** en test set (ES + PT combinados)
#
# ## Clases
# | # | Intención | Descripción |
# |---|-----------|-------------|
# | 0 | `cancelacion` | Solicitud de cancelación |
# | 1 | `consulta_general` | Consulta informativa general |
# | 2 | `facturacion_pago` | Problemas de facturación o pago |
# | 3 | `feedback` | Opinión o retroalimentación |
# | 4 | `gestion_cuenta` | Cambios en cuenta/datos personales |
# | 5 | `modificacion_pedido` | Cambios en pedido existente |
# | 6 | `queja` | Reclamo o queja formal |
# | 7 | `reembolso` | Solicitud de reembolso |
# | 8 | `seguimiento` | Consulta de estado/seguimiento |
#
# ## Entorno
# - **Plataforma**: Kaggle (GPU T4x2, 16 GB VRAM)
# - **Datos**: Parquets subidos como Kaggle Dataset
# - **Checkpoints**: Pusheados a HF Hub (`Rosela/xlm-r-intent-espt`)
# - **Tiempo estimado**: ~3-4 horas (5 épocas, early stopping patience=2)
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
os.environ["HF_HOME"] = "/kaggle/working/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/working/.cache/datasets"
os.environ["TORCH_HOME"] = "/kaggle/working/.cache/torch"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/.cache/huggingface"

# Crear estructura de directorios
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

# ── Seed ──
set_seed(42)

# ── Login HF ──
HF_TOKEN = "TU_TOKEN_HF_AQUI"
login(token=HF_TOKEN)
user_info = whoami()
print(f"✅ Login exitoso — {user_info['name']}")

# ── Repo en HF Hub ──
HUB_MODEL_ID = "Rosela/xlm-r-intent-espt"
print(f"📦 Hub model ID: {HUB_MODEL_ID}")

# %% [markdown]
# ## 4. Cargar datos
#
# Los datos deben estar en formato parquet en `data/processed/intent/` con splits:
# - `train.parquet`
# - `val.parquet`
# - `test.parquet`
#
# En Kaggle, se suben como dataset de Kaggle. Ajustar `DATA_PATH` según corresponda.

# %%
# ── Configurar ruta de datos ──
DATA_PATH = "/kaggle/input/conversaai-intent-data-limpio"

if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/processed/intent_limpio"
    print(f"[!] Ruta Kaggle no encontrada, usando: {DATA_PATH}")

print(f"[+] Data path: {DATA_PATH}")

# ── Definir mapping de intenciones ──
INTENT_LABELS = [
    "cancelacion",
    "consulta_general",
    "facturacion_pago",
    "feedback",
    "gestion_cuenta",
    "modificacion_pedido",
    "queja",
    "reembolso",
    "seguimiento",
]

id2label = {i: label for i, label in enumerate(INTENT_LABELS)}
label2id = {label: i for i, label in enumerate(INTENT_LABELS)}

NUM_LABELS = len(INTENT_LABELS)
print(f"🎯 {NUM_LABELS} clases de intención:")
for i, label in enumerate(INTENT_LABELS):
    print(f"   {i}: {label}")

# ── Cargar parquets ──
def load_parquet_safe(path, name):
    full_path = os.path.join(path, name)
    if not os.path.exists(full_path):
        print(f"   [!] {name} no encontrado, se intenta cargar dataset completo...")
        return None
    df = pd.read_parquet(full_path)
    print(f"   ✅ {name}: {len(df)} ejemplos")
    return df

df_train = load_parquet_safe(DATA_PATH, "train.parquet")
df_val = load_parquet_safe(DATA_PATH, "val.parquet")
df_test = load_parquet_safe(DATA_PATH, "test.parquet")

# ── Si no hay splits, cargar completo y dividir ──
if df_train is None:
    print("\n📦 No se encontraron splits. Cargando dataset completo y dividiendo...")

    full_path = os.path.join(DATA_PATH, "intent_full.parquet")
    if not os.path.exists(full_path):
        parquet_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(
                f"No se encontraron archivos parquet en {DATA_PATH}. "
                "Subir los datos como Kaggle Dataset o ajustar DATA_PATH."
            )
        full_path = os.path.join(DATA_PATH, parquet_files[0])
        print(f"   Usando: {parquet_files[0]}")

    df_all = pd.read_parquet(full_path)
    print(f"   Total ejemplos: {len(df_all)}")

    # Verificar columnas
    expected_cols = {"text", "label"}
    if not expected_cols.issubset(df_all.columns):
        raise ValueError(
            f"Columnas esperadas: {expected_cols}, encontradas: {set(df_all.columns)}"
        )

    # Verificar que las labels están en rango
    invalid_labels = df_all[~df_all["label"].isin(range(NUM_LABELS))]
    if len(invalid_labels) > 0:
        print(f"   ⚠️  {len(invalid_labels)} ejemplos con labels inválidas — serán filtrados")
        df_all = df_all[df_all["label"].isin(range(NUM_LABELS))]

    # Stratified split 80/10/10
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

    print(f"\n   Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

# ── Mostrar distribución de clases ──
print("\n📊 Distribución de clases (train):")
train_dist = df_train["label"].value_counts().sort_index()
for label_id, count in train_dist.items():
    pct = count / len(df_train) * 100
    print(f"   {label_id:>2} ({id2label[label_id]:<20}): {count:>6} ({pct:5.1f}%)")

print("\n📊 Distribución de clases (val):")
val_dist = df_val["label"].value_counts().sort_index()
for label_id, count in val_dist.items():
    pct = count / len(df_val) * 100
    print(f"   {label_id:>2} ({id2label[label_id]:<20}): {count:>6} ({pct:5.1f}%)")

print("\n📊 Distribución de clases (test):")
test_dist = df_test["label"].value_counts().sort_index()
for label_id, count in test_dist.items():
    pct = count / len(df_test) * 100
    print(f"   {label_id:>2} ({id2label[label_id]:<20}): {count:>6} ({pct:5.1f}%)")

# ── Verificar desbalance ──
print(f"\n📈 Total: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
min_class = train_dist.min()
max_class = train_dist.max()
print(f"   Clase minoritaria: {train_dist.idxmin()} ({id2label[train_dist.idxmin()]}) = {min_class}")
print(f"   Clase mayoritaria: {train_dist.idxmax()} ({id2label[train_dist.idxmax()]}) = {max_class}")
print(f"   Ratio desbalance: {max_class / min_class:.1f}x")

if max_class / min_class > 5:
    print("   ⚠️  Desbalance significativo — considerar weighted loss o oversampling")

# ── Crear DatasetDict ──
dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train[["text", "label"]]),
    "val": Dataset.from_pandas(df_val[["text", "label"]]),
    "test": Dataset.from_pandas(df_test[["text", "label"]]),
})
print("\n✅ DatasetDict creado")

# %% [markdown]
# ## 5. Tokenizer

# %%
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# XLM-R no tiene pad_token por defecto
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"🔤 Tokenizer: {MODEL_NAME}")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"   Max length: {MAX_LENGTH}")

# ── Tokenización ──
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

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
# Cargamos `xlm-roberta-base` con classification head de 9 labels para intención.
# A diferencia del modelo de sentimiento, NO usamos un checkpoint fine-tuneado previo,
# partimos del XLM-R base pre-entrenado multilingüe.

# %%
MODEL_CHECKPOINT = "xlm-roberta-base"

print(f"📦 Modelo base: {MODEL_CHECKPOINT}")
print(f"   Num labels: {NUM_LABELS}")
print(f"   Labels: {id2label}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

model = model.to(device)
print(f"✅ Modelo cargado en {device}")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parámetros totales: {total_params:,}")
print(f"   Parámetros entrenables: {trainable_params:,}")

# %% [markdown]
# ## 7. TrainingArguments
#
# Hiperparámetros (igual que sentimiento excepto métrica principal):
# - lr=2e-5, batch_size=16, epochs=5
# - weight_decay=0.01, warmup_ratio=0.1, fp16=True
# - **metric_for_best_model = "f1"** (macro F1 > 0.85)
# - Push to Hub con checkpoint cada epoch

# %%
training_args = TrainingArguments(
    # ── Output ──
    output_dir="/kaggle/working/checkpoints/intent",
    logging_dir="/kaggle/working/logs/intent",

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
    metric_for_best_model="f1_macro",  # <-- DIFERENTE: usamos macro F1
    greater_is_better=True,

    # ── Label smoothing (anti-overfitting) ──
    label_smoothing_factor=0.1,

    # ── Logging ──
    logging_steps=50,
    logging_first_step=True,
    report_to="none",

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
    """
    Calcula métricas para clasificación de 9 clases.
    La métrica principal es macro F1 (target > 0.85).
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")
    f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    f1_per_class = f1_metric.compute(
        predictions=preds, references=labels, average=None
    )

    metrics = {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"],
    }

    # F1 per class para diagnóstico
    for i, label in enumerate(INTENT_LABELS):
        metrics[f"f1_{label}"] = f1_per_class["f1"][i]

    return metrics

print("✅ compute_metrics definida: accuracy + f1_macro (principal) + f1_weighted + f1_per_class")

# %% [markdown]
# ## 9. Trainer con Early Stopping

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],  # SOLO val
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
print("   Best model by: f1_macro")
print("   Callback: EarlyStopping(patience=2, threshold=0.001)")

# %% [markdown]
# ## 10. Entrenar 🚀

# %%
print("=" * 70)
print("🚀 INICIANDO ENTRENAMIENTO")
print("=" * 70)
print(f"   Train: {len(tokenized_datasets['train'])} ejemplos")
print(f"   Val:   {len(tokenized_datasets['val'])} ejemplos")
print(f"   Test:  {len(tokenized_datasets['test'])} ejemplos (reservado)")
print(f"   Épocas máximas: {training_args.num_train_epochs}")
print(f"   Early stopping patience: 2")
print(f"   Checkpoints → HF Hub: {HUB_MODEL_ID}")
print(f"   Label smoothing: 0.1")
print(f"   Target: macro F1 > 0.85")
print("=" * 70)

train_result = trainer.train()

# ── Guardar métricas ──
train_metrics = train_result.metrics
print(f"\n📊 Métricas finales de entrenamiento:")
for key, value in train_metrics.items():
    print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")

metrics_path = "/kaggle/working/checkpoints/intent/train_metrics.json"
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(train_metrics, f, indent=2)
print(f"   📄 Métricas guardadas en: {metrics_path}")

# %% [markdown]
# ## 11. Evaluar en validation set (post-training)

# %%
print("📊 Evaluando en validation set...")
val_results = trainer.evaluate(eval_dataset=tokenized_datasets["val"])
print(f"\n   Val accuracy:    {val_results['eval_accuracy']:.4f}")
print(f"   Val f1_macro:    {val_results['eval_f1_macro']:.4f}")
print(f"   Val f1_weighted: {val_results['eval_f1_weighted']:.4f}")
print(f"   Val loss:        {val_results['eval_loss']:.4f}")

# Mostrar F1 por clase
print("\n📊 F1 por clase (val):")
for label in INTENT_LABELS:
    key = f"eval_f1_{label}"
    if key in val_results:
        print(f"   {label:<20}: {val_results[key]:.4f}")

# %% [markdown]
# ## 12. Evaluar en test set (SOLO al final)

# %%
print("=" * 70)
print("🧪 EVALUACIÓN EN TEST SET")
print("=" * 70)
print("⚠️  NO se usó test set durante training — esta es la primera vez")

test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print(f"\n   Test accuracy:    {test_results['eval_accuracy']:.4f}")
print(f"   Test f1_macro:    {test_results['eval_f1_macro']:.4f}")
print(f"   Test f1_weighted: {test_results['eval_f1_weighted']:.4f}")
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
header = [id2label[i][:12] for i in range(NUM_LABELS)]
print(f"       {'':>14}", end="")
for h in header:
    print(f"{h:>12}", end="")
print()
for i in range(NUM_LABELS):
    print(f"{id2label[i]:>14}", end="")
    for j in range(NUM_LABELS):
        print(f"{cm[i, j]:>12}", end="")
    print()

# ── Verificar target ──
test_f1_macro = test_results["eval_f1_macro"]
print(f"\n🎯 Target: macro F1 > 0.85")
print(f"   Resultado: {test_f1_macro:.4f} — {'✅ CUMPLE' if test_f1_macro > 0.85 else '❌ NO CUMPLE'}")

if test_f1_macro <= 0.85:
    print("\n⚠️  NO se alcanzó el target. Posibles acciones:")
    print("   1. Revisar calidad de datos (ruido, errores de etiquetado)")
    print("   2. Aumentar número de épocas (probar 6-8 con early stopping)")
    print("   3. Ajustar learning rate (probar 1e-5 o 3e-5)")
    print("   4. Balancear clases con oversampling de clases minoritarias")
    print("   5. Usar class weights en la loss function")
    print("   6. Probar frozen encoder (freeze primeros N layers) + solo head")

# ── Análisis de errores ──
print("\n🔍 Análisis de errores por clase:")
class_f1 = {}
for label in INTENT_LABELS:
    key = f"test_f1_{label}"
    if key in test_results:
        class_f1[label] = test_results[key]

if class_f1:
    worst_class = min(class_f1, key=class_f1.get)
    best_class = max(class_f1, key=class_f1.get)
    print(f"   Mejor clase:  {best_class:<20} → F1 = {class_f1[best_class]:.4f}")
    print(f"   Peor clase:   {worst_class:<20} → F1 = {class_f1[worst_class]:.4f}")

# Guardar métricas de test
test_metrics = {
    "accuracy": float(test_results["eval_accuracy"]),
    "f1_macro": float(test_results["eval_f1_macro"]),
    "f1_weighted": float(test_results["eval_f1_weighted"]),
    "loss": float(test_results["eval_loss"]),
    "confusion_matrix": cm.tolist(),
    "f1_per_class": {label: float(test_results.get(f"test_f1_{label}", 0))
                     for label in INTENT_LABELS},
    "timestamp": datetime.now().isoformat(),
}
with open("/kaggle/working/checkpoints/intent/test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"\n📄 Métricas de test guardadas en /kaggle/working/checkpoints/intent/test_metrics.json")

# %% [markdown]
# ## 13. Push final a Hugging Face Hub

# %%
print(f"📤 Pusheando modelo final a HF Hub: {HUB_MODEL_ID}...")

trainer.push_to_hub()
tokenizer.push_to_hub(HUB_MODEL_ID)

print(f"✅ Modelo pusheado exitosamente a https://huggingface.co/{HUB_MODEL_ID}")

# ── Model card ──
model_card = f"""
---
language:
- es
- pt
pipeline_tag: text-classification
tags:
- intent
- roberta
- xlm-roberta
- conversation-ai
- conversaai
- finetuned
metrics:
- f1
- accuracy
---

# ConversaAI — XLM-R Intent (ES/PT)

Fine-tuned from `xlm-roberta-base` for 9-class intent classification on Spanish and
Portuguese customer support messages.

## Labels

| ID | Intent | Description |
|----|--------|-------------|
| 0 | cancelacion | Solicitud de cancelación |
| 1 | consulta_general | Consulta informativa general |
| 2 | facturacion_pago | Problemas de facturación o pago |
| 3 | feedback | Opinión o retroalimentación |
| 4 | gestion_cuenta | Cambios en cuenta/datos personales |
| 5 | modificacion_pedido | Cambios en pedido existente |
| 6 | queja | Reclamo o queja formal |
| 7 | reembolso | Solicitud de reembolso |
| 8 | seguimiento | Consulta de estado/seguimiento |

## Metrics

| Split | Accuracy | F1 (macro) | F1 (weighted) |
|-------|----------|------------|---------------|
| Test  | {test_metrics['accuracy']:.4f} | {test_metrics['f1_macro']:.4f} | {test_metrics['f1_weighted']:.4f} |
"""
print("\n📝 Model card generada")

# %% [markdown]
# ## 14. Prueba de inferencia

# %%
print("🧪 Probando el modelo con ejemplos manuales...")

from transformers import pipeline

intent_pipeline = pipeline(
    "text-classification",
    model=HUB_MODEL_ID,
    tokenizer=MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1,
)

test_examples = [
    # Cancelación
    ("Quiero cancelar mi suscripción premium ya mismo", "cancelacion"),
    ("Quero cancelar meu pedido, por favor", "cancelacion"),
    # Consulta general
    ("¿Cuál es el horario de atención al cliente?", "consulta_general"),
    ("Qual o telefone do suporte?", "consulta_general"),
    # Facturación
    ("Me cobraron dos veces el mismo mes", "facturacion_pago"),
    ("Não reconheço esta cobrança no meu cartão", "facturacion_pago"),
    # Feedback
    ("Excelente servicio, muy recomendable", "feedback"),
    ("Adorei o atendimento, nota 10", "feedback"),
    # Gestión de cuenta
    ("Quiero cambiar mi dirección de correo", "gestion_cuenta"),
    ("Preciso atualizar meus dados cadastrais", "gestion_cuenta"),
    # Modificación de pedido
    ("Puedo cambiar la talla de mi pedido?", "modificacion_pedido"),
    ("Quero alterar o endereço de entrega", "modificacion_pedido"),
    # Queja
    ("Esto es inaceptable, pésimo servicio", "queja"),
    ("Estou muito insatisfeito com o produto", "queja"),
    # Reembolso
    ("Quiero que me devuelvan mi dinero", "reembolso"),
    ("Quero meu dinheiro de volta, fiz o pagamento errado", "reembolso"),
    # Seguimiento
    ("Cómo va mi pedido? Ya debería haber llegado", "seguimiento"),
    ("Qual o status do meu pedido?", "seguimiento"),
    # Edge case: sin sentido
    ("asdfgh qwerty 12345", None),
]

print(f"\n{'TEXTO':<65} → {'PRED':<20} {'PROB':<8} {'EXP':<20} {'ESTADO'}")
print("-" * 130)
passed = 0
failed = 0
for text, expected in test_examples:
    result = intent_pipeline(text[:512])[0]
    label = result["label"]
    score = result["score"]
    display_text = text[:60] + "..." if len(text) > 60 else text

    if expected is None:
        # Edge case: solo verificar que no falle y que devuelva una label válida
        status = "⚠️" if label not in INTENT_LABELS else "✅"
        expected_str = "(válida)"
    else:
        status = "✅" if label == expected else "❌"
        if label == expected:
            passed += 1
        else:
            failed += 1
        expected_str = expected

    print(f"{display_text:<65} → {label:<20} {score:.4f}  {expected_str:<20} {status}")

print(f"\n📊 Resultados: {passed} pasaron, {failed} fallaron")

# Verificar escenarios del spec
print("\n" + "=" * 70)
print("🔍 VERIFICACIÓN DE ESCENARIOS (SPEC)")
print("=" * 70)

scenarios = [
    {
        "name": "Cancelación ES",
        "text": "Quiero cancelar mi suscripción premium ya mismo",
        "expected": "cancelacion",
        "min_prob": 0.7,
    },
    {
        "name": "Reembolso PT",
        "text": "Quero meu dinheiro de volta, fiz o pagamento errado",
        "expected": "reembolso",
        "min_prob": 0.0,  # Solo verificar label
    },
    {
        "name": "Low-coherence edge",
        "text": "asdfgh qwerty 12345",
        "expected": None,  # No debe fallar, prob >= 0.2
        "min_prob": 0.2,
    },
]

all_pass = True
for scenario in scenarios:
    result = intent_pipeline(scenario["text"])[0]
    label = result["label"]
    score = result["score"]

    if scenario["expected"] is not None:
        label_ok = label == scenario["expected"]
        prob_ok = score >= scenario["min_prob"]
        status = "✅" if label_ok and prob_ok else "❌"
        if not label_ok:
            all_pass = False
        print(f"   {status} {scenario['name']:<25} → {label:<20} ({score:.4f}) "
              f"[esperado: {scenario['expected']}, min_prob: {scenario['min_prob']}]")
    else:
        prob_ok = score >= scenario["min_prob"]
        prob_in_range = 0.2 <= score <= 1.0
        status = "✅" if prob_in_range else "❌"
        if not prob_in_range:
            all_pass = False
        print(f"   {status} {scenario['name']:<25} → {label:<20} ({score:.4f}) "
              f"[debe tener prob >= 0.2]")

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
# | Modelo base | `xlm-roberta-base` |
# | Labels | 9 (taxonomía completa) |
# | Test macro F1 | `{test_metrics['f1_macro']:.4f}` |
# | Test accuracy | `{test_metrics['accuracy']:.4f}` |
# | HF Hub | [`{HUB_MODEL_ID}`](https://huggingface.co/{HUB_MODEL_ID}) |
# | Anti-overfitting | ✅ Early stopping, weight decay, label smoothing, stratified split |
