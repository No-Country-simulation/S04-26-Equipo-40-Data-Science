#!/usr/bin/env python3
# ============================================
# CONVERSAI - EVALUACIÓN DE MODELO DE SENTIMIENTO
# ============================================
# Este script entrena un modelo simple en CPU y genera
# métricas de evaluación: Accuracy, F1, Matriz de Confusión
# Para comparación posterior con modelo de Colab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Scikit-learn para ML en CPU
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline

print("="*70)
print("📊 CONVERSAI - EVALUACIÓN DE MODELO DE SENTIMIENTO")
print("="*70)
print("   (Este análisis corre en CPU - sin GPU necesaria)")

# ============================================
# 1. CARGAR DATOS
# ============================================
print("\n📥 Cargando datos...")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "raw")
df = pd.read_csv(os.path.join(DATA_PATH, "corpus_ecommerce.csv"))

print(f"   ✅ {len(df):,} mensajes cargados")

# Preparar datos
X = df['text'].fillna('')
y = (df['sentiment'] == 'negative').astype(int)  # 1=negative, 0=positive

print(f"   📊 Distribución: positive={sum(y==0):,}, negative={sum(y==1):,}")

# ============================================
# 2. SPLIT DATOS
# ============================================
print("\n🔀 Dividiendo datos (train/test/val)...")

# Usar los splits del CSV
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']
val_df = df[df['split'] == 'val']

X_train = train_df['text'].fillna('')
y_train = (train_df['sentiment'] == 'negative').astype(int)

X_test = test_df['text'].fillna('')
y_test = (test_df['sentiment'] == 'negative').astype(int)

X_val = val_df['text'].fillna('')
y_val = (val_df['sentiment'] == 'negative').astype(int)

print(f"   Train: {len(X_train):,}")
print(f"   Test:  {len(X_test):,}")
print(f"   Val:   {len(X_val):,}")

# ============================================
# 3. ENTRENAR MODELO (TF-IDF + Logistic Regression)
# ============================================
print("\n" + "="*70)
print("🚀 ENTRENANDO MODELO")
print("="*70)
print("   (TF-IDF + Logistic Regression - corre en CPU)")

# Pipeline: TF-IDF vectorizer + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# Entrenar
print("\n   Entrenando en train...")
pipeline.fit(X_train, y_train)
print("   ✅ Entrenamiento completado")

# ============================================
# 4. EVALUAR EN TEST SET
# ============================================
print("\n" + "="*70)
print("📊 EVALUACIÓN EN TEST SET")
print("="*70)

# Predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Métricas principales
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    MÉTRICAS DE EVALUACIÓN                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📈 Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)                    ║
║  📊 F1 Score:       {f1:.4f} ({f1*100:.2f}%)                    ║
║  🎯 Precision:      {precision:.4f} ({precision*100:.2f}%)                    ║
║  📣 Recall:         {recall:.4f} ({recall*100:.2f}%)                    ║
║  📉 ROC-AUC:        {roc_auc:.4f} ({roc_auc*100:.2f}%)                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# Classification report detallado
print("\n📋 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=['Positive', 'Negative']))

# ============================================
# 5. MATRIZ DE CONFUSIÓN
# ============================================
print("="*70)
print("🔢 MATRIZ DE CONFUSIÓN")
print("="*70)

cm = confusion_matrix(y_test, y_pred)

print(f"""
                 Predicted
                 Pos    Neg
Actual  Pos   [{cm[0,0]:5d}] [{cm[0,1]:5d}]
        Neg   [{cm[1,0]:5d}] [{cm[1,1]:5d}]
""")

# Interpretación
tn, fp, fn, tp = cm.ravel()
print("   Interpretación:")
print(f"   • True Positives (negativos correctos): {tp:,}")
print(f"   • True Negatives (positivos correctos): {tn:,}")
print(f"   • False Positives (falsos positivos): {fp:,}")
print(f"   • False Negatives (falsos negativos): {fn:,}")

# ============================================
# 6. EVALUAR EN VALIDATION SET
# ============================================
print("\n" + "="*70)
print("📊 EVALUACIÓN EN VALIDATION SET")
print("="*70)

y_pred_val = pipeline.predict(X_val)
y_pred_proba_val = pipeline.predict_proba(X_val)[:, 1]

accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val, average='weighted')
roc_auc_val = roc_auc_score(y_val, y_pred_proba_val)

print(f"   Accuracy: {accuracy_val:.4f} ({accuracy_val*100:.2f}%)")
print(f"   F1 Score: {f1_val:.4f} ({f1_val*100:.2f}%)")
print(f"   ROC-AUC:  {roc_auc_val:.4f} ({roc_auc_val*100:.2f}%)")

# ============================================
# 7. GENERAR VISUALIZACIONES
# ============================================
print("\n" + "="*70)
print("📊 GENERANDO VISUALIZACIONES")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Matriz de confusión
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Positive', 'Negative'],
            yticklabels=['Positive', 'Negative'])
ax1.set_title('Matriz de Confusión (Test Set)', fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# 2. Distribución de predicciones
ax2 = axes[0, 1]
pred_counts = pd.Series(y_pred).map({0: 'Positive', 1: 'Negative'}).value_counts()
colors = ['#2ecc71', '#e74c3c']
ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', colors=colors)
ax2.set_title('Distribución de Predicciones', fontweight='bold')

# 3. ROC Curve
ax3 = axes[1, 0]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax3.fill_between(fpr, tpr, alpha=0.3)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Curva ROC', fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# 4. Métricas comparadas
ax4 = axes[1, 1]
metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC']
metrics_values = [accuracy, f1, precision, recall, roc_auc]
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
ax4.set_ylim(0, 1)
ax4.set_title('Métricas del Modelo', fontweight='bold')
ax4.set_ylabel('Score')

# Agregar valores en las barras
for bar, val in zip(bars, metrics_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Guardar
metrics_path = os.path.join(BASE_PATH, "evaluacion_modelo.png")
plt.savefig(metrics_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"   ✅ Visualización guardada: {metrics_path}")

# ============================================
# 8. GUARDAR REPORTE DE EVALUACIÓN
# ============================================
print("\n" + "="*70)
print("📝 GUARDANDO REPORTE DE EVALUACIÓN")
print("="*70)

reporte_path = os.path.join(BASE_PATH, "reporte_evaluacion.py")

contenido = f'''# ============================================
# CONVERSAI - REPORTE DE EVALUACIÓN DEL MODELO
# ============================================
# Generado automáticamente por evaluacion_modelo.py
# Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}
# Modelo: TF-IDF + Logistic Regression (CPU)

import numpy as np

# ============================================
# MÉTRICAS PRINCIPALES
# ============================================

METRICAS = {{
    "modelo": "TF-IDF + Logistic Regression",
    "tipo": "CPU (scikit-learn)",
    "test_set": {{
        "accuracy": {accuracy:.4f},
        "f1_score": {f1:.4f},
        "precision": {precision:.4f},
        "recall": {recall:.4f},
        "roc_auc": {roc_auc:.4f}
    }},
    "validation_set": {{
        "accuracy": {accuracy_val:.4f},
        "f1_score": {f1_val:.4f},
        "roc_auc": {roc_auc_val:.4f}
    }},
    "confusion_matrix": {{
        "true_negatives": {tn},
        "false_positives": {fp},
        "false_negatives": {fn},
        "true_positives": {tp}
    }},
    "dataset_info": {{
        "train_size": {len(X_train)},
        "test_size": {len(X_test)},
        "val_size": {len(X_val)},
        "total_positive": {int(sum(y==0))},
        "total_negative": {int(sum(y==1))}
    }}
}}

# ============================================
# COMPARACIÓN PARA USO FUTURO
# ============================================
# Para comparar con modelo de Colab:
# 
# from reporte_evaluacion import METRICAS
# print(f"CPU Accuracy: {{METRICAS['test_set']['accuracy']}}")
# print(f"GPU Accuracy: {{modelo_gpu_accuracy}}")
#
# Diferencia = modelo_gpu_accuracy - METRICAS['test_set']['accuracy']

if __name__ == "__main__":
    print("="*50)
    print("📊 EVALUACIÓN MODELO CPU")
    print("="*50)
    print(f"\\nAccuracy: {{METRICAS['test_set']['accuracy']:.4f}}")
    print(f"F1 Score: {{METRICAS['test_set']['f1_score']:.4f}}")
    print(f"ROC-AUC: {{METRICAS['test_set']['roc_auc']:.4f}}")
'''

with open(reporte_path, 'w') as f:
    f.write(contenido)

print(f"   ✅ Reporte guardado: {reporte_path}")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*70)
print("✅ EVALUACIÓN COMPLETA")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              RESUMEN DE EVALUACIÓN                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Modelo: TF-IDF + Logistic Regression (CPU)                ║
║                                                              ║
║  Test Set ({len(X_test):,} mensajes):                                     ║
║    • Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)                       ║
║    • F1 Score:  {f1:.4f} ({f1*100:.2f}%)                       ║
║    • ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)                       ║
║                                                              ║
║  Validation Set ({len(X_val):,} mensajes):                                 ║
║    • Accuracy:  {accuracy_val:.4f} ({accuracy_val*100:.2f}%)                       ║
║    • F1 Score:  {f1_val:.4f} ({f1_val*100:.2f}%)                       ║
║                                                              ║
║  Archivos generados:                                        ║
║    • {metrics_path}                        ║
║    • {reporte_path}                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("📝 Este reporte permite comparar con el modelo de Colab (GPU)")
print("="*70)