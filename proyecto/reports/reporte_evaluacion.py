# ============================================
# CONVERSAI - REPORTE DE EVALUACIÓN DEL MODELO
# ============================================
# Generado automáticamente por evaluacion_modelo.py
# Fecha: 2026-05-18 16:23
# Modelo: TF-IDF + Logistic Regression (CPU)

import numpy as np

# ============================================
# MÉTRICAS PRINCIPALES
# ============================================

METRICAS = {
    "modelo": "TF-IDF + Logistic Regression",
    "tipo": "CPU (scikit-learn)",
    "test_set": {
        "accuracy": 0.9449,
        "f1_score": 0.9457,
        "precision": 0.9483,
        "recall": 0.9449,
        "roc_auc": 0.9848
    },
    "validation_set": {
        "accuracy": 0.9461,
        "f1_score": 0.9468,
        "roc_auc": 0.9868
    },
    "confusion_matrix": {
        "true_negatives": 7320,
        "false_positives": 469,
        "false_negatives": 137,
        "true_positives": 3074
    },
    "dataset_info": {
        "train_size": 88828,
        "test_size": 11000,
        "val_size": 11067,
        "total_positive": 78289,
        "total_negative": 32606
    }
}

# ============================================
# COMPARACIÓN PARA USO FUTURO
# ============================================
# Para comparar con modelo de Colab:
# 
# from reporte_evaluacion import METRICAS
# print(f"CPU Accuracy: {METRICAS['test_set']['accuracy']}")
# print(f"GPU Accuracy: {modelo_gpu_accuracy}")
#
# Diferencia = modelo_gpu_accuracy - METRICAS['test_set']['accuracy']

if __name__ == "__main__":
    print("="*50)
    print("📊 EVALUACIÓN MODELO CPU")
    print("="*50)
    print(f"\nAccuracy: {METRICAS['test_set']['accuracy']:.4f}")
    print(f"F1 Score: {METRICAS['test_set']['f1_score']:.4f}")
    print(f"ROC-AUC: {METRICAS['test_set']['roc_auc']:.4f}")
