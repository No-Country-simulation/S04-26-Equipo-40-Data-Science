# ============================================
# Script para subir proyecto a Google Drive
# ============================================
# Este script comprime el proyecto y genera
# instrucciones para subir a Drive

import os
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def crear_zip():
    """Crea un zip del proyecto para subir a Drive"""
    output_zip = os.path.join(BASE_DIR, "conversaai_proyecto.zip")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Excluir ciertos directorios
            if '__pycache__' in root or '.git' in root:
                continue

            for file in files:
                if file.endswith('.pyc') or file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, BASE_DIR)
                zipf.write(file_path, arcname)

    print(f"✅ Proyecto comprimido: {output_zip}")
    print(f"   Tamaño: {os.path.getsize(output_zip) / 1024 / 1024:.1f} MB")

    return output_zip

def instrucciones():
    """Imprime instrucciones para subir a Drive"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           INSTRUCCIONES PARA SUBIR A GOOGLE DRIVE                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OPCIÓN 1: Upload directo                                        ║
║  ────────────────────────                                        ║
║  1. Ve a drive.google.com                                        ║
║  2. Crea una carpeta "ConversaAI"                                ║
║  3. Sube la carpeta "proyecto" completa                          ║
║                                                                  ║
║  OPCIÓN 2: Por línea de comandos (gdown)                         ║
║  ───────────────────────────────────────                         ║
║  !pip install gdown                                             ║
║  !gdown --folder https://drive.google.com/...                  ║
║                                                                  ║
║  ESTRUCTURA EN DRIVE:                                            ║
║  ────────────────────────                                        ║
║  MyDrive/                                                        ║
║  └── ConversaAI/                                                 ║
║      └── proyecto/                                              ║
║          ├── data/                                              ║
│          │   ├── raw/                                            ║
│          │   │   ├── corpus_ecommerce.csv                        ║
│          │   │   ├── bitext_train.parquet                        ║
│          │   │   └── emotions_train.parquet                      ║
│          │   └── processed/                                      ║
│          ├── notebooks/                                          ║
│          │   ├── N1PreparacionDatos.py                           ║
│          │   ├── N2Entrenamiento.py                              ║
│          │   └── N3EvaluacionDashboard.py                       ║
│          ├── models/                                            ║
│          ├── checkpoints/                                       ║
│          └── README.md                                          ║
║                                                                  ║
║  EJECUTAR EN COLAB:                                             ║
║  ─────────────────                                              ║
║  1. Abrir Google Colab                                           ║
║  2. Montar Drive: from google.colab import drive                ║
║                    drive.mount('/content/drive')                ║
║  3. Ejecutar notebooks en orden:                                ║
║     • N1 → N2 → N3                                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    print("="*60)
    print("🚀 CONVERSAI - PREPARACIÓN PARA GOOGLE DRIVE")
    print("="*60)

    crear_zip()
    instrucciones()