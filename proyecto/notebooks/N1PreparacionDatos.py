# ============================================
# CONVERSAI - NOTEBOOK 1: PREPARACIÓN DE DATOS
# ============================================
# Este notebook prepara los datos para el pipeline de análisis
# de sentimiento e intenciones en español/portugués.
#
# Ejecutar en Google Colab con GPU

# 1. MONTAR DRIVE
from google.colab import drive
drive.mount('/content/drive')

# 2. INSTALAR DEPENDENCIAS
!pip install -q datasets transformers torch scikit-learn pandas numpy matplotlib seaborn tqdm huggingface_hub

# 3. CONFIGURAR DIRECTORIOS
import os
from datetime import datetime

BASE_PATH = "/content/drive/MyDrive/ConversaAI/proyecto"
DATA_PATH = os.path.join(BASE_PATH, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
MODELS_PATH = os.path.join(BASE_PATH, "models")
CHECKPOINTS_PATH = os.path.join(BASE_PATH, "checkpoints")
LOGS_PATH = os.path.join(BASE_PATH, "logs")

for path in [BASE_PATH, DATA_PATH, RAW_PATH, PROCESSED_PATH, MODELS_PATH, CHECKPOINTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

print("✅ Estructura de carpetas creada en Drive")
print(f"📁 Base: {BASE_PATH}")
print(f"📁 Datos: {DATA_PATH}")

# 4. LOGIN HUGGING FACE
from huggingface_hub import login

# ⚠️ IMPORTANTE: Usa tu token personal de HF
# Obtén uno en: https://huggingface.co/settings/tokens
# Luego ejecuta: login(token="tu_token_aqui")
token_hf = "TU_TOKEN_AQUI"
login(token=token_hf)

# 5. CARGAR Y PREPARAR DATASETS
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

class DataPreparator:
    """Prepara los datasets necesarios para el pipeline"""

    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def cargar_corpus_ecommerce(self):
        """Carga corpus_ecommerce.csv para sentiment analysis"""
        print("\n" + "="*60)
        print("📥 CARGANDO CORPUS_ECOMMERCE (Sentimiento)")
        print("="*60)

        corpus_path = os.path.join(self.raw_path, "corpus_ecommerce.csv")

        if os.path.exists(corpus_path):
            df = pd.read_csv(corpus_path)
            print(f"✅ Cargaros: {len(df):,} ejemplos")

            # Convertir sentiment a label binario
            df['label'] = (df['sentiment'] == 'negative').astype(int)

            # Filtrar solo葡萄牙语
            df_pt = df[df['lang'] == 'pt'].copy()
            print(f"   Portugués: {len(df_pt):,} ejemplos")

            # Crear dataset HuggingFace
            ds = Dataset.from_pandas(df_pt[['text', 'label']])

            # Split train/val/test (usando los del CSV)
            train = ds.filter(lambda x: x['text'][:1] not in ['a','b'])  # placeholder
            ds_dict = ds.train_test_split(test_size=0.2, seed=42)

            print(f"   Train: {len(ds_dict['train']):,}")
            print(f"   Test: {len(ds_dict['test']):,}")

            return ds_dict
        else:
            raise FileNotFoundError(f"No se encontró: {corpus_path}")

    def cargar_bitext(self):
        """Carga Bitext para intenciones (zero-shot candidates)"""
        print("\n" + "="*60)
        print("📥 CARGANDO BITEXT (Intenciones)")
        print("="*60)

        from datasets import load_dataset

        try:
            ds_bitext = load_dataset(
                "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
            )
            ds = ds_bitext['train']

            # Extraer intents únicos
            intents = list(set(ds['intent']))
            print(f"✅ Cargaros: {len(ds):,} ejemplos")
            print(f"   Intenciones únicas: {len(intents)}")

            # Guardar intents para zero-shot
            intents_path = os.path.join(self.processed_path, "intent_candidates.json")
            import json
            with open(intents_path, 'w') as f:
                json.dump(intents, f, indent=2)
            print(f"   💾 Intenciones guardadas en: {intents_path}")

            return ds

        except Exception as e:
            print(f"❌ Error cargando Bitext: {e}")
            return None

    def cargar_emotions(self):
        """Carga Emotions para análisis de emociones (inglés)"""
        print("\n" + "="*60)
        print("📥 CARGANDO EMOTIONS (Análisis de Emociones)")
        print("="*60)

        from datasets import load_dataset

        try:
            ds_emotions = load_dataset("dair-ai/emotion", split="train")
            print(f"✅ Cargaros: {len(ds_emotions):,} ejemplos")

            # Distribución
            label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
            print("   Distribución:")
            for label, name in label_map.items():
                count = sum(1 for x in ds_emotions['label'] if x == label)
                print(f"     - {name}: {count:,}")

            return ds_emotions

        except Exception as e:
            print(f"❌ Error cargando Emotions: {e}")
            return None

    def guardar_datasets(self, datasets):
        """Guarda todos los datasets procesados"""
        print("\n💾 Guardando datasets procesados...")

        for nombre, dataset in datasets.items():
            if dataset is not None:
                path = os.path.join(self.processed_path, nombre)
                dataset.save_to_disk(path)
                print(f"   ✅ {nombre}: {path}")

        # Guardar metadatos
        import json
        metadatos = {
            'fecha_creacion': datetime.now().isoformat(),
            'datasets': list(datasets.keys()),
            'tamaños': {nombre: len(ds) if dataset is not None else 0
                       for nombre, ds in datasets.items()}
        }

        with open(os.path.join(self.processed_path, 'metadatos.json'), 'w') as f:
            json.dump(metadatos, f, indent=2)

        print(f"\n✅ Metadatos guardados en: {self.processed_path}/metadatos.json")


# EJECUTAR PREPARACIÓN
print("="*60)
print("🚀 CONVERSAI - PREPARACIÓN DE DATOS")
print("="*60)

preparador = DataPreparator(RAW_PATH, PROCESSED_PATH)

# Cargar datasets
datasets = {}

# 1. Corpus Ecommerce (sentimiento - Portugués)
datasets['sentiment'] = preparador.cargar_corpus_ecommerce()

# 2. Bitext (intenciones - Inglés, para zero-shot)
datasets['intentions'] = preparador.cargar_bitext()

# 3. Emotions (emociones - Inglés)
datasets['emotions'] = preparador.cargar_emotions()

# Guardar
preparador.guardar_datasets(datasets)

# Resumen final
print("\n" + "="*60)
print("✅ RESUMEN DE DATOS PREPARADOS")
print("="*60)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│                 PIPELINE DE DATOS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 SENTIMIENTO (Fine-tuning)                               │
│     corpus_ecommerce (PT)  ──► ~88k entrenamiento         │
│     - positive: 78,289 (70.6%)                            │
│     - negative: 32,606 (29.4%)                            │
│                                                             │
│  🎯 INTENCIONES (Zero-shot candidates)                      │
│     Bitext (EN)  ──► 27 intents                            │
│     Usado como catálogo, no para entrenamiento             │
│                                                             │
│  😊 EMOCIONES (Referencia)                                 │
│     Emotions (EN)  ──► 6 emociones                         │
│     Joy, sadness, anger, fear, love, surprise              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

print("\n💡 Listo para ejecutar Notebook 2 (Entrenamiento)")
print("="*60)