# ============================================
# CONVERSAI - NOTEBOOK 3: EVALUACIÓN Y DASHBOARD
# ============================================
# Dashboard de insights para el equipo de producto
# Muestra frustración, intenciones no resueltas y patrones de escalation
#
# Ejecutar en Google Colab

from google.colab import drive
drive.mount('/content/drive')

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForZeroShotClassification
from datetime import datetime

# Configurar paths
BASE_PATH = "/content/drive/MyDrive/ConversaAI/proyecto"
DATA_PATH = os.path.join(BASE_PATH, "data")
MODELS_PATH = os.path.join(BASE_PATH, "models")

# Cargar corpus para análisis
CORPUS_PATH = os.path.join(DATA_PATH, "raw", "corpus_ecommerce.csv")


class ConversaAIDashboard:
    """Dashboard de análisis de conversaciones"""

    def __init__(self):
        self.sentiment_model = None
        self.intent_model = None
        self.intent_tokenizer = None
        self.candidate_intents = []
        self.corpus_df = None

    def cargar_modelos(self):
        """Carga los modelos entrenados"""
        print("📥 Cargando modelos...")

        # Modelo de sentimiento
        sentiment_path = os.path.join(MODELS_PATH, "sentiment_model")
        if os.path.exists(sentiment_path):
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=sentiment_path,
                tokenizer=sentiment_path,
                device=-1  # CPU
            )
            print("   ✅ Sentimiento cargado")

        # Modelo de intenciones (zero-shot)
        intent_path = os.path.join(MODELS_PATH, "intent_model")
        config_path = os.path.join(intent_path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.candidate_intents = config.get('intents', [])

            # Cargar clasificador zero-shot
            self.intent_model = AutoModelForZeroShotClassification.from_pretrained(intent_path)
            self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_path)
            print("   ✅ Intenciones cargado")

        # Cargar corpus para análisis
        if os.path.exists(CORPUS_PATH):
            self.corpus_df = pd.read_csv(CORPUS_PATH)
            print(f"   ✅ Corpus cargado: {len(self.corpus_df):,} mensajes")

    def analizar_mensaje(self, texto):
        """Analiza un mensaje individual"""
        if not texto or len(texto) < 5:
            return None

        resultado = {
            'texto': texto[:200],  # Truncar para display
            'timestamp': datetime.now().isoformat()
        }

        # Sentimiento
        if self.sentiment_model:
            try:
                sent = self.sentiment_model(texto[:512])[0]
                resultado['sentimiento'] = sent['label']
                resultado['sentimiento_score'] = sent['score']
            except Exception as e:
                resultado['sentimiento'] = 'ERROR'

        # Intención
        if self.intent_model and self.candidate_intents:
            try:
                inputs = self.intent_tokenizer(
                    texto,
                    candidate_labels=self.candidate_intents[:15],
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = self.intent_model(**inputs)

                probs = torch.softmax(outputs.logits, dim=1)
                top_idx = probs.argmax().item()

                resultado['intencion'] = self.candidate_intents[top_idx]
                resultado['intencion_score'] = probs[0][top_idx].item()

            except Exception as e:
                resultado['intencion'] = 'ERROR'

        return resultado

    def analizar_corpus(self, sample_size=5000):
        """Analiza el corpus completo"""
        if self.corpus_df is None:
            print("❌ No hay corpus cargado")
            return None

        print(f"\n📊 Analizando {sample_size:,} mensajes del corpus...")

        # Sample para análisis rápido
        df_sample = self.corpus_df.sample(min(sample_size, len(self.corpus_df)), random_state=42)

        resultados = []

        for idx, row in df_sample.iterrows():
            result = self.analizar_mensaje(row['text'])
            if result:
                result['original_sentiment'] = row.get('sentiment', 'unknown')
                result['lang'] = row.get('lang', 'unknown')
                resultados.append(result)

            if len(resultados) % 1000 == 0:
                print(f"   Procesados: {len(resultados):,}")

        print(f"   ✅ Analizados: {len(resultados):,} mensajes")

        return pd.DataFrame(resultados)

    def detectar_frustracion(self, resultados_df):
        """Detecta mensajes de frustración"""
        print("\n😤 Detectando frustración...")

        # Keywords de frustración
        frustration_keywords = [
            'péssimo', 'terrível', 'absurdo', 'inaceitável', 'pior',
            'nunca', 'jamais', 'desisto', 'reclamação', 'procon',
            'não recebi', 'não chegou', 'devolver', 'cancelar',
            'problema', 'erro', 'atraso', 'atrasado'
        ]

        def is_frustrated(row):
            text = str(row.get('texto', '')).lower()

            # Si el sentimiento es negative con alta confianza
            if row.get('sentimiento') == 'negative' and row.get('sentimiento_score', 0) > 0.7:
                return True

            # Si contiene keywords de frustración
            for keyword in frustration_keywords:
                if keyword in text:
                    return True

            return False

        resultados_df['is_frustrated'] = resultados_df.apply(is_frustrated, axis=1)

        frustrated_count = resultados_df['is_frustrated'].sum()
        total = len(resultados_df)

        print(f"   📊 Frustración detectada: {frustrated_count:,} ({frustrated_count/total*100:.1f}%)")

        return resultados_df

    def identificar_intenciones_no_resueltas(self, resultados_df):
        """Identifica intenciones con mayor frustración"""
        print("\n🎯 Identificando intenciones no resueltas...")

        # Filtrar solo negativos
        negativos = resultados_df[resultados_df['sentimiento'] == 'negative']

        # Agrupar por intención
        if 'intencion' in resultados_df.columns:
            intent_frustration = negativos.groupby('intencion').agg({
                'is_frustrated': 'sum',
                'sentimiento_score': 'mean'
            }).reset_index()

            intent_frustration = intent_frustration.sort_values('is_frustrated', ascending=False)

            print("   Top 10 intenciones con más frustración:")
            for i, row in intent_frustration.head(10).iterrows():
                print(f"     {row['intencion']}: {row['is_frustrated']} mensajes")

            return intent_frustration

        return None

    def generar_dashboard(self, resultados_df, intent_frustration=None):
        """Genera visualizaciones del dashboard"""
        print("\n📊 Generando dashboard...")

        fig = plt.figure(figsize=(16, 12))

        # 1. Distribución de sentimiento
        ax1 = fig.add_subplot(2, 3, 1)
        if 'sentimiento' in resultados_df.columns:
            sent_counts = resultados_df['sentimiento'].value_counts()
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            ax1.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(sent_counts)], startangle=90)
            ax1.set_title('Distribución de Sentimiento', fontsize=12, fontweight='bold')

        # 2. Histograma de scores de confianza
        ax2 = fig.add_subplot(2, 3, 2)
        if 'sentimiento_score' in resultados_df.columns:
            ax2.hist(resultados_df['sentimiento_score'].dropna(), bins=20,
                    color='#3498db', edgecolor='white', alpha=0.7)
            ax2.set_title('Scores de Confianza', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Fensaje')

        # 3. Intenciones con más frustración
        ax3 = fig.add_subplot(2, 3, 3)
        if intent_frustration is not None and len(intent_frustration) > 0:
            top_intents = intent_frustration.head(10)
            ax3.barh(range(len(top_intents)), top_intents['is_frustrated'].values,
                    color='#e74c3c', alpha=0.7)
            ax3.set_yticks(range(len(top_intents)))
            ax3.set_yticklabels(top_intents['intencion'].values, fontsize=9)
            ax3.set_title('Top Intenciones con Frustración', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Mensajes frustrados')

        # 4. Mapa de calor: sentimiento por idioma
        ax4 = fig.add_subplot(2, 3, 4)
        if 'lang' in resultados_df.columns and 'sentimiento' in resultados_df.columns:
            pivot = resultados_df.pivot_table(
                index='lang',
                columns='sentimiento',
                aggfunc='size',
                fill_value=0
            )
            sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax4,
                       cbar_kws={'label': 'Cantidad'})
            ax4.set_title('Sentimiento por Idioma', fontsize=12, fontweight='bold')

        # 5. Señales de abandono
        ax5 = fig.add_subplot(2, 3, 5)
        abandonment_keywords = ['desisto', 'cancelar', 'nunca mais', 'procon', 'reclamação formal']
        abandonment_count = sum(
            resultados_df['texto'].str.lower().str.contains(kw, na=False).sum()
            for kw in abandonment_keywords
        )

        ax5.bar(['Señales de\nAbandono'], [abandonment_count], color='#c0392b', alpha=0.7)
        ax5.set_title('Señales de Abandono Detectadas', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Cantidad')

        # 6. Métricas clave
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')

        metrics_text = f"""
        📊 MÉTRICAS CLAVE

        Total mensajes analizados: {len(resultados_df):,}

        😤 Frustración detectada: {resultados_df['is_frustrated'].sum():,}
           ({resultados_df['is_frustrated'].mean()*100:.1f}%)

        📉 Sentimiento negativo: {(resultados_df['sentimiento']=='negative').sum():,}
           ({(resultados_df['sentimiento']=='negative').mean()*100:.1f}%)

        🎯 Intenciones únicas: {resultados_df['intencion'].nunique() if 'intencion' in resultados_df.columns else 0}

        ⚠️ Señales de abandono: {abandonment_count}
        """

        ax6.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
                fontfamily='monospace', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

        plt.tight_layout()

        # Guardar
        dashboard_path = os.path.join(BASE_PATH, "dashboard_conversaAI.png")
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n💾 Dashboard guardado: {dashboard_path}")

        return dashboard_path

    def generar_reporte(self, resultados_df, intent_frustration=None):
        """Genera reporte con recomendaciones accionables"""
        print("\n📋 Generando reporte...")

        # Calcular métricas
        total = len(resultados_df)
        frustrados = resultados_df['is_frustrated'].sum()
        negativos = (resultados_df['sentimiento'] == 'negative').sum()

        reporte = {
            'fecha_analisis': datetime.now().isoformat(),
            'resumen': {
                'total_mensajes': total,
                'mensajes_frustrados': int(frustrados),
                'tasa_frustracion': f"{frustrados/total*100:.1f}%",
                'mensajes_negativos': int(negativos),
                'tasa_negativos': f"{negativos/total*100:.1f}%"
            },
            'insights': [],
            'recomendaciones': []
        }

        # Top intenciones no resueltas
        if intent_frustration is not None and len(intent_frustration) > 0:
            top_5_intents = intent_frustration.head(5)
            reporte['insights'].append({
                'tipo': 'intenciones_no_resueltas',
                'descripcion': 'Las 5 intenciones con mayor frustración',
                'data': top_5_intents.to_dict('records')
            })

            # Recomendaciones basadas en intenciones
            for _, row in top_5_intents.iterrows():
                intent = row['intencion']
                count = row['is_frustrated']

                recomendacion = {
                    'prioridad': 'alta' if count > 50 else 'media',
                    'intencion': intent,
                    'accion': self._get_recommendation(intent),
                    'impacto_estimado': f"{count} mensajes frustrados"
                }
                reporte['recomendaciones'].append(recomendacion)

        # Guardar reporte como Python (.py) - más fácil de usar
        reporte_path = os.path.join(BASE_PATH, "reporte_conversaAI.py")

        # Generar contenido como código Python
        contenido = f'''# ============================================
# CONVERSAI - REPORTE DE ANÁLISIS
# ============================================
# Generado automáticamente por N3
# Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}

# Para usar este reporte:
# from reporte_conversaAI import REPORTE
# print(REPORTE["resumen"]["total_mensajes"])

REPORTE = {{
    "fecha_analisis": "{datetime.now().isoformat()}",
    "resumen": {{
        "total_mensajes": {total},
        "mensajes_frustrados": {int(frustrados)},
        "tasa_frustracion": "{frustrados/total*100:.1f}%",
        "mensajes_negativos": {int(negativos)},
        "tasa_negativos": "{negativos/total*100:.1f}%"
    }},
    "recomendaciones": [
'''

        for rec in reporte['recomendaciones']:
            contenido += f'''        {{
            "prioridad": "{rec['prioridad']}",
            "intencion": "{rec['intencion']}",
            "accion": "{rec['accion']}",
            "impacto_estimado": "{rec['impacto_estimado']}"
        }},
'''

        contenido += '''    ]
}
'''

        with open(reporte_path, 'w') as f:
            f.write(contenido)

        print(f"💾 Reporte guardado: {reporte_path}")

        # También guardar versión JSON por si se necesita
        reporte_json_path = os.path.join(BASE_PATH, "reporte_conversaAI.json")
        with open(reporte_json_path, 'w') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        print(f"💾 (También guardado como JSON en {reporte_json_path})")

        return reporte

    def _get_recommendation(self, intent):
        """Obtiene recomendación basada en la intención"""
        recommendations = {
            'complaint': 'Revisar flujo de atención al cliente, implementar respuesta automática con tiempos de resolución claros',
            'cancel_order': 'Simplificar proceso de cancelación, añadir opción de cancelación inmediata',
            'delivery_period': 'Actualizar estimación de tiempos de entrega, enviar notificaciones proactivas',
            'payment_issue': 'Revisar integración con pasarelas de pago, añadir soporte para múltiples métodos',
            'contact_customer_service': 'Añadir opción de callback, mejorar tiempos de respuesta',
            'track_refund': 'Automatizar seguimiento de reembolsos, notificaciones de estado',
            'check_invoice': 'Habilitar descarga directa de facturas desde cuenta',
            'default': 'Revisar flujo de esta intención, identificar puntos de fricción'
        }

        return recommendations.get(intent, recommendations['default'])


# ==================== EJECUTAR ====================

print("="*60)
print("📊 CONVERSAI - DASHBOARD DE INSIGHTS")
print("="*60)

# Import torch para el análisis
import torch

# Inicializar dashboard
dashboard = ConversaAIDashboard()

# Cargar modelos
dashboard.cargar_modelos()

# Analizar corpus
resultados = dashboard.analizar_corpus(sample_size=5000)

if resultados is not None:
    # Detectar frustración
    resultados = dashboard.detectar_frustracion(resultados)

    # Identificar intenciones no resueltas
    intent_frustration = dashboard.identificar_intenciones_no_resueltas(resultados)

    # Generar dashboard
    dashboard_path = dashboard.generar_dashboard(resultados, intent_frustration)

    # Generar reporte
    reporte = dashboard.generar_reporte(resultados, intent_frustration)

    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETO")
    print("="*60)
    print(f"📊 Dashboard: {dashboard_path}")
    print(f"📋 Reporte: {BASE_PATH}/reporte_conversaAI.json")

    # Mostrar resumen
    print("\n📈 RESUMEN:")
    print(f"   Total mensajes analizados: {reporte['resumen']['total_mensajes']:,}")
    print(f"   Tasa de frustración: {reporte['resumen']['tasa_frustracion']}")
    print(f"   Tasa de negativos: {reporte['resumen']['tasa_negativos']}")

    if reporte['recomendaciones']:
        print("\n🎯 TOP RECOMENDACIONES:")
        for i, rec in enumerate(reporte['recomendaciones'][:3], 1):
            print(f"   {i}. [{rec['prioridad'].upper()}] {rec['intencion']}")
            print(f"      {rec['accion']}")

print("\n" + "="*60)