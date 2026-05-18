#!/usr/bin/env python3
# ============================================
# CONVERSAI - ANÁLISIS COMPLETO EN CPU
# ============================================
# Este script corre sin GPU y genera métricas,
# visualizaciones y reporte como archivo Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "raw")

print("="*70)
print("💬 CONVERSAI - ANÁLISIS COMPLETO (SIN GPU)")
print("="*70)

# ============================================
# 1. CARGAR DATOS
# ============================================
print("\n📥 Cargando corpus...")
df = pd.read_csv(os.path.join(DATA_PATH, "corpus_ecommerce.csv"))
print(f"   ✅ {len(df):,} mensajes cargados")

# ============================================
# 2. MÉTRICAS BÁSICAS
# ============================================
print("\n" + "="*70)
print("📊 MÉTRICAS BÁSICAS")
print("="*70)

total = len(df)
positivos = (df['sentiment'] == 'positive').sum()
negativos = (df['sentiment'] == 'negative').sum()

print(f"""
   Total mensajes:     {total:,}
   Positivos:          {positivos:,} ({positivos/total*100:.1f}%)
   Negativos:          {negativos:,} ({negativos/total*100:.1f}%)

   Train:              {len(df[df['split']=='train']):,}
   Test:               {len(df[df['split']=='test']):,}
   Val:                {len(df[df['split']=='val']):,}
""")

# ============================================
# 3. DETECTAR FRUSTRACIÓN (KEYWORDS)
# ============================================
print("="*70)
print("😤 DETECCIÓN DE FRUSTRACIÓN (POR KEYWORDS)")
print("="*70)

frustration_keywords = [
    'péssimo', 'terrível', 'absurdo', 'inaceitável', 'pior', 'horrível',
    'nunca', 'jamais', 'desisto', 'reclamação', 'procon',
    'não recebi', 'não chegou', 'devolver', 'dinheiro', 'estorno',
    'problema', 'erro', 'atraso', 'atrasado', 'atrasada',
    'quebrou', 'quebrou', 'defeito', 'falhou',
    'nunca mais', 'never again', 'no recomendo', 'não recomendo'
]

# Solo analizar mensajes negativos
negativos_df = df[df['sentiment'] == 'negative'].copy()

# Contar keywords
keyword_counts = {}
for kw in frustration_keywords:
    count = negativos_df['text'].str.lower().str.contains(kw, na=False).sum()
    if count > 0:
        keyword_counts[kw] = count

# Ordenar por frecuencia
frustrated_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

print("\n   Keywords de frustración más frecuentes (en negativos):")
for kw, count in frustrated_keywords[:15]:
    pct = count / negativos * 100
    print(f"   🔴 {kw}: {count:,} ({pct:.1f}%)")

# Marcar mensajes frustrados
def has_frustration_keyword(text):
    text_lower = str(text).lower()
    for kw in frustration_keywords:
        if kw in text_lower:
            return True
    return False

df['has_frustration'] = df['text'].apply(has_frustration_keyword)
frustrated_total = df['has_frustration'].sum()

print(f"\n   Total mensajes con señales de frustración: {frustrated_total:,} ({frustrated_total/total*100:.1f}%)")

# ============================================
# 4. DETECTAR SEÑALES DE ABANDONO
# ============================================
print("\n" + "="*70)
print("⚠️ DETECCIÓN DE SEÑALES DE ABANDONO")
print("="*70)

abandonment_keywords = [
    'desisto', 'desistir', 'cancelar minha conta', 'cancelo',
    'nunca mais', 'never again', 'não compr', 'não vou comprar',
    'procon', 'reclamação formal', 'advogado', 'processo',
    'péssimo atendimento', 'terrível atendimento', 'sem solução'
]

abandonment_counts = {}
for kw in abandonment_keywords:
    count = df['text'].str.lower().str.contains(kw, na=False).sum()
    if count > 0:
        abandonment_counts[kw] = count

print("\n   Señales de abandono detectadas:")
for kw, count in abandonment_counts.items():
    print(f"   ⚠️  {kw}: {count:,}")

total_abandonment = sum(abandonment_counts.values())
print(f"\n   Total señales de abandono: {total_abandonment:,}")

# ============================================
# 5. DETECTAR INTENCIONES (POR KEYWORDS)
# ============================================
print("\n" + "="*70)
print("🎯 DETECCIÓN DE INTENCIONES (POR KEYWORDS)")
print("="*70)

intents_keywords = {
    'cancel_order': ['cancelar', 'cancele', 'cancelamento', 'cancelar pedido'],
    'delivery_problem': ['entrega', 'entregar', 'entregue', 'entregue', 'frete', 'envio'],
    'refund': ['devolver', 'reembolso', 'dinheiro de volta', 'estorno', 'trocar'],
    'payment_issue': ['pagamento', 'pagar', 'cartão', 'boleto', 'transferência'],
    'complaint': ['reclamação', 'reclama', 'problema', 'quebrou', 'defeito'],
    'delivery_time': ['demora', 'quanto tempo', 'prazo', 'quando chega'],
    'contact_agent': ['atendente', 'falar com', ' falar com', 'supervisor', 'gerente'],
    'check_order': ['pedido', 'rastrear', 'rastreio', 'onde está', 'status'],
    'invoice': ['nota fiscal', 'fatura', 'recibo', 'invoice'],
    'return_product': ['devolução', 'troca', 'trocando', 'defeito']
}

# Detectar intenciones en mensajes negativos (los más relevantes)
intent_counts = {intent: 0 for intent in intents_keywords}

for intent, keywords in intents_keywords.items():
    for kw in keywords:
        count = negativos_df['text'].str.lower().str.contains(kw, na=False).sum()
        intent_counts[intent] += count

# Ordenar por frecuencia
sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)

print("\n   Intenciones más frecuentes en mensajes negativos:")
for intent, count in sorted_intents[:10]:
    pct = count / negativos * 100
    print(f"   • {intent}: {count:,} ({pct:.1f}%)")

# ============================================
# 6. INtenciones con más frustración
# ============================================
print("\n" + "="*70)
print("🎯 INTENCIONES CON MAYOR FRUSTRACIÓN")
print("="*70)

# Cruzar: mensajes negativos + con keywords de frustración + intenciones
frustrated_df = df[df['sentiment'] == 'negative'].copy()
frustrated_df['intencion_detectada'] = 'unknown'

for idx, row in frustrated_df.iterrows():
    text_lower = str(row['text']).lower()
    for intent, keywords in intents_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                frustrated_df.at[idx, 'intencion_detectada'] = intent
                break

# Agrupar por intención
intent_frustration = frustrated_df[frustrated_df['intencion_detectada'] != 'unknown'].groupby('intencion_detectada').size()
intent_frustration = intent_frustration.sort_values(ascending=False)

print("\n   Top intenciones con más mensajes negativos:")
for intent, count in intent_frustration.head(10).items():
    pct = count / negativos * 100
    print(f"   🔴 {intent}: {count:,} mensajes ({pct:.1f}% de negativos)")

# ============================================
# 7. GENERAR REPORTE COMO PYTHON
# ============================================
print("\n" + "="*70)
print("📝 GENERANDO REPORTE")
print("="*70)

reporte_path = os.path.join(BASE_PATH, "reporte_conversaAI.py")

contenido = f'''# ============================================
# CONVERSAI - REPORTE DE ANÁLISIS
# ============================================
# Generado automáticamente por analisis_local.py
# Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}
# Este análisis NO requiere GPU - usa keywords

# Para usar este reporte:
# from reporte_conversaAI import REPORTE
# print(REPORTE["resumen"]["total_mensajes"])

REPORTE = {{
    "fecha_analisis": "{datetime.now().isoformat()}",
    "tipo_analisis": "keyword-based (sin modelo ML)",
    "resumen": {{
        "total_mensajes": {total},
        "mensajes_positivos": {positivos},
        "mensajes_negativos": {negativos},
        "tasa_positivos": "{positivos/total*100:.1f}%",
        "tasa_negativos": "{negativos/total*100:.1f}%",
        "mensajes_con_frustracion": {frustrated_total},
        "tasa_frustracion": "{frustrated_total/total*100:.1f}%",
        "senales_abandono": {total_abandonment}
    }},
    "insights": {{
        "top_keywords_frustracion": {frustrated_keywords[:10]},
        "senales_abandono": {list(abandonment_counts.items())[:5]},
        "intenciones_negativos": {sorted_intents[:10]},
        "intenciones_con_mas_frustracion": {list(intent_frustration.head(5).items())}
    }},
    "recomendaciones": [
'''

# Generar recomendaciones automáticas
recommendations = {
    'complaint': 'Revisar flujo de atención al cliente, implementar respuesta automática con tiempos de resolución claros',
    'cancel_order': 'Simplificar proceso de cancelación, añadir opción de cancelación inmediata',
    'delivery_problem': 'Mejorar comunicación de estado de entrega, notificaciones proactivas',
    'refund': 'Automatizar proceso de reembolso, tiempos claros de devolución',
    'payment_issue': 'Revisar integración con pasarelas de pago, múltiplos métodos',
    'delivery_time': 'Actualizar estimaciones, comunicar retrasos proactivamente',
    'contact_agent': 'Añadir opción de callback, mejorar tiempos de respuesta',
    'check_order': 'Habilitar seguimiento en tiempo real del pedido',
    'invoice': 'Habilitar descarga directa de facturas desde cuenta',
    'return_product': 'Simplificar proceso de devolución, etiquetas prepagadas'
}

# Solo generar recomendaciones para las top 5 intenciones con frustración
for intent, count in intent_frustration.head(5).items():
    accion = recommendations.get(intent, 'Revisar flujo, identificar puntos de fricción')
    prioridad = 'alta' if count > 500 else 'media' if count > 200 else 'baja'
    contenido += f'''        {{
            "prioridad": "{prioridad}",
            "intencion": "{intent}",
            "accion": "{accion}",
            "impacto_estimado": "{count} mensajes negativos"
        }},
'''

contenido += '''    ]
}

# ============================================
# MÉTRICAS ADICIONALES
# ============================================

METRICAS = {
    "distribucion_sentimiento": {
        "positive": ''' + str(positivos) + ''',
        "negative": ''' + str(negativos) + '''
    },
    "por_split": {
        "train": ''' + str(len(df[df['split']=='train'])) + ''',
        "test": ''' + str(len(df[df['split']=='test'])) + ''',
        "val": ''' + str(len(df[df['split']=='val'])) + '''
    },
    "top_frustration_keywords": ''' + str([k for k, v in frustrated_keywords[:10]]) + '''
}

# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("📊 CONVERSAI - REPORTE DE ANÁLISIS")
    print("="*50)

    print(f"\\n📈 Total mensajes: {REPORTE['resumen']['total_mensajes']:,}")
    print(f"😤 Frustración: {REPORTE['resumen']['tasa_frustracion']}")
    print(f"📉 Negativos: {REPORTE['resumen']['tasa_negativos']}")
    print(f"⚠️ Abandono: {REPORTE['resumen']['senales_abandono']:,}")

    print("\\n🎯 TOP RECOMENDACIONES:")
    for i, rec in enumerate(REPORTE['recomendaciones'], 1):
        print(f"\\n{i}. [{rec['prioridad'].upper()}] {rec['intencion']}")
        print(f"   {rec['accion']}")
'''

with open(reporte_path, 'w') as f:
    f.write(contenido)

print(f"   ✅ Reporte guardado: {reporte_path}")

# ============================================
# 8. GENERAR VISUALIZACIONES
# ============================================
print("\n" + "="*70)
print("📊 GENERANDO VISUALIZACIONES")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribución de sentimiento
ax1 = axes[0, 0]
colors = ['#2ecc71', '#e74c3c']
ax1.pie([positivos, negativos], labels=['Positive', 'Negative'], 
        autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Distribución de Sentimiento', fontweight='bold')

# 2. Top keywords de frustración
ax2 = axes[0, 1]
if frustrated_keywords:
    kw_names = [k for k, v in frustrated_keywords[:10]]
    kw_values = [v for k, v in frustrated_keywords[:10]]
    ax2.barh(kw_names, kw_values, color='#e74c3c', alpha=0.7)
    ax2.set_title('Top Keywords de Frustración', fontweight='bold')
    ax2.set_xlabel('Cantidad')

# 3. Intenciones con más frustración
ax3 = axes[1, 0]
if len(intent_frustration) > 0:
    intents_names = list(intent_frustration.head(10).index)
    intents_values = list(intent_frustration.head(10).values)
    ax3.barh(intents_names, intents_values, color='#9b59b6', alpha=0.7)
    ax3.set_title('Intenciones con Más Frustración', fontweight='bold')
    ax3.set_xlabel('Mensajes negativos')

# 4. Métricas clave
ax4 = axes[1, 1]
ax4.axis('off')
metrics_text = f"""
📊 MÉTRICAS CLAVE

Total mensajes: {total:,}

Sentimiento:
  🟢 Positive: {positivos:,} ({positivos/total*100:.1f}%)
  🔴 Negative: {negativos:,} ({negativos/total*100:.1f}%)

Frustración:
  😤 Con keywords: {frustrated_total:,} ({frustrated_total/total*100:.1f}%)

Abandono:
  ⚠️ Señales: {total_abandonment:,}

Intenciones únicas detectadas: {len(intent_frustration)}
"""
ax4.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
         fontfamily='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

plt.tight_layout()

# Guardar
chart_path = os.path.join(BASE_PATH, "dashboard.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"   ✅ Dashboard guardado: {chart_path}")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETO")
print("="*70)

print(f"""
📊 RESUMEN DE MÉTRICAS:

   📁 Dataset: {total:,} mensajes
   📈 Positive: {positivos:,} ({positivos/total*100:.1f}%)
   📉 Negative: {negativos:,} ({negativos/total*100:.1f}%)
   😤 Frustración (keywords): {frustrated_total:,} ({frustrated_total/total*100:.1f}%)
   ⚠️  Abandono: {total_abandonment:,}
   🎯 Intenciones detectadas: {len(intent_frustration)}

📁 Archivos generados:
   • {reporte_path}
   • {chart_path}

🎯 TOP 5 INTENCIONES CON MÁS FRUSTRACIÓN:
""")

for i, (intent, count) in enumerate(intent_frustration.head(5).items(), 1):
    print(f"   {i}. {intent}: {count:,} mensajes")

print("\n" + "="*70)
print("Para ver el reporte en Python:")
print("   python reporte_conversaAI.py")
print("="*70)