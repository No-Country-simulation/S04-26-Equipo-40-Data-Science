# ============================================
# CONVERSAI - REPORTE DE ANÁLISIS
# ============================================
# Generado automáticamente por analisis_local.py
# Fecha: 2026-05-18 16:15
# Este análisis NO requiere GPU - usa keywords
import numpy as np

# Para usar este reporte:
# from reporte_conversaAI import REPORTE
# print(REPORTE["resumen"]["total_mensajes"])

REPORTE = {
    "fecha_analisis": "2026-05-18T16:15:32.569179",
    "tipo_analisis": "keyword-based (sin modelo ML)",
    "resumen": {
        "total_mensajes": 110895,
        "mensajes_positivos": 78289,
        "mensajes_negativos": 32606,
        "tasa_positivos": "70.6%",
        "tasa_negativos": "29.4%",
        "mensajes_con_frustracion": 19201,
        "tasa_frustracion": "17.3%",
        "senales_abandono": 3150
    },
    "insights": {
        "top_keywords_frustracion": [('não recebi', np.int64(3549)), ('não recomendo', np.int64(2326)), ('problema', np.int64(2160)), ('defeito', np.int64(1998)), ('dinheiro', np.int64(1760)), ('nunca', np.int64(1427)), ('péssimo', np.int64(1245)), ('não chegou', np.int64(939)), ('devolver', np.int64(797)), ('pior', np.int64(699))],
        "senales_abandono": [('desisto', np.int64(9)), ('desistir', np.int64(43)), ('cancelo', np.int64(88)), ('nunca mais', np.int64(660)), ('não compr', np.int64(1457))],
        "intenciones_negativos": [('delivery_problem', np.int64(9416)), ('complaint', np.int64(5995)), ('return_product', np.int64(5357)), ('delivery_time', np.int64(3195)), ('refund', np.int64(2612)), ('check_order', np.int64(1697)), ('payment_issue', np.int64(1253)), ('cancel_order', np.int64(1102)), ('invoice', np.int64(704)), ('contact_agent', np.int64(612))],
        "intenciones_con_mas_frustracion": [('return_product', 4584), ('delivery_problem', 2710), ('delivery_time', 2221), ('complaint', 2095), ('check_order', 1288)]
    },
    "recomendaciones": [
        {
            "prioridad": "alta",
            "intencion": "return_product",
            "accion": "Simplificar proceso de devolución, etiquetas prepagadas",
            "impacto_estimado": "4584 mensajes negativos"
        },
        {
            "prioridad": "alta",
            "intencion": "delivery_problem",
            "accion": "Mejorar comunicación de estado de entrega, notificaciones proactivas",
            "impacto_estimado": "2710 mensajes negativos"
        },
        {
            "prioridad": "alta",
            "intencion": "delivery_time",
            "accion": "Actualizar estimaciones, comunicar retrasos proactivamente",
            "impacto_estimado": "2221 mensajes negativos"
        },
        {
            "prioridad": "alta",
            "intencion": "complaint",
            "accion": "Revisar flujo de atención al cliente, implementar respuesta automática con tiempos de resolución claros",
            "impacto_estimado": "2095 mensajes negativos"
        },
        {
            "prioridad": "alta",
            "intencion": "check_order",
            "accion": "Habilitar seguimiento en tiempo real del pedido",
            "impacto_estimado": "1288 mensajes negativos"
        },
    ]
}

# ============================================
# MÉTRICAS ADICIONALES
# ============================================

METRICAS = {
    "distribucion_sentimiento": {
        "positive": 78289,
        "negative": 32606
    },
    "por_split": {
        "train": 88828,
        "test": 11000,
        "val": 11067
    },
    "top_frustration_keywords": ['não recebi', 'não recomendo', 'problema', 'defeito', 'dinheiro', 'nunca', 'péssimo', 'não chegou', 'devolver', 'pior']
}

# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("📊 CONVERSAI - REPORTE DE ANÁLISIS")
    print("="*50)

    print(f"\n📈 Total mensajes: {REPORTE['resumen']['total_mensajes']:,}")
    print(f"😤 Frustración: {REPORTE['resumen']['tasa_frustracion']}")
    print(f"📉 Negativos: {REPORTE['resumen']['tasa_negativos']}")
    print(f"⚠️ Abandono: {REPORTE['resumen']['senales_abandono']:,}")

    print("\n🎯 TOP RECOMENDACIONES:")
    for i, rec in enumerate(REPORTE['recomendaciones'], 1):
        print(f"\n{i}. [{rec['prioridad'].upper()}] {rec['intencion']}")
        print(f"   {rec['accion']}")
