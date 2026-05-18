# ============================================
# CONVERSAI - DASHBOARD INTERACTIVO EN STREAMLIT
# ============================================
# Dashboard para el equipo de producto
# Ejecutar: streamlit run dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Configuración de página
st.set_page_config(
    page_title="ConversaAI - Dashboard de Insights",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
MODELS_PATH = os.path.join(BASE_PATH, "models")


@st.cache_data
def cargar_datos():
    """Carga los datos del corpus y resultados"""
    corpus_path = os.path.join(DATA_PATH, "raw", "corpus_ecommerce.csv")
    reporte_path = os.path.join(BASE_PATH, "reporte_conversaAI.py")

    datos = {}

    # Cargar corpus
    if os.path.exists(corpus_path):
        datos['corpus'] = pd.read_csv(corpus_path)
        datos['corpus_size'] = len(datos['corpus'])
    else:
        datos['corpus'] = None
        datos['corpus_size'] = 0

    # Cargar reporte (preferir .py, fallback a .json)
    if os.path.exists(reporte_path):
        try:
            # Importar directamente desde el archivo .py
            import importlib.util
            spec = importlib.util.spec_from_file_location("reporte", reporte_path)
            reporte_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reporte_module)
            datos['reporte'] = reporte_module.REPORTE
        except Exception as e:
            st.warning(f"⚠️ Error cargando .py: {e}")
            # Fallback a JSON
            json_path = os.path.join(BASE_PATH, "reporte_conversaAI.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    datos['reporte'] = json.load(f)
    else:
        # Buscar JSON como fallback
        json_path = os.path.join(BASE_PATH, "reporte_conversaAI.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                datos['reporte'] = json.load(f)
        else:
            datos['reporte'] = None

    return datos


def metricas_principales(datos):
    """Muestra las métricas principales"""
    st.header("📊 Métricas Principales")

    if datos.get('reporte'):
        resumen = datos['reporte']['resumen']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Mensajes",
                f"{resumen['total_mensajes']:,}",
                help="Cantidad de mensajes analizados"
            )

        with col2:
            st.metric(
                "Tasa de Frustración",
                resumen['tasa_frustracion'],
                delta_color="inverse",
                help="Porcentaje de mensajes con frustración detectada"
            )

        with col3:
            st.metric(
                "Mensajes Negativos",
                resumen['mensajes_negativos'],
                delta=f"{resumen['tasa_negativos']}",
                delta_color="inverse"
            )

        with col4:
            st.metric(
                "Intenciones No Resueltas",
                len(datos['reporte'].get('recomendaciones', [])),
                help="Cantidad de intenciones con alta frustración"
            )
    else:
        # Datos de ejemplo si no hay modelo entrenado
        if datos.get('corpus') is not None:
            corpus = datos['corpus']
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Mensajes", f"{len(corpus):,}")

            with col2:
                neg = (corpus['sentiment'] == 'negative').sum()
                st.metric("Mensajes Negativos", f"{neg:,}", delta=f"{neg/len(corpus)*100:.1f}%")

            with col3:
                pos = (corpus['sentiment'] == 'positive').sum()
                st.metric("Mensajes Positivos", f"{pos:,}", delta=f"{pos/len(corpus)*100:.1f}%")


def distribucion_sentimiento(datos):
    """Muestra distribución de sentimiento"""
    st.header("📈 Distribución de Sentimiento")

    if datos.get('corpus') is not None:
        corpus = datos['corpus']

        col1, col2 = st.columns([1, 2])

        with col1:
            sent_counts = corpus['sentiment'].value_counts()

            st.write("### Distribución")
            for sent, count in sent_counts.items():
                pct = count / len(corpus) * 100
                color = "🟢" if sent == "positive" else "🔴"
                st.write(f"{color} {sent.capitalize()}: {count:,} ({pct:.1f}%)")

        with col2:
            # Gráfico de barras
            sent_chart = pd.DataFrame({
                'Sentimiento': sent_counts.index,
                'Cantidad': sent_counts.values
            })
            st.bar_chart(sent_chart.set_index('Sentimiento'))

        # Distribución por split
        st.write("### Por Split")
        split_dist = corpus.groupby(['split', 'sentiment']).size().unstack(fill_value=0)
        st.dataframe(split_dist, use_container_width=True)


def intenciones_frustracion(datos):
    """Muestra intenciones con más frustración"""
    st.header("🎯 Intenciones con Mayor Frustración")

    if datos.get('reporte') and datos['reporte'].get('recomendaciones'):
        recomendaciones = datos['reporte']['recomendaciones']

        for i, rec in enumerate(recomendaciones, 1):
            with st.expander(f"#{i} {rec['intencion'].replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Prioridad:**")
                    priority_color = "🔴" if rec['prioridad'] == 'alta' else "🟡"
                    st.write(f"{priority_color} {rec['prioridad'].upper()}")

                    st.write("**Impacto:**")
                    st.write(f"  {rec['impacto_estimado']}")

                with col2:
                    st.write("**Acción Recomendada:**")
                    st.info(rec['accion'])

        # Visualización de prioridades
        st.write("### 📊 Prioridades de Mejora")
        priorities = pd.DataFrame(recomendaciones)
        if not priorities.empty:
            priority_counts = priorities['prioridad'].value_counts()
            st.write(priority_counts)
    else:
        st.warning("⚠️ Ejecuta el Notebook 3 para obtener análisis de intenciones")
        st.info("El análisis de intenciones requiere que los modelos estén entrenados")


def señales_abandono(datos):
    """Muestra señales de abandono detectadas"""
    st.header("⚠️ Señales de Abandono")

    if datos.get('corpus') is not None:
        corpus = datos['corpus']

        # Keywords de abandono
        abandono_keywords = [
            'desisto', 'cancelar', 'nunca mais', 'procon',
            'reclamação formal', 'péssimo', 'terrível',
            'absurdo', 'inaceitável', 'pior', 'nunca',
            'jamais', 'devolver', 'dinheiro de volta'
        ]

        st.write("### Keywords de Abandono Detectadas")

        # Buscar en mensajes negativos
        negativos = corpus[corpus['sentiment'] == 'negative']

        keyword_counts = {}
        for kw in abandono_keywords:
            count = negativos['text'].str.lower().str.contains(kw, na=False).sum()
            if count > 0:
                keyword_counts[kw] = count

        if keyword_counts:
            df_abandono = pd.DataFrame(
                list(keyword_counts.items()),
                columns=['Keyword', 'Cantidad']
            ).sort_values('Cantidad', ascending=False)

            st.dataframe(
                df_abandono,
                use_container_width=True,
                hide_index=True
            )

            # Gráfico
            st.bar_chart(df_abandono.set_index('Keyword'))
        else:
            st.success("✅ No se detectaron señales claras de abandono")
    else:
        st.warning("⚠️ No hay datos disponibles")


def explorar_mensajes(datos):
    """Permite explorar mensajes manualmente"""
    st.header("🔍 Explorador de Mensajes")

    if datos.get('corpus') is not None:
        corpus = datos['corpus']

        # Filtros
        col1, col2 = st.columns(2)

        with col1:
            filtro_sentimiento = st.selectbox(
                "Filtrar por Sentimiento",
                ["Todos", "positive", "negative"]
            )

        with col2:
            filtro_split = st.selectbox(
                "Filtrar por Split",
                ["Todos", "train", "test", "val"]
            )

        # Aplicar filtros
        df_filtrado = corpus.copy()

        if filtro_sentimiento != "Todos":
            df_filtrado = df_filtrado[df_filtrado['sentiment'] == filtro_sentimiento]

        if filtro_split != "Todos":
            df_filtrado = df_filtrado[df_filtrado['split'] == filtro_split]

        st.write(f"### {len(df_filtrado):,} mensajes encontrados")

        # Mostrar mensajes
        st.dataframe(
            df_filtrado[['text', 'sentiment', 'split']].head(50),
            use_container_width=True,
            height=400
        )

        # Descargar
        csv = df_filtrado.to_csv(index=False)
        st.download_button(
            "📥 Descargar CSV filtrado",
            csv,
            "conversaai_mensajes_filtrados.csv",
            "text/csv"
        )
    else:
        st.warning("⚠️ No hay datos disponibles")


def recomendaciones_accionables(datos):
    """Muestra recomendaciones accionables para el equipo"""
    st.header("💡 Recomendaciones para el Equipo de Producto")

    st.markdown("""
    ### Flujo de Trabajo Recomendado:
    
    1. **Revisar métricas principales** - Entender la situación actual
    2. **Identificar intenciones no resueltas** - Priorizar las de mayor frustración
    3. **Analizar señales de abandono** - Detectar puntos de fuga
    4. **Implementar mejoras** - Asignar a sprint
    5. **Medir impacto** - Re-entrenar modelo monthly
    """)

    if datos.get('reporte') and datos['reporte'].get('recomendaciones'):
        st.write("### Acciones Inmediatas")

        recomendaciones = datos['reporte']['recomendaciones']

        for i, rec in enumerate(recomendaciones, 1):
            priority_emoji = "🔴" if rec['prioridad'] == 'alta' else "🟡"

            st.write(f"""
            **{priority_emoji} PRIORIDAD {i}:** {rec['intencion'].replace('_', ' ').title()}
            - {rec['accion']}
            - Impacto: {rec['impacto_estimado']}
            """)

    st.write("### Próximos Pasos")

    st.info("""
    1. Ejecutar Pipeline mensual con nuevos datos
    2. Re-entrenar modelos con datos actualizados
    3. Comparar métricas mes a mes
    4. Presentar resultados en sprint planning
    """)


def sidebar():
    """Configura el sidebar"""
    st.sidebar.title("💬 ConversaAI")

    st.sidebar.write("### Navegación")
    page = st.sidebar.radio(
        "Ir a:",
        ["📊 Resumen", "📈 Sentimiento", "🎯 Intenciones", "⚠️ Abandono", "🔍 Explorar", "💡 Acciones"]
    )

    st.sidebar.divider()

    st.sidebar.write("### Información")
    st.sidebar.info("""
    **Dashboard de Análisis**
    - Equipo de Producto
    - Data Analyst

    Actualizado: {}
    """.format(datetime.now().strftime("%Y-%m-%d")))

    st.sidebar.write("### Links")
    st.sidebar.markdown("""
    - [GitHub]() 
    - [Documentación]()
    """)

    return page


def main():
    """Función principal"""
    # Cargar datos
    datos = cargar_datos()

    # Sidebar
    page = sidebar()

    # Contenido según navegación
    if page == "📊 Resumen":
        st.title("💬 ConversaAI - Dashboard de Insights")
        st.markdown("### Sistema de análisis de conversaciones de soporte")
        metricas_principales(datos)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            distribucion_sentimiento(datos)

        with col2:
            recomendaciones_accionables(datos)

    elif page == "📈 Sentimiento":
        st.title("📈 Análisis de Sentimiento")
        distribucion_sentimiento(datos)

    elif page == "🎯 Intenciones":
        st.title("🎯 Intenciones con Mayor Frustración")
        intenciones_frustracion(datos)

    elif page == "⚠️ Abandono":
        st.title("⚠️ Señales de Abandono")
        señales_abandono(datos)

    elif page == "🔍 Explorar":
        st.title("🔍 Explorador de Mensajes")
        explorar_mensajes(datos)

    elif page == "💡 Acciones":
        st.title("💡 Recomendaciones")
        recomendaciones_accionables(datos)


if __name__ == "__main__":
    main()