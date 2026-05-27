import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.pipeline_fallback import FallbackPipeline

st.set_page_config(
    page_title="ConversaAI — Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(BASE_PATH, "data", "corpus_demo_bilingue.csv")

INTENT_LABELS = [
    "cancelacion", "consulta_general", "facturacion_pago", "feedback",
    "gestion_cuenta", "modificacion_pedido", "queja", "reembolso", "seguimiento",
]
INTENT_DISPLAY = {
    "cancelacion": "Cancelación", "consulta_general": "Consulta General",
    "facturacion_pago": "Facturación/Pago", "feedback": "Feedback",
    "gestion_cuenta": "Gestión de Cuenta", "modificacion_pedido": "Modificación Pedido",
    "queja": "Queja", "reembolso": "Reembolso", "seguimiento": "Seguimiento",
}
SENTIMENT_COLORS = {"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#2ecc71"}

@st.cache_resource
def get_pipeline():
    try:
        return FallbackPipeline()
    except Exception as e:
        st.error(f"No se pudo cargar el pipeline: {e}")
        return None

@st.cache_data
def load_sample_corpus():
    if os.path.exists(SAMPLE_CSV):
        return pd.read_csv(SAMPLE_CSV).head(200)
    return None

def run_analysis(data: pd.DataFrame):
    pipeline = get_pipeline()
    if pipeline is None:
        return None, None
    if "text" not in data.columns:
        st.error("El CSV debe tener una columna 'text'")
        return None, None

    texts = data["text"].fillna("").tolist()
    results = pipeline.batch_predict(texts)
    rows = []
    for r in results:
        sent = r["sentiment"]
        intt = r["intent"]
        churn = r["churn"]
        rows.append({
            "text": r["text"],
            "sentiment": sent["label"],
            "sentiment_prob": sent["probability"],
            "sentiment_neg_prob": sent["probabilities"]["negative"],
            "sentiment_neu_prob": sent["probabilities"]["neutral"],
            "sentiment_pos_prob": sent["probabilities"]["positive"],
            "intent": intt["intent"],
            "intent_prob": intt["probability"],
            "churn_score": churn["aggregate_score"],
            "sentiment_contrib": churn["sentiment_contribution"],
            "frustration_contrib": churn["frustration_contribution"],
            "intent_contrib": churn["intent_contribution"],
        })
    df_results = pd.DataFrame(rows)
    model_status = pipeline.get_model_status()
    return df_results, model_status

def sidebar():
    st.sidebar.title("💬 ConversaAI Demo")
    st.sidebar.markdown("### Pipeline 100% Rule-based")
    st.sidebar.info(
        "🤖 **Modo Fallback**\n\n"
        "Sin modelos ML ni dependencias externas.\n"
        "Clasificación por keywords + patrones."
    )

    data_source = st.sidebar.radio(
        "Fuente de datos",
        ["📂 Corpus de ejemplo", "📁 Subir CSV", "✏️ Escribir texto"],
        index=0,
    )

    data = None
    single_text = None

    if data_source == "📂 Corpus de ejemplo":
        data = load_sample_corpus()
        if data is not None:
            st.sidebar.success(f"Corpus cargado: {len(data)} mensajes")
        else:
            st.sidebar.warning("No se encontró corpus_demo_bilingue.csv")
    elif data_source == "📁 Subir CSV":
        uploaded = st.sidebar.file_uploader("Subí tu CSV", type=["csv"])
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
                st.sidebar.success(f"CSV cargado: {len(data)} mensajes")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    else:
        single_text = st.sidebar.text_area(
            "Escribí un mensaje",
            placeholder="Ej: Quiero cancelar mi suscripción, es pésimo el servicio",
            height=120,
        )
        if single_text and single_text.strip():
            data = pd.DataFrame({"text": [single_text.strip()]})

    st.sidebar.divider()
    run_button = st.sidebar.button("▶️ Analizar", type="primary", use_container_width=True, disabled=data is None)

    if run_button and data is not None:
        with st.sidebar.status("Analizando...", expanded=True) as status:
            df_results, model_status = run_analysis(data)
            if df_results is not None and not df_results.empty:
                st.session_state["results_df"] = df_results
                st.session_state["model_status"] = model_status
                status.update(label="Análisis completo", state="complete")
            elif df_results is not None and df_results.empty:
                st.sidebar.warning("El análisis devolvió 0 resultados")
                status.update(label="Sin resultados", state="error")
            else:
                status.update(label="Error", state="error")

    st.sidebar.divider()
    st.sidebar.markdown("### Estado")
    if "model_status" in st.session_state and st.session_state["model_status"]:
        for model, mode in st.session_state["model_status"].items():
            st.sidebar.markdown(f"⚙️ **{model.capitalize()}**: {mode}")
    else:
        st.sidebar.info("Ejecutá 'Analizar' para ver estado")

    st.sidebar.divider()
    st.sidebar.caption(
        "**ConversaAI Demo** · Pipeline Rule-based\n"
        "Desarrollado para Demoday · Sin modelos ML"
    )

    page = st.sidebar.radio("Navegación", ["📊 Demo", "🔍 Explorar", "💡 Acciones"], label_visibility="collapsed")
    return page

def show_banner():
    st.info(
        "⚙️ **Modo Demo — Pipeline Rule-based** · "
        "Clasificación por keywords y patrones regex. "
        "Sin modelos ML, sin dependencias externas. "
        "Listo para demostración.",
        icon="🤖",
    )

def show_metrics_row(df):
    st.header("📊 Métricas")
    if df.empty:
        st.warning("No hay datos para mostrar métricas")
        return
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Mensajes", f"{len(df):,}")
    with col2:
        st.metric("Churn Promedio", f"{df['churn_score'].mean():.3f}")
    with col3:
        high = (df["churn_score"] >= 0.7).sum()
        st.metric("Alto Riesgo (≥0.7)", f"{high:,}", delta=f"{high/len(df)*100:.1f}%", delta_color="inverse")
    with col4:
        neg_pct = (df["sentiment"] == "negative").mean() * 100
        st.metric("Sentimiento Negativo", f"{neg_pct:.1f}%", delta=f"{(df['sentiment']=='negative').sum():,} msgs", delta_color="inverse")

def show_sentiment_pie(df):
    st.subheader("🎭 Distribución de Sentimiento")
    counts = df["sentiment"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
    fig = go.Figure(data=[go.Pie(
        labels=[s.capitalize() for s in counts.index],
        values=counts.values,
        marker=dict(colors=[SENTIMENT_COLORS[s] for s in counts.index]),
        textinfo="label+percent",
    )])
    fig.update_layout(height=320, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

def show_intent_bar(df):
    st.subheader("🎯 Distribución de Intenciones")
    counts = df["intent"].value_counts().reindex(INTENT_LABELS, fill_value=0)
    labels = [INTENT_DISPLAY.get(i, i.replace("_", " ").title()) for i in counts.index]
    fig = go.Figure(data=[go.Bar(x=labels, y=counts.values, marker_color="#9b59b6")])
    fig.update_layout(height=320, xaxis=dict(tickangle=45), margin=dict(t=0, b=80, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

CHURN_BAJO = 0.4
CHURN_ALTO = 0.7

def classify_churn(score):
    if score >= CHURN_ALTO:
        return "Alto"
    if score >= CHURN_BAJO:
        return "Medio"
    return "Bajo"

CHURN_CLASS_COLORS = {"Alto": "#e74c3c", "Medio": "#f39c12", "Bajo": "#2ecc71"}

def show_churn_histogram(df):
    st.header("⚠️ Distribución de Churn")
    if df.empty:
        st.warning("No hay datos para el histograma de churn")
        return CHURN_ALTO
    df["churn_class"] = df["churn_score"].apply(classify_churn)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["churn_score"], nbinsx=40, marker_color="#3498db",
                                   name="Score"))
        fig.add_vline(x=CHURN_BAJO, line_dash="dash", line_color="#f39c12",
                      annotation_text=f" Medio ({CHURN_BAJO})", annotation_position="top left")
        fig.add_vline(x=CHURN_ALTO, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f" Alto ({CHURN_ALTO})", annotation_position="top left")
        fig.update_layout(height=320, xaxis=dict(title="Churn Score", range=[0, 1]),
                          margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        counts = df["churn_class"].value_counts()
        st.markdown("**Clasificación**")
        for cls, color in CHURN_CLASS_COLORS.items():
            c = counts.get(cls, 0)
            pct = c / len(df) * 100
            st.markdown(f"<span style='color:{color}'>⬤</span> **{cls}**: {c} ({pct:.1f}%)",
                        unsafe_allow_html=True)
    with col3:
        threshold = st.slider("Threshold", 0.0, 1.0, CHURN_ALTO, 0.05)
        high = (df["churn_score"] >= threshold).sum()
        st.metric("Alto Riesgo", f"{high}", delta=f"{high/len(df)*100:.1f}%", delta_color="inverse")
    return threshold

def show_high_risk_table(df, threshold):
    st.header(f"🔴 Alto Riesgo (Churn ≥ {threshold:.2f})")
    high_risk = df[df["churn_score"] >= threshold]
    if high_risk.empty:
        st.success("No hay mensajes de alto riesgo")
        return
    high = high_risk.copy()
    high["churn_class"] = high["churn_score"].apply(classify_churn)
    display = high[["text", "sentiment", "intent", "churn_score", "churn_class"]].copy()
    display["sentiment"] = display["sentiment"].apply(lambda s: f"{'🔴' if s=='negative' else '🟡' if s=='neutral' else '🟢'} {s.capitalize()}")
    display["intent"] = display["intent"].apply(lambda i: INTENT_DISPLAY.get(i, i.replace("_", " ").title()))
    COLOR_DOT = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}
    display["churn_class"] = display["churn_class"].apply(lambda c: f"{COLOR_DOT[c]} {c}")
    display["churn_score"] = display["churn_score"].apply(lambda x: f"{x:.3f}")
    display = display.rename(columns={"text": "Mensaje", "sentiment": "Sentimiento", "intent": "Intención",
                                      "churn_score": "Churn", "churn_class": "Riesgo"})
    st.dataframe(display, use_container_width=True, height=350)

def show_explorer(df):
    st.title("🔍 Explorador de Mensajes")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filtro_sent = st.selectbox("Sentimiento", ["Todos", "negative", "neutral", "positive"])
    with col2:
        filtro_int = st.selectbox("Intención", ["Todos"] + INTENT_LABELS,
                                  format_func=lambda x: INTENT_DISPLAY.get(x, x.replace("_", " ").title()) if x != "Todos" else x)
    with col3:
        filtro_churn_class = st.selectbox("Riesgo Churn", ["Todos", "Alto", "Medio", "Bajo"])
    with col4:
        min_c, max_c = st.slider("Churn Score", 0.0, 1.0, (0.0, 1.0), 0.05)

    filtered = df.copy()
    filtered["churn_class"] = filtered["churn_score"].apply(classify_churn)
    if filtro_sent != "Todos":
        filtered = filtered[filtered["sentiment"] == filtro_sent]
    if filtro_int != "Todos":
        filtered = filtered[filtered["intent"] == filtro_int]
    if filtro_churn_class != "Todos":
        filtered = filtered[filtered["churn_class"] == filtro_churn_class]
    filtered = filtered[(filtered["churn_score"] >= min_c) & (filtered["churn_score"] <= max_c)]

    st.write(f"### {len(filtered)} mensajes encontrados")
    display = filtered[["text", "sentiment", "intent", "churn_score", "churn_class"]].head(100).copy()
    display["sentiment"] = display["sentiment"].apply(lambda s: f"{'🔴' if s=='negative' else '🟡' if s=='neutral' else '🟢'} {s.capitalize()}")
    display["intent"] = display["intent"].apply(lambda i: INTENT_DISPLAY.get(i, i.replace("_", " ").title()))
    COLOR_DOT = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}
    display["churn_class"] = display["churn_class"].apply(lambda c: f"{COLOR_DOT[c]} {c}")
    display["churn_score"] = display["churn_score"].apply(lambda x: f"{x:.3f}")
    display = display.rename(columns={"text": "Mensaje", "sentiment": "Sentimiento", "intent": "Intención",
                                      "churn_score": "Churn", "churn_class": "Riesgo"})
    st.dataframe(display, use_container_width=True, height=400)

def show_demo_page(df):
    st.title("💬 ConversaAI — Demo Dashboard")
    st.markdown("### Sistema de Análisis de Conversaciones de Soporte")
    show_banner()
    show_metrics_row(df)
    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        show_sentiment_pie(df)
    with col_r:
        show_intent_bar(df)
    st.divider()
    threshold = show_churn_histogram(df)
    show_high_risk_table(df, threshold)

def show_recommendations(df):
    st.title("💡 Recomendaciones para el Equipo de Producto")

    st.markdown("### 📊 Churn Promedio por Intención")
    intent_churn = df.groupby("intent")["churn_score"].mean().reindex(INTENT_LABELS, fill_value=0)
    intent_labels = [INTENT_DISPLAY.get(i, i.replace("_", " ").title()) for i in intent_churn.index]
    churn_overall = df["churn_score"].mean()
    fig = go.Figure()
    colors = ["#e74c3c" if v > churn_overall else "#2ecc71" for v in intent_churn.values]
    fig.add_trace(go.Bar(x=intent_labels, y=intent_churn.values, marker_color=colors))
    fig.add_hline(y=churn_overall, line_dash="dash", line_color="#7f8c8d",
                  annotation_text=f"Promedio general: {churn_overall:.3f}")
    fig.update_layout(height=350, xaxis=dict(tickangle=45), margin=dict(t=0, b=80, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Barras rojas = por encima del promedio. Priorizar estas intenciones.")

    st.markdown("### 📊 Distribución por Clase de Churn")
    df["churn_class"] = df["churn_score"].apply(classify_churn)
    class_counts = df["churn_class"].value_counts()
    fig2 = go.Figure(data=[go.Pie(
        labels=class_counts.index.tolist(),
        values=class_counts.values.tolist(),
        marker=dict(colors=[CHURN_CLASS_COLORS[c] for c in class_counts.index]),
        textinfo="label+percent",
    )])
    fig2.update_layout(height=280, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.divider()

    st.markdown("### 🎯 Top 3 Intenciones con Mayor Churn")
    top3_churn = df.groupby("intent")["churn_score"].mean().sort_values(ascending=False).head(3)
    col1, col2, col3 = st.columns(3)
    for col, (intent, avg) in zip([col1, col2, col3], top3_churn.items()):
        display = INTENT_DISPLAY.get(intent, intent.replace("_", " ").title())
        count = len(df[df["intent"] == intent])
        with col:
            st.metric(display, f"{avg:.3f}", delta=f"{count} mensajes", delta_color="inverse")
    st.divider()

    st.markdown("### 🔍 Desglose de Churn por Intención")
    contrib = df.groupby("intent")[["sentiment_contrib", "frustration_contrib", "intent_contrib"]].mean()
    contrib = contrib.reindex(INTENT_LABELS, fill_value=0)
    intent_labels_contrib = [INTENT_DISPLAY.get(i, i.replace("_", " ").title()) for i in contrib.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sentimiento", x=intent_labels_contrib, y=contrib["sentiment_contrib"],
                         marker_color="#3498db"))
    fig.add_trace(go.Bar(name="Frustración", x=intent_labels_contrib, y=contrib["frustration_contrib"],
                         marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="Intención", x=intent_labels_contrib, y=contrib["intent_contrib"],
                         marker_color="#9b59b6"))
    fig.update_layout(barmode="stack", height=350, xaxis=dict(tickangle=45), margin=dict(t=0, b=80, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Azul = sentimiento negativo · Rojo = frustración por keywords · Violeta = intención de riesgo")
    st.divider()

    st.markdown("### 🚨 Alertas y Acciones Sugeridas")
    top3_churn_list = df.groupby("intent")["churn_score"].mean().sort_values(ascending=False).head(3)
    alerts = []
    for intent, avg_churn in top3_churn_list.items():
        display = INTENT_DISPLAY.get(intent, intent.replace("_", " ").title())
        count = len(df[df["intent"] == intent])
        if intent == "cancelacion":
            action = "🔴 Revisar proceso de retención — ofrecer descuentos o beneficios"
        elif intent == "queja":
            action = "🔴 Escalar a equipo de calidad — contactar al cliente en < 24h"
        elif intent == "reembolso":
            action = "🟠 Agilizar proceso de reembolso — priorizar sobre otros tickets"
        elif intent == "facturacion_pago":
            action = "🟠 Revisar errores de cobro — verificar integraciones con gateway"
        else:
            action = "🟡 Investigar causa raíz — revisar conversaciones recientes"
        alerts.append((display, count, avg_churn, action))
    if alerts:
        for display, count, avg_churn, action in alerts:
            with st.expander(f"⚠️ {display} — {count:,} casos (churn prom. {avg_churn:.2f})"):
                st.markdown(action)
    else:
        st.success("✅ No hay alertas críticas en este lote de datos")
    st.divider()

    st.markdown("### 📋 Resumen Ejecutivo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total mensajes", f"{len(df):,}")
    with col2:
        pct_neg = (df["sentiment"] == "negative").mean() * 100
        st.metric("Sentimiento negativo", f"{pct_neg:.1f}%")
    with col3:
        high_risk_pct = (df["churn_score"] >= 0.7).mean() * 100
        st.metric("Alto riesgo de churn", f"{high_risk_pct:.1f}%")


def main():
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "model_status" not in st.session_state:
        st.session_state["model_status"] = None

    page = sidebar()

    if st.session_state["results_df"] is None:
        st.title("💬 ConversaAI — Demo Dashboard")
        st.markdown(
            "### Sistema de Análisis de Conversaciones de Soporte\n\n"
            "Cargá datos desde el panel izquierdo y presioná **▶️ Analizar**.\n\n"
            "El pipeline (100% rule-based) clasifica:\n"
            "- **Sentimiento** (3 clases: negativo, neutral, positivo) por keywords\n"
            "- **Intención** (9 clases) por patrones regex\n"
            "- **Churn Score** combinando sentimiento + frustración + intención\n\n"
            "Usá el *Corpus de ejemplo* precargado, subí tu CSV, o escribí un texto."
        )
        show_banner()
        return

    df = st.session_state["results_df"]

    if page.startswith("📊 Demo"):
        show_demo_page(df)
    elif page.startswith("🔍 Explorar"):
        show_explorer(df)
    elif page.startswith("💡 Acciones"):
        show_recommendations(df)

if __name__ == "__main__":
    main()
