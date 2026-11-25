"""
app.py
Dashboard de Mantenimiento Predictivo + Órdenes de Trabajo con Groq.
Solo depende de modelo_mantenimiento.pkl (no lee el CSV).
"""
import random
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Mantenimiento Predictivo", page_icon="🏭", layout="wide")

FEATURES = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
ESTADOS = {
    0: "Operación Normal",
    1: "Mantenimiento Preventivo",
    2: "Fallo Crítico",
}


# ---------------- Carga del modelo ----------------
RUTA_MODELO = Path(__file__).resolve().parent / "modelo_mantenimiento.pkl"


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar modelo_mantenimiento.pkl. Ejecuta primero entrenar.py. Detalle: {e}")
    st.stop()


# ---------------- Sidebar: controles de sensores ----------------
st.sidebar.header("🎛️ Lecturas de Sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", 40.0, 110.0, 60.0, 0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 12.0, 1.5, 0.1)
horas = st.sidebar.number_input("Horas de operación", min_value=0, max_value=6000, value=300, step=10)
presion = st.sidebar.slider("Presión (PSI)", 70.0, 170.0, 92.0, 0.5)

# ---------------- Encabezado y métricas ----------------
st.title("🏭 Sistema Predictivo de Mantenimiento Industrial")
st.caption("Random Forest + Órdenes de Trabajo generadas con Groq")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

# ---------------- Predicción ----------------
entrada = pd.DataFrame([[temperatura, vibracion, horas, presion]], columns=FEATURES)
estado = int(modelo.predict(entrada)[0])
probabilidades = modelo.predict_proba(entrada)[0]
clases = [int(c) for c in modelo.classes_]
prob_estado = probabilidades[clases.index(estado)]

st.subheader("🔍 Diagnóstico del Modelo")
mensaje = f"Estado {estado}: **{ESTADOS[estado]}** — Probabilidad: {prob_estado * 100:.1f}%"
if estado == 0:
    st.success(f"✅ {mensaje}")
elif estado == 1:
    st.warning(f"⚠️ {mensaje}")
else:
    st.error(f"🚨 {mensaje}")

st.write("Probabilidad por estado:")
df_prob = pd.DataFrame(
    {"Estado": [ESTADOS[c] for c in clases], "Probabilidad (%)": [round(p * 100, 1) for p in probabilidades]}
)
st.bar_chart(df_prob.set_index("Estado"))


# ---------------- Groq: Orden de Trabajo ----------------
SYSTEM_PROMPT = (
    "Eres el Jefe de Mantenimiento Industrial de una planta de manufactura automatizada. "
    "Respondes en español, con tono técnico, claro y directo. "
    "Basas tus indicaciones en las lecturas de sensores y el diagnóstico del modelo de Machine Learning. "
    "Rangos de referencia de la planta (no inventes otros):\n"
    "- Normal: temperatura 57-66 °C, vibración 1.1-2.3 mm/s, menos de 750 h, presión 89-99 PSI.\n"
    "- Desgaste (preventivo): temperatura 75-84 °C, vibración 3.7-5.3 mm/s, 1,500-2,400 h, presión 109-123 PSI.\n"
    "- Crítico: temperatura de 91 °C o más, vibración de 7.2 mm/s o más, 3,100 h o más, presión de 132 PSI o más.\n"
    "EPP significa Equipo de Protección Personal (casco, lentes, guantes, botas, etc.); "
    "siempre lístalo como artículos concretos."
)


def construir_prompt(estado, folio):
    lecturas = (
        f"- Temperatura: {temperatura} °C\n"
        f"- Vibración: {vibracion} mm/s\n"
        f"- Horas de operación: {horas} hrs\n"
        f"- Presión: {presion} PSI\n"
        f"- Diagnóstico ML: Estado {estado} ({ESTADOS[estado]}) con {prob_estado * 100:.1f}% de probabilidad"
    )
    if estado == 0:
        return (
            f"Lecturas actuales del equipo:\n{lecturas}\n\n"
            "El equipo opera normalmente. Emite un DICTAMEN BREVE de monitoreo preventivo "
            "(máximo 4 líneas) con una recomendación de rutina. No generes orden de trabajo."
        )
    urgencia = (
        "PREVENTIVA (programar revisión)" if estado == 1
        else "URGENTE: incluye protocolo de PARO INMEDIATO y bloqueo/etiquetado (LOTO)"
    )
    return (
        f"Lecturas actuales del equipo:\n{lecturas}\n\n"
        f"Genera una ORDEN DE TRABAJO {urgencia} en formato Markdown con exactamente estas secciones:\n"
        f"## 🧾 Orden de Trabajo\n**Folio:** {folio}\n**Prioridad:** ...\n"
        "### Diagnóstico\n(máximo 4 líneas: qué lecturas salen del rango normal según los rangos de referencia y el riesgo)\n"
        "### Acciones Técnicas\n(exactamente 3 acciones numeradas, concretas y breves)\n"
        "### Equipo de Seguridad (EPP)\n(lista de viñetas con artículos concretos de Equipo de Protección Personal)\n\n"
        "No repitas información entre secciones ni agregues texto fuera de estas secciones."
    )


@st.cache_resource
def obtener_cliente():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_data(ttl=3600)
def detectar_modelo(_client):
    # Detección dinámica de modelos de chat activos (evita errores 404)
    modelos_activos = sorted(
        m.id for m in _client.models.list().data
        if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "playai", "embed", "allam", "compound"])
    )
    # Orden de preferencia: modelos grandes primero; si ninguno está activo, se usa el primero disponible
    for preferido in ["llama-3.3-70b", "gpt-oss-120b", "70b", "llama"]:
        encontrado = next((m for m in modelos_activos if preferido in m.lower()), None)
        if encontrado:
            return encontrado
    return modelos_activos[0]


st.subheader("🤖 Orden de Trabajo (Groq)")

if st.button("Generar dictamen / Orden de Trabajo", type="primary"):
    folio = f"OT-{datetime.now():%Y%m%d}-{random.randint(1000, 9999)}"
    try:
        client = obtener_cliente()
        modelo_groq = detectar_modelo(client)
        with st.spinner(f"Consultando a Groq ({modelo_groq})..."):
            respuesta = client.chat.completions.create(
                model=modelo_groq,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": construir_prompt(estado, folio)},
                ],
                temperature=0.4,
            )
        st.caption(f"Modelo de Groq utilizado: `{modelo_groq}`")
        st.markdown(respuesta.choices[0].message.content)
    except KeyError:
        st.error("🔑 No se encontró GROQ_API_KEY en st.secrets. Configúrala en .streamlit/secrets.toml o en Streamlit Cloud.")
        st.info(f"Diagnóstico ML disponible: Estado {estado} — {ESTADOS[estado]} ({prob_estado * 100:.1f}%)")
    except Exception as e:
        st.error(f"⚠️ No se pudo generar la orden con Groq (clave inválida, límite de peticiones o red). Detalle: {e}")
        st.info(f"Diagnóstico ML disponible: Estado {estado} — {ESTADOS[estado]} ({prob_estado * 100:.1f}%)")
