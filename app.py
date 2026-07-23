import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción Hospital Ica", page_icon="🏥")

st.title("🏥 Predicción de Demanda de Pacientes")
st.write("Hospital Regional de Ica - Analítica Predictiva")

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_demanda_hospital.pkl")
    variables = joblib.load("variables_modelo.pkl")
    return modelo, variables

modelo, variables = cargar_modelo()

st.success("✅ Modelo .pkl cargado y activo.")

st.subheader("📋 Ingrese los datos operativos del día:")

col1, col2 = st.columns(2)

with col1:
    dia_semana = st.selectbox("Día de la semana", ["lunes", "martes", "miercoles", "jueves", "viernes"])
    pct_viejos = st.slider("Porcentaje de Adultos Mayores (%)", 0, 100, 25) / 100.0
    mes = st.slider("Mes", 1, 12, 5)

with col2:
    prom_t_cita = st.number_input("Tiempo promedio de cita (min)", value=23.5)
    max_t_cita = st.number_input("Tiempo máximo de cita (min)", value=46.0)
    min_t_cita = st.number_input("Tiempo mínimo de cita (min)", value=4.0)

primera_hora_min = 480.0
ultima_hora_min = 780.0

if st.button("🔮 Calcular Afluencia Predicha"):
    nuevo_caso = pd.DataFrame([{
        "% VIEJOS": pct_viejos,
        "PROM_T_CITA": prom_t_cita,
        "MAX_T_CITA": max_t_cita,
        "MIN_T_CITA": min_t_cita,
        "Mes": mes,
        "PRIMERA_HORA_MIN": primera_hora_min,
        "ULTIMA_HORA_MIN": ultima_hora_min,
        "DIA_SEMANA": dia_semana
    }])
    
    prediccion = modelo.predict(nuevo_caso[variables])[0]
    
    st.metric(label="Pacientes Estimados", value=f"{int(round(prediccion))} personas")
