import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Predicción Hospital Ica", page_icon="🏥")

st.title("🏥 Predicción de Demanda de Pacientes")
st.write("Hospital Regional de Ica - Analítica Predictiva")

@st.cache_resource
def entrenar_modelo():

    df = pd.read_excel("DatosHopital.xlsx", sheet_name="Resumen")
    
    df['Mes'] = pd.to_datetime(df['FECHA']).dt.month
    
    def hora_a_minutos(h):
        if pd.isnull(h):
            return np.nan
        if isinstance(h, str):
            h = pd.to_datetime(h).time()
        return h.hour * 60 + h.minute + h.second / 60.0

    df['PRIMERA_HORA_MIN'] = df['PRIMERA_HORA'].apply(hora_a_minutos)
    df['ULTIMA_HORA_MIN'] = df['ULTIMA_HORA'].apply(hora_a_minutos)

    objetivo = "CLIENTES"
    variables_numericas = ["% VIEJOS", "PROM_T_CITA", "MAX_T_CITA", "MIN_T_CITA", "Mes", "PRIMERA_HORA_MIN", "ULTIMA_HORA_MIN"]
    variables_categoricas = ["DIA_SEMANA"]
    variables = variables_numericas + variables_categoricas

    df_modelo = df[variables + [objetivo]].dropna()
    X = df_modelo[variables]
    y = df_modelo[objetivo]

    preprocesamiento = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), variables_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), variables_categoricas)
        ]
    )

    modelo_rf = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", RandomForestRegressor(n_estimators=200, random_state=42))
        ]
    )

    modelo_rf.fit(X, y)
    return modelo_rf, variables

modelo, variables = entrenar_modelo()

st.success("✅ Modelo Random Forest cargado y activo.")

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
