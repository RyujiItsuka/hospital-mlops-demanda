import streamlit as st
import pandas as pd
import joblib

# Configurar el título de la aplicación web
st.title("🏥 Sistema de Predicción de Demanda Hospitalaria")
st.write("Ingrese las variables operativas del día para predecir la afluencia de pacientes.")

# Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_demanda_hospital.pkl')

modelo = cargar_modelo()

# Formulario interactivo para que el usuario ingrese datos
st.sidebar.header("Parámetros del Día")
dia_semana = st.sidebar.selectbox("Día de la semana", ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado"])
mes = st.sidebar.slider("Mes del año", 1, 12, 5)
porcentaje_viejos = st.sidebar.slider("Porcentaje de Adultos Mayores (%)", 0, 100, 25) / 100.0
prom_t_cita = st.sidebar.number_input("Tiempo promedio de cita (minutos)", value=23.5)
max_t_cita = st.sidebar.number_input("Tiempo máximo de cita (minutos)", value=46.0)
min_t_cita = st.sidebar.number_input("Tiempo mínimo de cita (minutos)", value=4.0)

# Horarios de atención
primera_hora = st.sidebar.time_input("Primera hora de atención", pd.to_datetime("08:00").time())
ultima_hora = st.sidebar.time_input("Última hora de atención", pd.to_datetime("13:00").time())

# Convertir hora a minutos desde la medianoche
primera_hora_min = primera_hora.hour * 60 + primera_hora.minute
ultima_hora_min = ultima_hora.hour * 60 + ultima_hora.minute

# Botón para predecir
if st.button("Calcular Demanda Predicha"):
    # Crear el nuevo caso como DataFrame
    nuevo_caso = pd.DataFrame([{
        "% VIEJOS": porcentaje_viejos,
        "PROM_T_CITA": prom_t_cita,
        "MAX_T_CITA": max_t_cita,
        "MIN_T_CITA": min_t_cita,
        "Mes": mes,
        "PRIMERA_HORA_MIN": primera_hora_min,
        "ULTIMA_HORA_MIN": ultima_hora_min,
        "DIA_SEMANA": dia_semana
    }])
    
    # Realizar la predicción
    prediccion = modelo.predict(nuevo_caso)[0]
    pacientes_est = int(round(prediccion))
    
    # Mostrar resultados en la interfaz
    st.success(f"📊 Demanda estimada para este día: **{pacientes_est} pacientes**")
    
    # Recomendación prescriptiva básica
    if pacientes_est > 160:
        st.warning("⚠️ Alta afluencia proyectada: Se recomienda habilitar ventanillas adicionales.")
    else:
        st.info("✅ Afluencia dentro del rango operativo normal.")