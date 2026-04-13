import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(
    page_title="Predictive Maintenance NASA AI",
    page_icon="✈️",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"

st.title("🛰️ NASA Turbofan - Predictive Maintenance Dashboard")
st.markdown("""
Bienvenido al panel central de comando. Sube el registro histórico de telemetría extraído de la caja negra de uno (o varios) motores Turbofán comerciales.
Nuestro motor híbrido de Inteligencia Artificial (K-Means + Optuna XGBoost) descontaminará los efectos del clima y te dirá si el motor **fallará catastróficamente en sus próximos 30 vuelos.**
""")

st.divider()

# Columna de subida
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Carga de Telemetría RAW")
    uploaded_file = st.file_uploader("Arrastra el archivo plano (.txt o .csv)", type=['txt', 'csv'])
    
    if uploaded_file is not None:
        st.success("Archivo interceptado correctamente.")
        
        # Opcional: mostrar una preview
        file_bytes = uploaded_file.getvalue()
        df_preview = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", header=None)
        st.write(f"Previsualización rápida: {df_preview.shape[0]} ciclos y {df_preview.shape[1]} columnas interceptados.")
        
        if st.button("🚀 INICIAR ESCANEO MLOps", type="primary"):
            with st.spinner('Purgando ruido ambiental, operando redes no-supervisadas de inferencia...'):
                try:
                    # Enviar petición POST a FastAPI local
                    files = {'file': (uploaded_file.name, file_bytes, 'text/plain')}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        resultados = response.json()
                        st.session_state['predicciones'] = resultados['analisis']
                    else:
                        st.error(f"Error de Servidor Droid: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ El motor MLOps (FastAPI) está apagado. Ejecuta `uvicorn src.api:app --reload` en consola.")

with col2:
    st.subheader("2. Veredicto Cíber-Físico")
    
    if 'predicciones' in st.session_state:
        # Dibujar por cada motor
        for res in st.session_state['predicciones']:
            motor_id = res['unit_id']
            ciclo = res['last_cycle_recorded']
            estado = res['status']
            prob = res['failure_probability_percent']
            accion = res['action']
            
            with st.container():
                st.markdown(f"### Motor de Análisis `ID: {motor_id}`")
                st.caption(f"Última transmisión interceptada: Ciclo número {ciclo}")
                
                if estado == "SANO":
                    st.success(f"**Veredicto MLOps:** {estado} (Riesgo: {prob}%)")
                    st.info(f"**Directiva del Sistema:** {accion}")
                else:
                    st.error(f"**Veredicto MLOps:** {estado} (Riesgo Crítico: {prob}%)")
                    st.warning(f"**Directiva del Sistema:** {accion}", icon="⚠️")
                    
                st.divider()
    else:
        st.info("Esperando inicialización de datos para proyectar hologramas predictivos.")
        st.image("https://images.unsplash.com/photo-1540962351504-03099e0a754b?q=80&w=2000&auto=format&fit=crop", caption="Jet Engine Core", use_container_width=True)
