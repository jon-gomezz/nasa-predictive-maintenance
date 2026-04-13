import os
import sys
from pathlib import Path
import io

import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import warnings

# Ignorar un warning molesto de sklearn con xgboost
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar rutas para encontrar modulos locales
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import ALL_COLUMNS
from src.features import add_rolling_features, ConditionNormalizer

app = FastAPI(
    title="Predictive Maintenance API (FD002)",
    description="Motor de Inferencia MLOps para motores Turbofán de la NASA",
    version="1.0.0"
)

# Variables globales para cachear los modelos en RAM al iniciar
MODEL = None
NORMALIZER = None
EXPECTED_FEATURES = None

@app.on_event("startup")
def load_models():
    """
    Carga de modelos en memoria (Cold Start).
    Se ejecuta automáticamente al levantar uvicorn.
    """
    global MODEL, NORMALIZER, EXPECTED_FEATURES
    base_dir = Path(__file__).resolve().parent.parent / 'models'
    
    try:
        print("Cargando pipeline Híbrido K-Means + XGBoost en RAM...")
        NORMALIZER = joblib.load(base_dir / 'condition_normalizer_fd002.joblib')
        MODEL = joblib.load(base_dir / 'xgboost_fd002_tuned.joblib')
        EXPECTED_FEATURES = MODEL.feature_names_in_
        print(">>> Modelos Productivos cargados y en escucha. <<<")
    except Exception as e:
        print(f"!!! Error catastrofico cargando modelos: {e}")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "message": "Bienvenido al Sistema de Alerta Temprana de Turbinas."
    }

@app.post("/predict")
async def predict_engine_telemetry(file: UploadFile = File(...)):
    """
    Recibe un arrastre histórico (o múltiples arrastres) de telemetría de motor.
    La arquitectura extrae la Media Móvil más reciente, normaliza los climas,
    y devuelve la probabilidad de Colapso Crítico (antes de 30 ciclos).
    """
    if MODEL is None or NORMALIZER is None:
        raise HTTPException(status_code=503, detail="Los modelos no están cargados. Reinicie el servidor.")
        
    try:
        # 1. Leer el archivo binario subido a la memoria pandas
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=r"\s+", header=None, names=ALL_COLUMNS)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"El archivo debe ser un .txt o .csv con formato NASA C-MAPSS. Error: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="El archivo subido no contiene datos válidos.")

    try:
        # 2. Pipeline de Preprocesamiento MLOps
        df_norm = NORMALIZER.transform(df)
        df_temporal = add_rolling_features(df_norm, windows=[5, 15])
        df_temporal.fillna(0, inplace=True)
        
        # 3. Extraer la última "fotografía" de salud por cada motor detectado en el archivo
        latest_cycles = df_temporal.groupby('unit_id').last().reset_index()
        
        # 4. Inferencia
        X_live = latest_cycles[EXPECTED_FEATURES]
        preds = MODEL.predict(X_live)
        probs = MODEL.predict_proba(X_live)[:, 1] # Probabilidad de la clase 1 (Fallo)
        
        # 5. Estructurar respuesta de negocio
        resultados = []
        for i, unit_id in enumerate(latest_cycles['unit_id']):
            prob_fallo = float(probs[i])
            if prob_fallo >= 0.5:
                # Alarma roja
                estado = "PELIGRO INMINENTE"
                recomendacion = "Solicitar Inspección en Pista INMEDIATAMENTE al Aterrizar."
            else:
                estado = "SANO"
                recomendacion = "Continuar Operaciones Comerciales."
                
            resultados.append({
                "unit_id": int(unit_id),
                "last_cycle_recorded": int(latest_cycles['time_cycle'].iloc[i]),
                "status": estado,
                "failure_probability_percent": round(prob_fallo * 100, 2),
                "action": recomendacion
            })
            
        return {"analisis": resultados}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error durante el MLOps Pipeline interno: {str(e)}")
