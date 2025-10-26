"""
API de Bitcoin - SOLO DATOS REALES, NO SAMPLES
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # Cambiar pickle por joblib
import pandas as pd
import os
import sys
import random
from datetime import datetime
from typing import Dict, Optional
import requests
from ..models.bitcoin_api_models import PredictionRequest, PredictionResponse

# Importar constantes desde el paquete api
from .. import constants as const

# Importar sistema de logging estandarizado
from ..config_logger import get_api_logger, log_model_loading, log_prediction

app = FastAPI(title="Bitcoin Real Data Prediction API", version="1.0.0")

# Logger estandarizado
logger = get_api_logger("bitcoin_api", console_output=False)

# Variable global para el modelo
_loaded_model_data = None

def load_bitcoin_model():
    """Carga el modelo Prophet entrenado"""
    global _loaded_model_data
    if _loaded_model_data is not None:
        return _loaded_model_data
    
    try:
        model_path = os.path.join(const.BASE_DIR, 'ml_models', 'prophet_bitcoin_v2_2025-10-24.pkl')
        _loaded_model_data = joblib.load(model_path)  # Usar joblib.load en lugar de pickle.load
        
        # Log successful model loading
        log_model_loading(logger, "Bitcoin Prophet", model_path, True)
        
        return _loaded_model_data
    except Exception as e:
        # Log failed model loading
        log_model_loading(logger, "Bitcoin Prophet", model_path, False, str(e))
        raise HTTPException(status_code=500, detail=f"Error cargando modelo Prophet: {str(e)}")

def make_prophet_prediction(model, dates):
    """Genera predicciones usando el modelo Prophet"""

    # Crear DataFrame con las fechas proporcionadas
    future = pd.DataFrame({'ds': pd.to_datetime(dates)})
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


@app.get("/")
def root():
    return {"message": "Bitcoin REAL Data Prediction API", "version": "3.0.0", "note": "Uses real market data"}

@app.post("/models/bitcoin/predict", response_model=dict)
def predict_bitcoin_price(request: PredictionRequest):
    """
    Predice precios de Bitcoin usando modelo Prophet v2
    """
    try:
        logger.info(f"Predicción Bitcoin solicitada: {len(request.dates)} fechas")
        
        model = load_bitcoin_model()
        forecast = make_prophet_prediction(model, request.dates)

        results = [
            {
                "date": row.ds.strftime("%Y-%m-%d"),
                "predicted_price": round(row.yhat, 2),
                "lower_bound": round(row.yhat_lower, 2),
                "upper_bound": round(row.yhat_upper, 2)
            }
            for _, row in forecast.iterrows()
        ]

        # Log prediction success
        log_prediction(
            logger, 
            endpoint="/models/bitcoin/predict", 
            input_payload={"dates_count": len(request.dates), "dates": request.dates[:3]}, 
            output_summary={"predictions_count": len(results), "avg_price": round(sum(r["predicted_price"] for r in results) / len(results), 2)}
        )

        return {
            "model_version": "Prophet v2",
            "dates_predicted": len(results),
            "predictions": results
        }
    except Exception as e:
        logger.error(f"Error en predicción Bitcoin: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/health")
def health():
    logger.info("Health check solicitado para Bitcoin API")
    try:
        model_data = load_bitcoin_model()
        logger.info("Health check Bitcoin: modelo cargado exitosamente")
        return {
            "status": "healthy",
            "model_loaded": True,
            "real_data_available": True,
            "current_btc_price": model_data['current_price'],
            "data_source": model_data['source']
        }
    except Exception as e:
        error_msg = f"Error en health check de Bitcoin: {str(e)}"
        logger.error(error_msg)
        return {"status": "unhealthy", "error": error_msg, "model_loaded": False}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Iniciando Bitcoin REAL Data Prediction API...")
    print("₿ Bitcoin Real Data Prediction API")
    uvicorn.run(app, host="0.0.0.0", port=8000)