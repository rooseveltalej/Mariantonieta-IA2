from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal
from datetime import datetime

class AvocadoPredictionRequest(BaseModel):
    """Request para predicción de precios de aguacate"""
    
    # Información temporal requerida
    date: str = Field(..., description="Fecha para la predicción (YYYY-MM-DD)", example="2024-12-01")
    region: str = Field(..., description="Región geográfica", example="California")
    type: Literal["conventional", "organic"] = Field(..., description="Tipo de aguacate", example="conventional")
    
    # Datos de volumen (opcional, se pueden usar valores por defecto)
    total_volume: Optional[float] = Field(default=150000.0, description="Volumen total de ventas", ge=0)
    plu_4046: Optional[float] = Field(default=60000.0, description="Volumen PLU 4046 (aguacates pequeños)", ge=0)
    plu_4225: Optional[float] = Field(default=40000.0, description="Volumen PLU 4225 (aguacates grandes)", ge=0)
    plu_4770: Optional[float] = Field(default=1000.0, description="Volumen PLU 4770 (aguacates extra grandes)", ge=0)
    total_bags: Optional[float] = Field(default=50000.0, description="Total de bolsas vendidas", ge=0)
    small_bags: Optional[float] = Field(default=40000.0, description="Bolsas pequeñas", ge=0)
    large_bags: Optional[float] = Field(default=8000.0, description="Bolsas grandes", ge=0)
    xlarge_bags: Optional[float] = Field(default=2000.0, description="Bolsas extra grandes", ge=0)
    
    # Datos históricos opcionales (para lags y rolling means)
    historical_prices: Optional[list[float]] = Field(default=None, description="Precios históricos para cálculo de lags")

class AvocadoPredictionResponse(BaseModel):
    """Response con predicción de precio de aguacate"""
    
    prediction: float = Field(..., description="Precio predicho del aguacate")
    confidence: float = Field(..., description="Nivel de confianza de la predicción (0-100)")
    model_info: Dict = Field(..., description="Información sobre el modelo utilizado")
    interpretation: str = Field(..., description="Interpretación en lenguaje natural")
    
    # Información adicional específica de aguacate
    price_category: str = Field(..., description="Categoría del precio (bajo, medio, alto)")
    market_context: Dict = Field(..., description="Contexto del mercado (región, temporada, etc.)")
    feature_importance: Optional[Dict] = Field(default=None, description="Importancia de las características")

class AvocadoHealthResponse(BaseModel):
    """Response para health check del modelo de aguacate"""
    
    status: str = Field(..., description="Estado del modelo")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    model_type: str = Field(..., description="Tipo de modelo")
    features_count: int = Field(..., description="Número de características del modelo")
    supported_regions: list[str] = Field(..., description="Regiones soportadas")
    supported_types: list[str] = Field(..., description="Tipos de aguacate soportados")
    model_metrics: Optional[Dict] = Field(default=None, description="Métricas del modelo")

class AvocadoMarketAnalysisRequest(BaseModel):
    """Request para análisis de mercado de aguacate"""
    
    region: str = Field(..., description="Región a analizar")
    start_date: str = Field(..., description="Fecha de inicio (YYYY-MM-DD)")
    end_date: str = Field(..., description="Fecha de fin (YYYY-MM-DD)")
    type: Optional[Literal["conventional", "organic"]] = Field(default="conventional", description="Tipo de aguacate")

class AvocadoMarketAnalysisResponse(BaseModel):
    """Response con análisis de mercado de aguacate"""
    
    region: str = Field(..., description="Región analizada")
    average_price: float = Field(..., description="Precio promedio en el período")
    price_trend: str = Field(..., description="Tendencia del precio (alcista, bajista, estable)")
    seasonal_patterns: Dict = Field(..., description="Patrones estacionales identificados")
    recommendations: list[str] = Field(..., description="Recomendaciones basadas en el análisis")
    forecast_next_month: float = Field(..., description="Pronóstico del precio para el próximo mes")