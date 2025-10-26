# 📋 Logs Directory / Directorio de Logs

## 🇪🇸 Español

### Descripción General
Esta carpeta contiene todos los archivos de registro (logs) del sistema de predicción con IA. Cada API y componente del sistema genera sus propios logs para facilitar el monitoreo, debugging y análisis de rendimiento.

### 🗂️ Estructura de Archivos

#### **Logs de APIs Principales**
- **`main_api.log`** - API principal y coordinador LLM
  - Registra consultas de usuarios, routing de requests, y comunicación con Ollama
  - Incluye timing de respuestas y manejo de errores
  
- **`acv_api.log`** - API de predicción de riesgo de ACV
  - Carga de modelo Decision Tree
  - Predicciones médicas con parámetros de entrada y resultados
  - Recomendaciones de salud generadas

- **`avocado_api.log`** - API de predicción de precios de aguacate
  - Modelo CatBoost para predicción de precios
  - Features engineering con lags temporales y rolling means
  - Interpretaciones de mercado

- **`bitcoin_api.log`** - API de predicción de precios de Bitcoin
  - Modelo Prophet v2 con datos reales de mercado
  - Predicciones con intervalos de confianza
  - Health checks y estado del modelo

- **`flights_api.log`** - API de predicción de retrasos de vuelos
  - Modelo Random Forest para retrasos
  - Mapeo de aerolíneas y aeropuertos
  - Timing de predicciones

- **`movies_api.log`** - API de recomendación de películas
  - Sistema KNN con similitud de géneros
  - Recomendaciones personalizadas por usuario
  - Predicciones de ratings

#### **Logs del Sistema**
- **`uvicorn.log`** - Servidor web Uvicorn
  - Eventos de arranque y parada del servidor
  - Configuración de puertos y hosts
  
- **`uvicorn_access.log`** - Accesos HTTP
  - Todas las peticiones HTTP con códigos de respuesta
  - IPs de clientes y timing de requests
  
- **`fastapi.log`** - Framework FastAPI
  - Middleware, validaciones y errores del framework
  - Configuración de rutas y dependencias

### 🔧 Configuración de Logs

#### **Características**
- **Rotación automática**: Archivos máximo 5MB, 5 backups por archivo
- **Formato estandarizado**: `YYYY-MM-DD HH:MM:SS - LEVEL - LOGGER - MESSAGE`
- **Codificación**: UTF-8 para soporte completo de caracteres
- **Solo archivos**: No se muestran logs en consola para interfaz limpia

#### **Niveles de Log**
- **INFO**: Operaciones normales, cargas exitosas, predicciones
- **ERROR**: Errores de carga de modelos, fallos de predicción, excepciones
- **WARNING**: Situaciones anómalas que no bloquean la ejecución
- **DEBUG**: Información detallada para desarrollo (desactivado en producción)

### 📊 Monitoreo y Análisis

#### **Comandos Útiles**
```bash
# Ver logs en tiempo real
tail -f main_api.log

# Buscar errores en todos los logs
grep -r "ERROR" .

# Ver últimas 50 líneas de una API específica
tail -50 bitcoin_api.log

# Contar predicciones exitosas del día
grep "$(date +%Y-%m-%d)" acv_api.log | grep "Prediction" | wc -l

# Monitorear múltiples APIs simultáneamente
tail -f main_api.log bitcoin_api.log avocado_api.log
```

#### **Métricas Relevantes**
- **Tiempo de respuesta**: Buscar patrones `Time: X.XXXs`
- **Errores de carga**: `Model Load Failed`
- **Predicciones exitosas**: `Prediction - Endpoint:`
- **Health checks**: `Health check solicitado`

### 🚨 Troubleshooting

#### **Problemas Comunes**
1. **"Model Load Failed"** → Verificar ruta y permisos del archivo modelo
2. **"Connection refused port 11434"** → Ollama no está ejecutándose
3. **"HTTPException 500"** → Error interno, revisar stack trace completo
4. **Archivos de log muy grandes** → Verificar configuración de rotación

#### **Limpieza de Logs**
```bash
# Eliminar logs antiguos (cuidado!)
find . -name "*.log*" -mtime +30 -delete

# Comprimir logs antiguos
gzip *.log.1 *.log.2 *.log.3
```

---

## 🇺🇸 English

### Overview
This folder contains all log files from the AI prediction system. Each API and system component generates its own logs to facilitate monitoring, debugging, and performance analysis.

### 🗂️ File Structure

#### **Main API Logs**
- **`main_api.log`** - Main API and LLM coordinator
  - Records user queries, request routing, and Ollama communication
  - Includes response timing and error handling
  
- **`acv_api.log`** - Stroke risk prediction API
  - Decision Tree model loading
  - Medical predictions with input parameters and results
  - Generated health recommendations

- **`avocado_api.log`** - Avocado price prediction API
  - CatBoost model for price prediction
  - Feature engineering with temporal lags and rolling means
  - Market interpretations

- **`bitcoin_api.log`** - Bitcoin price prediction API
  - Prophet v2 model with real market data
  - Predictions with confidence intervals
  - Health checks and model status

- **`flights_api.log`** - Flight delay prediction API
  - Random Forest model for delays
  - Airline and airport mapping
  - Prediction timing

- **`movies_api.log`** - Movie recommendation API
  - KNN system with genre similarity
  - Personalized user recommendations
  - Rating predictions

#### **System Logs**
- **`uvicorn.log`** - Uvicorn web server
  - Server startup and shutdown events
  - Port and host configuration
  
- **`uvicorn_access.log`** - HTTP access logs
  - All HTTP requests with response codes
  - Client IPs and request timing
  
- **`fastapi.log`** - FastAPI framework
  - Middleware, validations, and framework errors
  - Route configuration and dependencies

### 🔧 Log Configuration

#### **Features**
- **Automatic rotation**: Maximum 5MB files, 5 backups per file
- **Standardized format**: `YYYY-MM-DD HH:MM:SS - LEVEL - LOGGER - MESSAGE`
- **Encoding**: UTF-8 for full character support
- **File-only**: No console output for clean interface

#### **Log Levels**
- **INFO**: Normal operations, successful loads, predictions
- **ERROR**: Model loading errors, prediction failures, exceptions
- **WARNING**: Anomalous situations that don't block execution
- **DEBUG**: Detailed information for development (disabled in production)

### 📊 Monitoring and Analysis

#### **Useful Commands**
```bash
# View logs in real time
tail -f main_api.log

# Search for errors in all logs
grep -r "ERROR" .

# View last 50 lines of specific API
tail -50 bitcoin_api.log

# Count successful predictions today
grep "$(date +%Y-%m-%d)" acv_api.log | grep "Prediction" | wc -l

# Monitor multiple APIs simultaneously
tail -f main_api.log bitcoin_api.log avocado_api.log
```

#### **Relevant Metrics**
- **Response time**: Look for patterns `Time: X.XXXs`
- **Loading errors**: `Model Load Failed`
- **Successful predictions**: `Prediction - Endpoint:`
- **Health checks**: `Health check solicitado`

### 🚨 Troubleshooting

#### **Common Issues**
1. **"Model Load Failed"** → Check model file path and permissions
2. **"Connection refused port 11434"** → Ollama is not running
3. **"HTTPException 500"** → Internal error, review complete stack trace
4. **Large log files** → Verify rotation configuration

#### **Log Cleanup**
```bash
# Remove old logs (careful!)
find . -name "*.log*" -mtime +30 -delete

# Compress old logs
gzip *.log.1 *.log.2 *.log.3
```

---

## 📝 Notes / Notas

### Configuration Location / Ubicación de Configuración
- **Logger config**: `api/config_logger.py`
- **Main API config**: `api/main.py`
- **Individual API configs**: Each file in `api/routes/`

### Log Retention / Retención de Logs
- **Default**: 5 files × 5MB = 25MB max per API
- **Total system**: ~200MB estimated
- **Recommendation**: Review monthly, archive important logs

### Performance Impact / Impacto en Rendimiento
- **Minimal**: Async file writing
- **No console output**: Reduces terminal noise
- **Rotation prevents**: Disk space issues

---

**Last Updated**: October 26, 2025  
**System Version**: Mariantonieta AI v1.0  
**Maintained by**: Development Team