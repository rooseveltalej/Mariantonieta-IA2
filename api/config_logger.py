"""
Configuración centralizada de logging para todas las APIs
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional
from . import constants as const


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 5_000_000,  # 5 MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
    console_output: bool = False
) -> logging.Logger:
    """
    Configura y retorna un logger con rotación de archivos
    
    Args:
        name: Nombre del logger
        log_file: Nombre del archivo de log (opcional, se genera automáticamente si no se proporciona)
        level: Nivel de logging (default: INFO)
        max_bytes: Tamaño máximo del archivo antes de rotar (default: 5MB)
        backup_count: Número de archivos de backup a mantener (default: 5)
        format_string: Formato personalizado de log (opcional)
        console_output: Si también mostrar logs en consola (default: False)
    
    Returns:
        logging.Logger: Logger configurado
    """
    
    # Crear directorio de logs si no existe
    log_dir = os.path.join(const.BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Generar nombre de archivo si no se proporciona
    if log_file is None:
        log_file = f"{name.replace('.', '_')}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Formato por defecto
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    # Crear formatter
    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicar handlers si el logger ya existe
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=max_bytes, 
        backupCount=backup_count, 
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    # Handler para consola (opcional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger


def get_api_logger(api_name: str, console_output: bool = False) -> logging.Logger:
    """
    Obtiene un logger específico para una API
    
    Args:
        api_name: Nombre de la API (ej: "bitcoin_api", "movies_api")
        console_output: Si mostrar logs en consola también
    
    Returns:
        logging.Logger: Logger configurado para la API
    """
    return setup_logger(
        name=f"{api_name}_logger",
        log_file=f"{api_name}.log",
        console_output=console_output
    )


def get_coordinator_logger(console_output: bool = False) -> logging.Logger:
    """
    Obtiene un logger específico para el coordinador LLM
    
    Args:
        console_output: Si mostrar logs en consola también
    
    Returns:
        logging.Logger: Logger configurado para el coordinador
    """
    return setup_logger(
        name="llm_coordinator_logger",
        log_file="coordinator.log",
        console_output=console_output
    )


def get_main_logger(console_output: bool = False) -> logging.Logger:
    """
    Obtiene un logger para la aplicación principal
    
    Args:
        console_output: Si mostrar logs en consola también
    
    Returns:
        logging.Logger: Logger configurado para main
    """
    return setup_logger(
        name="main_api_logger",
        log_file="main_api.log",
        console_output=console_output
    )


def log_api_request(logger: logging.Logger, endpoint: str, request_data: dict, user_ip: str = "unknown"):
    """
    Log estandarizado para requests de API
    
    Args:
        logger: Logger a usar
        endpoint: Endpoint llamado
        request_data: Datos del request
        user_ip: IP del usuario
    """
    logger.info(f"API Request - Endpoint: {endpoint} | IP: {user_ip} | Data: {request_data}")


def log_api_response(logger: logging.Logger, endpoint: str, response_status: str, execution_time: float):
    """
    Log estandarizado para responses de API
    
    Args:
        logger: Logger a usar
        endpoint: Endpoint llamado
        response_status: Estado de la respuesta (success/error)
        execution_time: Tiempo de ejecución en segundos
    """
    logger.info(f"API Response - Endpoint: {endpoint} | Status: {response_status} | Time: {execution_time:.3f}s")


def log_model_loading(logger: logging.Logger, model_name_or_path: str, success_or_path: any = None, success: bool = None, error_msg: str = None, details: dict = None):
    """
    Log estandarizado para carga de modelos (compatible con múltiples formatos de llamada)
    
    Args:
        logger: Logger a usar
        model_name_or_path: Nombre del modelo o path (primer parámetro)
        success_or_path: Success bool o path (segundo parámetro, dependiendo del formato)
        success: Success bool (para formato de 4 parámetros)
        error_msg: Mensaje de error si falló
        details: Detalles adicionales (opcional)
    """
    # Determinar formato de llamada
    if success is not None:
        # Formato: (logger, model_name, model_path, success, error_msg)
        model_name = model_name_or_path
        model_path = success_or_path
        is_success = success
    else:
        # Formato: (logger, model_path, success, details)
        model_name = "Model"
        model_path = model_name_or_path
        is_success = success_or_path
    
    if is_success:
        details_str = f" | Details: {details}" if details else ""
        logger.info(f"Model Loaded - Name: {model_name} | Path: {model_path}{details_str}")
    else:
        error_str = error_msg or (details.get('error', 'Unknown error') if details else 'Unknown error')
        logger.error(f"Model Load Failed - Name: {model_name} | Path: {model_path} | Error: {error_str}")


def log_prediction(logger: logging.Logger, model_name_or_endpoint: str = None, input_data: dict = None, prediction: any = None, confidence: float = None, endpoint: str = None, input_payload: dict = None, output_summary: dict = None, **kwargs):
    """
    Log estandarizado para predicciones (compatible con múltiples formatos de llamada)
    
    Args:
        logger: Logger a usar
        model_name_or_endpoint: Nombre del modelo o endpoint
        input_data: Datos de entrada (formato original)
        prediction: Predicción realizada (formato original)
        confidence: Nivel de confianza (opcional)
        endpoint: Endpoint llamado (formato nuevo)
        input_payload: Payload de entrada (formato nuevo)
        output_summary: Resumen de salida (formato nuevo)
    """
    # Determinar formato de llamada
    if endpoint is not None:
        # Formato nuevo: endpoint, input_payload, output_summary
        log_endpoint = endpoint
        log_input = input_payload or {}
        log_output = output_summary or {}
        logger.info(f"Prediction - Endpoint: {log_endpoint} | Input: {log_input} | Output: {log_output}")
    else:
        # Formato original: model_name, input_data, prediction, confidence
        model_name = model_name_or_endpoint or "Unknown"
        confidence_str = f" | Confidence: {confidence:.3f}" if confidence is not None else ""
        logger.info(f"Prediction - Model: {model_name} | Input: {input_data} | Result: {prediction}{confidence_str}")


# Configuración global de logging para FastAPI
def configure_fastapi_logging():
    """
    Configura el logging de FastAPI para que use nuestro sistema y NO imprima en consola
    """
    # Configurar uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    fastapi_logger = logging.getLogger("fastapi")
    
    # Desactivar console output para todos los loggers de sistema
    for logger in [uvicorn_logger, uvicorn_access_logger, fastapi_logger]:
        # Limpiar handlers existentes
        logger.handlers.clear()
        
        # Crear nuestro file handler si no existe
        log_dir = os.path.join(const.BASE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{logger.name.replace('.', '_')}.log")
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=5_000_000, 
            backupCount=3, 
            encoding="utf-8"
        )
        
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Evitar que se propague a root logger


def disable_console_logging():
    """
    Desactiva completamente el logging a consola para todos los loggers
    """
    # Configurar root logger para no usar consola
    root_logger = logging.getLogger()
    root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)]
    
    # Configurar loggers específicos
    configure_fastapi_logging()
