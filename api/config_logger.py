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


def log_model_loading(logger: logging.Logger, model_name: str, model_path: str, success: bool, error_msg: str = None):
    """
    Log estandarizado para carga de modelos
    
    Args:
        logger: Logger a usar
        model_name: Nombre del modelo
        model_path: Path del modelo
        success: Si la carga fue exitosa
        error_msg: Mensaje de error si falló
    """
    if success:
        logger.info(f"Model Loaded - Name: {model_name} | Path: {model_path}")
    else:
        logger.error(f"Model Load Failed - Name: {model_name} | Path: {model_path} | Error: {error_msg}")


def log_prediction(logger: logging.Logger, model_name: str, input_data: dict, prediction: any, confidence: float = None):
    """
    Log estandarizado para predicciones
    
    Args:
        logger: Logger a usar
        model_name: Nombre del modelo
        input_data: Datos de entrada
        prediction: Predicción realizada
        confidence: Nivel de confianza (opcional)
    """
    confidence_str = f" | Confidence: {confidence:.3f}" if confidence is not None else ""
    logger.info(f"Prediction - Model: {model_name} | Input: {input_data} | Result: {prediction}{confidence_str}")


# Configuración global de logging para FastAPI
def configure_fastapi_logging():
    """
    Configura el logging de FastAPI para que use nuestro sistema
    """
    # Configurar uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    
    # Usar nuestro formato
    for logger in [uvicorn_logger, uvicorn_access_logger]:
        if logger.handlers:
            for handler in logger.handlers:
                if hasattr(handler, 'setFormatter'):
                    formatter = logging.Formatter(
                        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                    )
                    handler.setFormatter(formatter)
