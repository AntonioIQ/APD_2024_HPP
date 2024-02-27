# src/logs.py

import logging
from datetime import datetime
import os

def configure_logger(script_name):
    # Crear un timestamp para los nombres de los archivos de log
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")

    # Crear los directorios necesarios
    os.makedirs(f'logs/{date_time}', exist_ok=True)

    # Crear un logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Eliminar todos los manejadores existentes del logger
    logger.handlers = []

    # Añadir un FileHandler al logger
    file_handler = logging.FileHandler(filename=f'logs/{date_time}/{script_name}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
