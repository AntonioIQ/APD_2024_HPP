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

    # Configurar el logger
    logging.basicConfig(filename=f'logs/{date_time}/{script_name}.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Crear un logger
    logger = logging.getLogger()
    return logger

