"""
Este script realiza la inferencia de un modelo de Machine Learning
para calcular precios de casas.
"""

import argparse
import numpy as np
import logging
from datetime import datetime
from src.outils import get_features, load_model, load_dataframe, save_dataframe, check_create_dir

# Crear un timestamp para los nombres de los archivos de log
now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")

# Crear el directorio si no existe
check_create_dir(f'logs/{date_time}')

# Configurar el logger
logging.basicConfig(filename=f'logs/{date_time}/inference.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Crear un logger
logger = logging.getLogger()

# Crear el analizador
parser = argparse.ArgumentParser(description='Realizar inferencias con el modelo entrenado')

# Agregar argumentos con valores predeterminados
parser.add_argument('--predict_data', type=str, default='data/inference/test.csv', help='Ruta al archivo de datos para la inferencia')
parser.add_argument('--output', type=str, default='data/predictions/data_predicted.csv', help='Ruta al archivo de salida para las predicciones')

# Analizar los argumentos
args = parser.parse_args()

# Cargar el modelo entrenado
best_model = load_model('artifacts/best_model.pkl')

# Cargar el estado del imputer y del selector
imputer = load_model('data/prep/imputer.pkl')
selector = load_model('data/prep/selector.pkl')

# Definir las características
features = get_features()

# Cargar los datos de predicción
df_predict = load_dataframe(args.predict_data)

# Preparar los datos de predicción
X_predict = df_predict[features]
X_predict_imputed = imputer.transform(X_predict)
X_predict_selected = selector.transform(X_predict_imputed)

# Hacer predicciones con el modelo
try:
    predictions = best_model.predict(X_predict_selected)
    logger.info('Las predicciones se han realizado con éxito')
except Exception as e:
    logger.error(f'Error al realizar las predicciones: {e}', exc_info=True)

# Agregar una columna 'SalePrice' al DataFrame df_predict con las predicciones
try:
    df_predict['SalePrice'] = np.exp(predictions)
    logger.info('Las predicciones se han añadido al DataFrame con éxito')
except Exception as e:
    logger.error(f'Error al añadir las predicciones al DataFrame: {e}', exc_info=True)

# Guardar el DataFrame como un archivo CSV
try:
    save_dataframe(df_predict, args.output)
    logger.info('El DataFrame se ha guardado como un archivo CSV con éxito')
except Exception as e:
    logger.error(f'Error al guardar el DataFrame como un archivo CSV: {e}', exc_info=True)

logger.info('Se ha generado un archivo con la predicción de precios de casas, puede proceder a descargarlo de la carpeta data/predictions.')
