"""
Este script realiza el preprocesamiento de los datos para
el entrenamiento de un modelo de Machine Learning que tiene
como objetivo calcular precios de casas.
Se requieren las siguientes librerías: numpy, sklearn,
Se requieren los siguientes archivos de datos: data/train.csv, data/test.csv
Estos archivos se pueden descargar de la página de Kaggle:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
NOTA IMPORTANTE: Para descargarlos, se necesita tener una cuenta de Kaggle
"""

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
import logging
from datetime import datetime

from src.outils import check_create_dir, load_csv_data
from src.outils import save_model, save_data, get_features
from src import logs
from src.outils import check_create_dir

# Crear un timestamp para los nombres de los archivos de log
now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")

# Crear el directorio si no existe
check_create_dir(f'logs/{date_time}')

# Configurar el logger
logging.basicConfig(filename=f'logs/{date_time}/prep.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Crear un logger
logger = logging.getLogger()

# Crear el analizador
parser = argparse.ArgumentParser(description='Preprocesar los datos para el entrenamiento del modelo')

# Agregar argumentos con valores predeterminados
parser.add_argument('--raw_data', type=str, default='data/raw/train.csv', help='Ruta al archivo de datos sin procesar')
parser.add_argument('--test_size', type=float, default=0.2, help='Proporción del conjunto de datos a incluir en la división de prueba')

# Analizar los argumentos
args = parser.parse_args()

# Verificar si la carpeta data/raw/ existe, y si no, crearla
check_create_dir('data/raw/')

# Cargar los datos de entrenamiento
try:
    # Cargar los datos de entrenamiento
    df_train = load_csv_data(args.raw_data)
    logger.info('Datos cargados exitosamente')
    logger.debug(f'Número de filas: {df_train.shape[0]}, número de columnas: {df_train.shape[1]}')
except Exception as e:
    logger.error(f'Error al cargar los datos: {e}', exc_info=True)

# Ingeniería de características
try:
    features = get_features()
    logger.info('Ingeniería de características realizada exitosamente')
    logger.debug(f'Características: {features}')
except Exception as e:
    logger.error(f'Error al realizar la ingeniería de características: {e}', exc_info=True)

# Transformar las variables sesgadas
try:
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    logger.info('Transformación de la variable sesgada realizada exitosamente')
    logger.debug(f'Primeras 5 filas de "SalePrice" transformado: {df_train["SalePrice"].head()}')
except Exception as e:
    logger.error(f'Error al transformar la variable sesgada: {e}', exc_info=True)

# Normalizar o estandarizar las variables
try:
    scaler = StandardScaler()
    df_train[features] = scaler.fit_transform(df_train[features])
    logger.info('Normalización de las variables realizada exitosamente')
    logger.debug(f'Primeras 5 filas de las variables normalizadas: {df_train[features].head()}')
except Exception as e:
    logger.error(f'Error al normalizar las variables: {e}', exc_info=True)


# Guardar el estado del scaler
try:
    save_model(scaler, 'data/prep/scaler.pkl')
    logger.info('El estado del scaler se ha guardado exitosamente')
except Exception as e:
    logger.error(f'Error al guardar el estado del scaler: {e}', exc_info=True)


# Definición de variables para el entrenamiento
X = df_train[features]
y = df_train['SalePrice']

# Conjunto de entrenamiento (80% de los datos) y un temporal (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42)

# Temporal en conjuntos de validación y prueba (cada uno con 10% )
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Crear un imputador con estrategia de reemplazo por la media
imputer = SimpleImputer(strategy='mean')

# Ajustar el imputador de entrenamiento y transformar
X_train_imputed = imputer.fit_transform(X_train)

# Guardar el estado del imputer
try:
    save_model(imputer, 'data/prep/imputer.pkl')
    logger.info('El estado del imputer se ha guardado exitosamente')
except Exception as e:
    logger.error(f'Error al guardar el estado del imputer: {e}', exc_info=True)
    


save_model(imputer, 'data/prep/imputer.pkl')

# Selección automática de características
selector = RFECV(LinearRegression(), step=1, cv=5)
X_train_selected = selector.fit_transform(X_train_imputed, y_train)

# Guardar el estado del selector
save_model(selector, 'data/prep/selector.pkl')

# Guardar los datos preprocesados
try:
    save_data(X_train_selected, 'data/prep/X_train_selected.npy')
    save_data(y_train, 'data/prep/y_train.npy')
    logger.info('Los datos preprocesados se han guardado exitosamente')
except Exception as e:
    logger.error(f'Error al guardar los datos preprocesados: {e}', exc_info=True)


# Guardar los conjuntos de validación y prueba
save_data(X_val, 'data/prep/X_val.npy')
save_data(y_val, 'data/prep/y_val.npy')
save_data(X_test, 'data/prep/X_test.npy')
save_data(y_test, 'data/prep/y_test.npy')

logger.info('El preprocesamiento de datos se ha llevado con éxito, puede proceder a entrenar el modelo.')
