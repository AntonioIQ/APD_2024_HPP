
"""
Este script realiza el entrenamiento de un modelo
de Machine Learning para calcular precios de casas.
"""

import argparse
import pandas as pd
from xgboost import XGBRegressor
from src.outils import load_model, load_numpy_data, save_model, save_dataframe
from src.metrics import evaluate_model
import yaml
import logging
from datetime import datetime

from src.outils import check_create_dir

# Crear un timestamp para los nombres de los archivos de log
now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")

# Crear el directorio si no existe
check_create_dir(f'logs/{date_time}')

# Configurar el logger
logging.basicConfig(filename=f'logs/{date_time}/train.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
# Crear un logger
logger = logging.getLogger()

# Crear el analizador
parser = argparse.ArgumentParser(description='Entrenar un modelo de Machine Learning')

# Agregar argumentos con valores predeterminados
parser.add_argument('--train_data', type=str, default='data/prep/X_train_selected.npy', help='Ruta al archivo de datos de entrenamiento')
parser.add_argument('--val_data', type=str, default='data/prep/X_val.npy', help='Ruta al archivo de datos de validación')
parser.add_argument('--test_data', type=str, default='data/prep/X_test.npy', help='Ruta al archivo de datos de prueba')
parser.add_argument('--model', type=str, default='artifacts/model.pkl', help='Ruta al archivo del modelo')
parser.add_argument('--test_size', type=float, default=0.2, help='Proporción del conjunto de datos a incluir en la división de prueba')

# Analizar los argumentos
args = parser.parse_args()

# Cargar los hiperparámetros desde el archivo de configuración
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

hyperparameters = config['best_model']['hyperparameters']
seed = config['best_model']['seed']

# Cargar los datos preprocesados
X_train_selected = load_numpy_data(args.train_data)
y_train = load_numpy_data('data/prep/y_train.npy')

# Crear y entrenar el modelo XGBoost
try:
    model = XGBRegressor(n_estimators=hyperparameters['n_estimators'],
                         max_depth=hyperparameters['max_depth'],
                         learning_rate=hyperparameters['learning_rate'],
                         gamma=hyperparameters['gamma'],
                         subsample=hyperparameters['subsample'],
                         colsample_bytree=hyperparameters['colsample_bytree'],
                         reg_lambda=hyperparameters['reg_lambda'],
                         reg_alpha=hyperparameters['reg_alpha'],
                         random_state=seed)
    model.fit(X_train_selected, y_train)
    logger.info('El modelo XGBoost se ha entrenado exitosamente')
except Exception as e:
    logger.error(f'Error al entrenar el modelo XGBoost: {e}', exc_info=True)

# Guardar el modelo entrenado
try:
    save_model(model, args.model)
    logger.info('El modelo entrenado se ha guardado exitosamente')
except Exception as e:
    logger.error(f'Error al guardar el modelo entrenado: {e}', exc_info=True)

# Evaluar el modelo en los datos de entrenamiento y guardar las métricas
try:
    df_metrics = evaluate_model(model, X_train_selected, y_train, 'XGBoost_Training')
    save_dataframe(df_metrics, 'artifacts/evaluation.txt')
    logger.info('Las métricas de evaluación del modelo se han guardado exitosamente')
except Exception as e:
    logger.error(f'Error al evaluar el modelo y guardar las métricas: {e}', exc_info=True)

# Cargar los datos de validación y prueba
X_val = load_numpy_data(args.val_data)
y_val = load_numpy_data('data/prep/y_val.npy')
X_test = load_numpy_data(args.test_data)
y_test = load_numpy_data('data/prep/y_test.npy')

# Cargar el selector de características
selector = load_model('data/prep/selector.pkl')

# Aplicar el selector de características a los datos de validación y prueba
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Evaluar el modelo en los datos de validación y añadir las métricas al DataFrame
try:
    df_metrics_val = evaluate_model(model, X_val_selected, y_val, 'XGBoost_Validation')
    df_metrics = pd.concat([df_metrics, df_metrics_val])
    logger.info('El modelo se ha evaluado en los datos de validación y las métricas se han añadido al DataFrame')
except Exception as e:
    logger.error(f'Error al evaluar el modelo en los datos de validación: {e}', exc_info=True)

# Evaluar el modelo en los datos de prueba y añadir las métricas al DataFrame
try:
    df_metrics_test = evaluate_model(model, X_test_selected, y_test, 'XGBoost_Test')
    df_metrics = pd.concat([df_metrics, df_metrics_test])
    logger.info('El modelo se ha evaluado en los datos de prueba y las métricas se han añadido al DataFrame')
except Exception as e:
    logger.error(f'Error al evaluar el modelo en los datos de prueba: {e}', exc_info=True)

# Guardar el DataFrame de métricas actualizado como un archivo .txt en la carpeta artifacts
try:
    save_dataframe(df_metrics, 'artifacts/evaluation.txt')
    logger.info('El DataFrame de métricas actualizado se ha guardado como un archivo .txt en la carpeta artifacts')
except Exception as e:
    logger.error(f'Error al guardar el DataFrame de métricas: {e}', exc_info=True)

logger.info('El entrenamiento del modelo se ha realizado con éxito. La métrica evaluada se ubica en la carpeta artifacts con el nombre de evaluation.txt, puede proceder a generar predicciones.')

