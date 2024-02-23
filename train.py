
"""
Este script realiza el entrenamiento de un modelo
de Machine Learning para calcular precios de casas.
"""

# Importar las librerías necesarias
import pandas as pd
from xgboost import XGBRegressor

# Importar las funciones necesarias desde outils.py y metrics.py
from src.outils import load_model, load_numpy_data, save_model, save_dataframe
from src.metrics import evaluate_model

import yaml

# Cargar los hiperparámetros desde el archivo de configuración
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

hyperparameters = config['best_model']['hyperparameters']
seed = config['best_model']['seed']

# Cargar los datos preprocesados
X_train_selected = load_numpy_data('data/prep/X_train_selected.npy')
y_train = load_numpy_data('data/prep/y_train.npy')

# Crear y entrenar el modelo XGBoost
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

# Guardar el modelo entrenado
save_model(model, 'artifacts/model.pkl')

# Evaluar el modelo en los datos de entrenamiento y guardar las métricas
df_metrics = evaluate_model(model, X_train_selected, y_train, 'XGBoost_Training')
save_dataframe(df_metrics, 'artifacts/evaluation.txt')

# Cargar los datos de validación y prueba
X_val = load_numpy_data('data/prep/X_val.npy')
y_val = load_numpy_data('data/prep/y_val.npy')
X_test = load_numpy_data('data/prep/X_test.npy')
y_test = load_numpy_data('data/prep/y_test.npy')

# Cargar el selector de características
selector = load_model('data/prep/selector.pkl')

# Aplicar el selector de características a los datos de validación y prueba
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Evaluar el modelo en los datos de validación y añadir las métricas al DataFrame
df_metrics_val = evaluate_model(model, X_val_selected, y_val, 'XGBoost_Validation')
df_metrics = pd.concat([df_metrics, df_metrics_val])

# Evaluar el modelo en los datos de prueba y añadir las métricas al DataFrame
df_metrics_test = evaluate_model(model, X_test_selected, y_test, 'XGBoost_Test')
df_metrics = pd.concat([df_metrics, df_metrics_test])

# Guardar el DataFrame de métricas actualizado como un archivo .txt en la carpeta artifacts
save_dataframe(df_metrics, 'artifacts/evaluation.txt')

print('El entrenamiento del modelo se ha realizado con exito. La métrica evaluada se '
      'ubica en la carpeta artifacts con el nombre de evaluation.txt, puede proceder a '
      'generar predicciones.')
