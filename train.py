
"""
Este script realiza el entrenamiento de un modelo
de Machine Learning para calcular precios de casas.
"""

# Importar las librerías necesarias

import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Importar las funciones necesarias desde outils.py y metrics.py
from src.outils import load_model, load_numpy_data, save_model, save_dataframe
from src.metrics import evaluate_model

# Cargar los datos preprocesados
X_train_selected = load_numpy_data('data/prep/X_train_selected.npy')
y_train = load_numpy_data('data/prep/y_train.npy')

# Crear y entrenar el modelo XGBoost
model = XGBRegressor()

# Ajuste de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_selected, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Guardar el modelo entrenado
save_model(best_model, 'artifacts/best_model.pkl')

# Evaluar el modelo en los datos de entrenamiento y guardar las métricas
df_metrics = evaluate_model(best_model, X_train_selected, y_train, 'XGBoost_Training')
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
df_metrics_val = evaluate_model(best_model, X_val_selected, y_val, 'XGBoost_Validation')
df_metrics = pd.concat([df_metrics, df_metrics_val])

# Evaluar el modelo en los datos de prueba y añadir las métricas al DataFrame
df_metrics_test = evaluate_model(best_model, X_test_selected, y_test, 'XGBoost_Test')
df_metrics = pd.concat([df_metrics, df_metrics_test])

# Guardar el DataFrame de métricas actualizado como un archivo .txt en la carpeta artifacts
save_dataframe(df_metrics, 'artifacts/evaluation.txt')

print('El entrenamiento del modelo se ha realizado con exito. La métrica evaluada se '
      'ubica en la carpeta artifacts con el nombre de evaluation.txt, puede proceder a '
      'generar predicciones.')
