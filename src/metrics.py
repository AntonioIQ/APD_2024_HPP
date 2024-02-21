# src/metrics.py

"""
Este módulo contiene funciones útiles para la evaluación de modelos.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calcula el error porcentual absoluto medio (MAPE).
    
    El MAPE es una medida de precisión de un método de pronóstico en estadísticas,
    por ejemplo, en la predicción de tiempo o en la predicción de ventas. 
    Como tal, es una cantidad que es negativa cuando la predicción supera la realidad 
    y positiva en el caso contrario.

    Parámetros:
    y_true (array-like): Valores verdaderos para y.
    y_pred (array-like): Valores predichos para y.

    Retorna:
    float: El error porcentual absoluto medio (MAPE).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(model, X, y, model_name):
    """
    Evalúa un modelo en los datos proporcionados y devuelve un DataFrame con las métricas de evaluación.

    Parámetros:
    model (sklearn estimator): El modelo para evaluar.
    X (array-like): Los datos de entrada para el modelo.
    y (array-like): Los datos de salida verdaderos.
    model_name (str): El nombre del modelo para usar en el DataFrame de métricas.

    Retorna:
    df_metrics (pandas DataFrame): Un DataFrame con las métricas de evaluación.
    """
    # Hacer predicciones con el modelo
    predictions = model.predict(X)

    # Calcular las métricas de evaluación
    mse = mean_squared_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    # Crear un DataFrame para las métricas
    df_metrics = pd.DataFrame({
        'Model': [model_name],
        'MSE': [mse],
        'MAPE': [mape],
        'MAE': [mae],
        'RMSE': [rmse],
        'R2': [r2]})

    return df_metrics
