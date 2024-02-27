import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from metrics import mean_absolute_percentage_error, evaluate_model

def test_mean_absolute_percentage_error():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    epsilon = 1e-10
    expected_mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    calculated_mape = mean_absolute_percentage_error(y_true, y_pred)
    assert calculated_mape == pytest.approx(expected_mape)


def test_evaluate_model():
    # Crear un modelo de prueba y datos de prueba
    model = LinearRegression()
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model.fit(X, y)

    # Evaluar el modelo
    df_metrics = evaluate_model(model, X, y, 'LinearRegression')

    # Verificar que el DataFrame de métricas tiene las columnas correctas
    expected_columns = ['Model', 'MSE', 'MAPE', 'MAE', 'RMSE', 'R2']
    assert all(column in df_metrics.columns for column in expected_columns)

    # Verificar que el DataFrame de métricas tiene un valor para cada métrica
    assert not df_metrics.isnull().values.any()
