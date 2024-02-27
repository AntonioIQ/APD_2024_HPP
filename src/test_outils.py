import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from outils import check_create_dir, load_csv_data, save_model, load_model, save_data, load_numpy_data, save_dataframe, load_dataframe, get_features

def test_check_create_dir():
    # Crear un directorio de prueba
    test_dir = "test_dir"
    check_create_dir(test_dir)

    # Verificar que el directorio se creó
    assert os.path.exists(test_dir)

    # Limpiar después de la prueba
    os.rmdir(test_dir)

def test_load_and_save_csv_data():
    # Crear algunos datos de prueba
    test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Guardar los datos de prueba
    test_path = "test_data.csv"
    test_data.to_csv(test_path, index=False)

    # Cargar los datos de prueba
    loaded_data = load_csv_data(test_path)

    # Verificar que los datos cargados son iguales a los datos originales
    assert test_data.equals(loaded_data)

    # Limpiar después de la prueba
    os.remove(test_path)

def test_load_and_save_model():
    # Crear un modelo de prueba
    test_model = LinearRegression()

    # Guardar el modelo de prueba
    test_path = "test_model.pkl"
    save_model(test_model, test_path)

    # Cargar el modelo de prueba
    loaded_model = load_model(test_path)

    # Verificar que el modelo cargado es igual al modelo original
    assert str(test_model) == str(loaded_model)

    # Limpiar después de la prueba
    os.remove(test_path)

def test_load_and_save_numpy_data():
    # Crear algunos datos de prueba
    test_data = np.array([1, 2, 3, 4, 5])

    # Guardar los datos de prueba
    test_path = "test_data.npy"
    save_data(test_data, test_path)

    # Cargar los datos de prueba
    loaded_data = load_numpy_data(test_path)

    # Verificar que los datos cargados son iguales a los datos originales
    assert np.array_equal(test_data, loaded_data)

    # Limpiar después de la prueba
    os.remove(test_path)

def test_load_and_save_dataframe():
    # Crear algunos datos de prueba
    test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Guardar los datos de prueba
    test_path = "test_data.csv"
    save_dataframe(test_data, test_path)

    # Cargar los datos de prueba
    loaded_data = load_dataframe(test_path)

    # Verificar que los datos cargados son iguales a los datos originales
    assert test_data.equals(loaded_data)

    # Limpiar después de la prueba
    os.remove(test_path)

def test_get_features():
    # Llamar a la función get_features
    features = get_features()

    # Verificar que la función devuelve una lista
    assert isinstance(features, list)

    # Verificar que la lista no está vacía
    assert len(features) > 0
