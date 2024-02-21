# src/outils.py

"""
Este módulo contiene funciones útiles para el preprocesamiento de datos, el entrenamiento del modelo y la inferencia.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def check_create_dir(dir_path):
    """
    Verifica si un directorio existe, y si no, lo crea.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_csv_data(file_path):
    """
    Carga los datos desde un archivo CSV.
    """
    return pd.read_csv(file_path)



def load_numpy_data(path):
    """
    Carga los datos desde un archivo binario de NumPy.
    """
    return np.load(path)

def save_model(model, path):
    """
    Guarda un modelo entrenado en el path especificado.
    """
    joblib.dump(model, path)

def load_model(path):
    """
    Carga un modelo entrenado desde el path especificado.
    """
    return joblib.load(path)

def save_data(data, path):
    """
    Guarda los datos en el path especificado.
    """
    np.save(path, data)


def save_dataframe(df, path):
    """
    Guarda un DataFrame en el path especificado.
    """
    df.to_csv(path, index=False)

def load_dataframe(path):
    """
    Carga un DataFrame desde el path especificado.
    """
    return pd.read_csv(path)


def get_features():
    """
    Devuelve la lista de características seleccionadas del EDA a usar en el modelo.

    Retorna:
    list: La lista de características.
    """
    features = ['LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr',
                'BsmtHalfBath', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    return features