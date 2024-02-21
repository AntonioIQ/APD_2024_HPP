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

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

from src.outils import check_create_dir, load_csv_data
from src.outils import save_model, save_data, get_features

# Verificar si la carpeta data/raw/ existe, y si no, crearla
check_create_dir('data/raw/')

# Cargar los datos de entrenamiento
df_train = load_csv_data('data/raw/train.csv')

# Ingeniería de características
features = get_features()

# Transformar las variables sesgadas
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# Normalizar o estandarizar las variables
scaler = StandardScaler()
df_train[features] = scaler.fit_transform(df_train[features])

# Guardar el estado del scaler
save_model(scaler, 'data/prep/scaler.pkl')

# Definición de variables para el entrenamiento
X = df_train[features]
y = df_train['SalePrice']

# Conjunto de entrenamiento (80% de los datos) y un temporal (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Temporal en conjuntos de validación y prueba (cada uno con 10% )
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Crear un imputador con estrategia de reemplazo por la media
imputer = SimpleImputer(strategy='mean')

# Ajustar el imputador de entrenamiento y transformar
X_train_imputed = imputer.fit_transform(X_train)

# Guardar el estado del imputer
save_model(imputer, 'data/prep/imputer.pkl')

# Selección automática de características
selector = RFECV(LinearRegression(), step=1, cv=5)
X_train_selected = selector.fit_transform(X_train_imputed, y_train)

# Guardar el estado del selector
save_model(selector, 'data/prep/selector.pkl')

# Guardar los datos preprocesados
save_data(X_train_selected, 'data/prep/X_train_selected.npy')
save_data(y_train, 'data/prep/y_train.npy')

# Guardar los conjuntos de validación y prueba
save_data(X_val, 'data/prep/X_val.npy')
save_data(y_val, 'data/prep/y_val.npy')
save_data(X_test, 'data/prep/X_test.npy')
save_data(y_test, 'data/prep/y_test.npy')

print('El preprocesamiento de datos se ha llevado con exito, '
      'puede proceder a entrenar el modelo.')
