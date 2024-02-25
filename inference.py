"""
Este script realiza la inferencia de un modelo de Machine Learning
para calcular precios de casas.
"""

import argparse
import numpy as np
from src.outils import get_features, load_model, load_dataframe, save_dataframe

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
predictions = best_model.predict(X_predict_selected)

# Agregar una columna 'SalePrice' al DataFrame df_predict con las predicciones
df_predict['SalePrice'] = np.exp(predictions)

# Guardar el DataFrame como un archivo CSV
save_dataframe(df_predict, args.output)

print('Se ha generado un archivo con la predicción de precios de casas, '
      'puede proceder a descargarlo de la carpeta data/predictions.')

