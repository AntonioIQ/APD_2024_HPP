from sklearn.model_selection import GridSearchCV
from src.outils import load_model, save_model, load_numpy_data
import time

# Registrar el tiempo de inicio
start_time = time.time()

# Cargar el modelo entrenado
model = load_model('artifacts/model.pkl')

#{'colsample_bytree': 0.6, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 120, 'reg_alpha': 0.05, 'reg_lambda': 0.8, 'subsample': 0.7}
#{'colsample_bytree': 0.6, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'subsample': 0.6}

# Definir los hiperparámetros y sus opciones para la búsqueda de cuadrícula
param_grid = {
    'n_estimators': [110, 120, 130, 140],  # número de árboles
    'max_depth': [4, 5, 6, 7],  # máxima profundidad
    'learning_rate': [0.05, 0.1, 0.15],  # tasa de aprendizaje
    'gamma': [0.00, 0.0001, 0.001],  # mínimo descenso de pérdida para hacer una partición
    'subsample': [0.7, 0.8, 0.9],  # proporción de muestras para cada árbol
    'colsample_bytree': [0.55, 0.6, 0.65],  # proporción de columnas para cada árbol
    'reg_lambda': [0.6, 0.8, 1.0],  # término de regularización L2
    'reg_alpha': [0.01, 0.05, 0.1]  # término de regularización L1
}

# Cargar los datos de entrenamiento
X_train_selected = load_numpy_data('data/prep/X_train_selected.npy')
y_train = load_numpy_data('data/prep/y_train.npy')

# Ajuste de hiperparámetros
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_selected, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Imprimir los mejores hiperparámetros
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Guardar el mejor modelo
save_model(best_model, 'artifacts/best_model.pkl')

# Registrar el tiempo de finalización
end_time = time.time()

# Calcular e imprimir el tiempo de ejecución
execution_time = end_time - start_time
print(f"El script se ejecutó en: {execution_time} segundos")