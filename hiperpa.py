from sklearn.model_selection import GridSearchCV
from src.outils import load_model, save_model, load_numpy_data

# Cargar el modelo entrenado
model = load_model('artifacts/model.pkl')

# Definir los hiperparámetros y sus opciones para la búsqueda de cuadrícula
param_grid = {
    'n_estimators': [50, 100, 150, 200],  # número de árboles
    'max_depth': [3, 5, 7, 9],  # máxima profundidad
    'learning_rate': [0.01, 0.1, 0.2],  # tasa de aprendizaje
    'gamma': [0.0, 0.1, 0.2],  # mínimo descenso de pérdida para hacer una partición
    'subsample': [0.6, 0.8, 1.0],  # proporción de muestras para cada árbol
    'colsample_bytree': [0.6, 0.8, 1.0],  # proporción de columnas para cada árbol
    'reg_lambda': [0.5, 1.0, 1.5],  # término de regularización L2
    'reg_alpha': [0.0, 0.1, 0.2]  # término de regularización L1
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
