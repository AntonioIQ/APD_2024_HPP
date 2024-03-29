# Machine Learning Architecture
## Predicción de precios de casas
#### José Antonio Tapia Godínez

## Estructura del Proyecto

El proyecto consta de varios scripts de Python y carpetas:


|____artifacts  
|____data  
| \____inference  
| | \____test.csv  
| \____predictions  
| | \____data_predicted.csv  
| \____prep  
| \____raw  
| | \____train.csv  
|____inference.py  
|____notebooks  
| |____EDA.ipynb  
| |____Prototipo.ipynb  
|____prep.py  
|____src  
| |____metrics.py  
| |____outils.py  
|____train.py  


artifacts/: Esta carpeta contiene los modelos entrenados y las métricas de evaluación.

data/: Esta carpeta contiene los datos utilizados para el entrenamiento y la validación del modelo.

inference/: Contiene los datos que se utilizarán para hacer inferencias con el modelo entrenado.

test.csv: Los datos de prueba para hacer inferencias.

predictions/: Contiene las predicciones generadas por el modelo.

data_predicted.csv: Las predicciones generadas por el modelo.

prep/: Contiene los datos preprocesados que se utilizarán para entrenar el modelo.

raw/: Contiene los datos sin procesar.

train.csv: Los datos de entrenamiento sin procesar.

inference.py: Este script utiliza el modelo entrenado para hacer inferencias en nuevos datos.

notebooks/: Esta carpeta contiene los cuadernos Jupyter utilizados para el análisis exploratorio de datos y la prototipificación.

EDA.ipynb: Cuaderno Jupyter para el análisis exploratorio de datos.

Prototipo.ipynb: Cuaderno Jupyter para la prototipificación del modelo.

prep.py: Este script realiza el preprocesamiento de los datos para el entrenamiento del modelo.

src/: Esta carpeta contiene los scripts de Python que definen funciones útiles para el preprocesamiento de datos, el entrenamiento del modelo y la inferencia.

metrics.py: Define funciones para calcular el error porcentual absoluto medio (MAPE) y para evaluar un modelo.

outils.py: Define funciones para cargar y guardar datos y modelos, y para verificar y crear directorios.

train.py: Este script carga los datos preprocesados, entrena un modelo XGBoost, evalúa el modelo y guarda tanto el modelo como las métricas de evaluación.

## Arquitectura
![Arquitectura del Producto de Datos](Images/Arquitectura.png)



## Cómo usar

1. Asegúrate de tener instaladas todas las librerías necesarias. Puedes instalarlas mediante el método pip


2. Ejecuta el script prep.py para preprocesar los datos:
python prep.py

3. Ejecuta el script train.py para entrenar el modelo y evaluarlo:
python train.py

3. Usar el modelo entrenado para hacer predicciones con el script inference.py:
python inference.py

## Resultados
El entrenamiento del modelo se realiza con éxito. La métrica evaluada se ubica en la carpeta artifacts con el nombre de evaluation.txt. Puedes proceder a generar predicciones.



PS C:\Users\antap\OneDrive\Documents\M CD\Arquitectura del Producto de Datos\Casas_Price_Predict\APD_2024_HPP\src> python -m pytest test_outils.py
============================================================================================== test session starts ==============================================================================================
platform win32 -- Python 3.9.0, pytest-8.0.2, pluggy-1.4.0
rootdir: C:\Users\antap\OneDrive\Documents\M CD\Arquitectura del Producto de Datos\Casas_Price_Predict\APD_2024_HPP\src
collected 6 items

test_outils.py ......                                                                                                                                                                                      [100%]

=============================================================================================== 6 passed in 1.15s ===============================================================================================

S C:\Users\antap\OneDrive\Documents\M CD\Arquitectura del Producto de Datos\Casas_Price_Predict\APD_2024_HPP\src> python -m pytest test_metrics.py
============================================================================================== test session starts ==============================================================================================
platform win32 -- Python 3.9.0, pytest-8.0.2, pluggy-1.4.0
rootdir: C:\Users\antap\OneDrive\Documents\M CD\Arquitectura del Producto de Datos\Casas_Price_Predict\APD_2024_HPP\src
collected 2 items

test_metrics.py ..                                                                                                                                                                                         [100%] 

=============================================================================================== 2 passed in 0.63s ===============================================================================================
S C:\Users\antap\OneDrive\Documents\M CD\Arquitectura del Producto de Datos\Casas_Price_Predict\APD_2024_HPP\src> python -m pytest test_logs.py
============================================================================================== test session starts ==============================================================================================
platform win32 -- Python 3.9.0, pytest-8.0.2, pluggy-1.4.0
rootdir: C:\Users\antap\OneDrive\Documents\M CD\Arquitectura del Producto de Datos\Casas_Price_Predict\APD_2024_HPP\src
collected 1 item                                                                                                                                                                                                  

test_logs.py .                                                                                                                                                                                             [100%]

=============================================================================================== 1 passed in 0.02s ===============================================================================================