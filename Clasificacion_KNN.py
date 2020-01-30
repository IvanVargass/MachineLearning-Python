
# Librerias
import numpy as np 
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Algoritmo de clasificación de vecinos cercanos
from sklearn.neighbors import KNeighborsClassifier

# Se carga el datasets en iris
iris = load_iris()

# Se ve lo que contiene iris
print(iris.keys())

# Cada columna es una medición, cada fila es una flor
print(iris['data'])

# Salida de la red neuronal
print(iris['target'])

# Función para dividir datos en entrenamiento y validación
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'])

# Definimos parametros para clasificación, en este caso se consideran 7 vecinos
knn = KNeighborsClassifier(n_neighbors= 7)

# Se realiza el entrenamiento
knn.fit(x_train, y_train)

# Se valida el entramiento. Obtiene porcentaje de acierto.
print(knn.score(x_test, y_test))

# Validamos unas medidas en especificas para saber a cual pertenece
print(knn.predict([[1.2,3.4,5.6,1.1]]))

