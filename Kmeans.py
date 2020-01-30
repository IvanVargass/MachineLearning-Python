
# Libreria para clasificación
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
# Libreria para ver que tan bien aprende el algoritmo
from sklearn import metrics
import numpy as np

iris = load_iris()

# Entrada 
x = iris.data
print(x)

# Salida
y = iris.target

km = KMeans(n_clusters=3, max_iter=3000)
km.fit(x)

predicion = km.predict(x)
print(predicion)

score = metrics.adjusted_rand_score(y, predicion)
print(score)