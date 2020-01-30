

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

boston = load_boston()

print(boston.keys())
print(boston.data.shape)
# Vecinos cercanos con regresión lineal
x_ent, x_test, y_ent, y_test = train_test_split(boston.data, boston.target)
knn = KNeighborsRegressor(n_neighbors= 3)
knn.fit(x_ent, y_ent)
print(knn.score(x_test, y_test))

# Regresión lineal
rl = LinearRegression()
rl.fit(x_ent, y_ent)
print(rl.score(x_test, y_test))

# Algoritmo de ridge

rd = Ridge()
rd.fit(x_ent, y_ent)
print(rd.score(x_test, y_test))