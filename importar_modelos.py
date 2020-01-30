
from sklearn.externals import joblib
from sklearn.datasets import load_iris

clf = joblib.load("Modelo_entrenado.pkl")
iris = load_iris()
print(clf.score(iris.data, iris.target))