
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

iris = datasets.load_iris()

print(iris.keys())

xe, xt, ye, yt = train_test_split(iris.data, iris.target)

clf = linear_model.LogisticRegression()

clf.fit(xe, ye)
print(clf.score(xt,yt))

# Guardar el modelo

joblib.dump(clf, "Modelo_entrenado.pkl")