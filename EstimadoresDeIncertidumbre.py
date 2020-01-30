
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm

iris = load_iris()
x_ent, x_test, y_ent, y_test = train_test_split(iris.data, iris.target)

algoritmo = svm.SVC(probability=True)

# Se realiza entrenamiento con maquina vectorial de soporte
algoritmo.fit(x_ent, y_ent)
print(algoritmo.score(x_test, y_test))

algoritmo.decision_function_shape = "ovr"
print(algoritmo.decision_function(x_test)[:10])

# Que tan seguro esta el algoritmo
print(algoritmo.predict_proba(x_test)[:10])

print(algoritmo.predict(x_test)[:1])
