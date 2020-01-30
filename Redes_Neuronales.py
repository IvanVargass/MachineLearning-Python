

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = load_iris()

x_ent, x_test, y_ent, y_test = train_test_split(iris.data, iris.target)

red = MLPClassifier(max_iter=1000, hidden_layer_sizes= (10000,10000))
red.fit(x_ent, y_ent)
print(red.score(x_test,y_test))