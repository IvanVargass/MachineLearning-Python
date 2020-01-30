

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

noticias = fetch_20newsgroups(subset="train")
print(noticias.data[5])
# Tamaño de la data (Número de textos)
print(len(noticias.data))
# Como se dividen 
print(noticias.target_names)

vector = CountVectorizer()

vector.fit(noticias.data)
# Se visualizan todos los token que se crearon
print(vector.vocabulary_)
# Se crea la matriz
bolsa = vector.transform(noticias.data)
print(bolsa.shape)

# Salida del algoritmo
bolsay = noticias.target
xe, xt, ye, yt = train_test_split(bolsa, bolsay)
lr = LogisticRegression()
lr.fit(xe,ye)
print(lr.score(xt,yt))