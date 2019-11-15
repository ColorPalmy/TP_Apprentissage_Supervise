# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.neighbors as neighbors
import numpy as np

mnist = fetch_openml('mnist_784')

#tableau de 5000 valeurs aléatoires entre à et 70000
index = np.random.randint(70000, size=5000)
#échantillon de 5000 données de Mnist
data = mnist.data[index]
#étiquettes correspondant aux 5000 données échantillons
target = mnist.target[index]
#division de la base en données d'apprentissage (80%) et de test (20%)
datatrain, datatest, targettrain, targettest = train_test_split(data, target,train_size=0.80)
#définition du nombre de voisins
k = 10
#création de l'objet classifier à k voisins
clf = neighbors.KNeighborsClassifier(k)
#apprentissage avec les données
clf.fit(datatrain,targettrain)
#classe de l'image 4
print(targettest[3])
#classe prédite de l'image 4
clf.predict(datatest)[3]
#score sur l'échantillon de test
clf.score(datatest,targettest)
#taux d'erreur sur données d'apprentissage
#tend vers 0 car on l'a entrainé sur ces données mais pas 0 car certaines données peuvent être proches de plusieurs classes
1 - clf.score(datatrain,targettrain)

#score pour des k de 2 a 15 avec 90% apprentissage (nb_splits=10) / 80% apprentissage(nb_splits=5)
nb_splits = 5
kf = KFold(n_splits=nb_splits,shuffle=True)
mean_scores = []
for k in range(2,16):
    clf = neighbors.KNeighborsClassifier(k)
    score_sum = 0
    for indextrain, indextest in kf.split(data):
        xtrain, xtest, ytrain, ytest = data[indextrain], data[indextest], target[indextrain], target[indextest]
        clf.fit(xtrain,ytrain)
        score = clf.score(xtest,ytest)
        score_sum = score_sum + score
    mean_k = score_sum/nb_splits
    mean_scores.append(mean_k)
    print("k = ", k, "score moyen = ", mean_k)
    
#on fixe k avec l'optimal trouvé (peut faire automatiquement)
#k = mean_scores.index(max(mean_scores)) + 2
k = 3
clf = neighbors.KNeighborsClassifier(k)
#score pour pourcentages variés
scores = []
for pourc in np.arange(0.7,0.9,0.05):
    xtrain, xtest, ytrain, ytest = train_test_split(data, target,train_size=pourc)
    clf.fit(xtrain,ytrain)
    score = clf.score(xtest,ytest)
    scores.append(score)
    print("pourcentage = ", pourc * 100, "% score = ", score)
        
# on fixe le pourcentage à 90%
# pourcentage = scores.index(max(scores)) * 0.05 + 0.7
pourcentage = 0.9
#variation de la taille de l'échantillon
for s in range(2000,62001,5000):
    index = np.random.randint(70000, size=s)
    data = mnist.data[index]
    target = mnist.target[index]
    xtrain, xtest, ytrain, ytest = train_test_split(data, target,train_size=pourcentage)
    clf.fit(xtrain,ytrain)
    score = clf.score(xtest,ytest)
    print("taille = ", s, "score = ", score)

