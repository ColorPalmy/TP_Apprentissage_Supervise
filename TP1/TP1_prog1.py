# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784')

#affiche tous les attributs du jeu de données
print(mnist)
#affiche le taleau contenant les données: 
#une ligne = une instance = représentation des valeurs des pixels d'une image
#une colonne = valeur d'un pixel a une certaine position 
print (mnist.data)
#affiche les étiquettes des instances, ici les digits représentés par les images
print (mnist.target)
#affiche le nombre de lignes du tableau de données = nombre d'instances dans le dataset
len(mnist.data)
#affiche la documentation de la fonction len
help(len)
#affiche le nombre de lignes et colonnes du tableau de données = (nombre d'instances, nombre d'attributs ici de pixels)
print (mnist.data.shape)
#affiche la forme du tableau des étiquettes ici le nombre d'instances
print (mnist.target.shape)
#affiche le tableau représentant la valeur des pixels de la première instance/image
mnist.data[0]
#affiche la valeur du second pixel de la première image
mnist.data[0][1]
#affiche la valeur du second pixel de toutes les images = la seconde colonne du tableau de données
mnist.data[:,1]
#affiche les 100 premières instances/images
mnist.data[:100]

#modifie le tableau de données en conservant les 70000 lignes (grâce au -1) et transforme les tableaux (lignes) de 784 valeurs (images) en tableaux de 28 lignes * 28 colonnes représentant les valeurs des pixels des images 
images = mnist.data.reshape((-1, 28, 28))
#affiche le premier tableau de pixel (instance) sous forme d'image en nuance de gris
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
#classe de l'image affichée
print(mnist.target[0])