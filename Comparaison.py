#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sklearn.neighbors as neighbors
from sklearn.metrics import precision_score, recall_score, zero_one_loss, confusion_matrix
import sklearn.neural_network as nn
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import time


# In[2]:


mnist = fetch_openml('mnist_784')


# In[3]:


# KNN avec k = 3 et distance euclidienne (p = 2)
k = 3
distance = 2

precision_knn = []
recall_knn = []
time_knn = []
tot_time_knn = []
error_knn = []
cm_knn = []

# MLPClassifier: RNN avec 10 couches de 50 neurones et alpha = 0.1 (avec ADAM et ReLu par défaut)
nb_couches = 10
archi = tuple(50 for i in range(nb_couches))
a = 0.1

precision_nn = []
recall_nn = []
time_nn = []
tot_time_nn = []
error_nn = []
cm_nn = []

# SVM
noyau = 'poly'
precision_svm = []
recall_svm = []
time_svm = []
tot_time_svm = []
error_svm = []
cm_svm = []

for d_size in [5000,10000,20000,30000,40000,50000,70000]:
    index = np.random.randint(70000, size=d_size)
    data = mnist.data[index]
    target = mnist.target[index]
    #division de la base en données d'apprentissage (80%) et de test (20%)
    datatrain, datatest, targettrain, targettest = train_test_split(data, target,train_size=0.80)
    print("size =", d_size)
    # KNN
    clf = neighbors.KNeighborsClassifier(k,p=distance,n_jobs=-1)
    start_time = time.time()
    clf.fit(datatrain,targettrain)
    elapsed = time.time()-start_time
    score = clf.score(datatest,targettest)
    tottime = time.time() - start_time
    testpred = clf.predict(datatest)
    rappel = recall_score(targettest,testpred,average='micro')
    erreur = zero_one_loss(targettest,testpred)
    cm = confusion_matrix(targettest,testpred)
    print(cm)
    cm_knn.append(cm)
    precision_knn.append(score)
    recall_knn.append(rappel)
    time_knn.append(elapsed)
    tot_time_knn.append(tottime)
    error_knn.append(erreur)
    print("KNN precision =", score,"time =",elapsed)
    # MLP (NN)
    mlp = nn.MLPClassifier(hidden_layer_sizes=archi,alpha=a)
    start_time = time.time()
    mlp.fit(datatrain,targettrain)
    elapsed = time.time()-start_time
    score = mlp.score(datatest,targettest)
    tottime = time.time() - start_time
    testpred = mlp.predict(datatest)
    rappel = recall_score(targettest,testpred,average='micro')
    erreur = zero_one_loss(targettest,testpred)
    cm = confusion_matrix(targettest,testpred)
    print(cm)
    cm_nn.append(cm)
    precision_nn.append(score)
    recall_nn.append(rappel)
    time_nn.append(elapsed)
    tot_time_nn.append(tottime)
    error_nn.append(erreur)
    print("NN precision =", score,"time =",elapsed)
    # SVM
    clsvm = SVC(kernel=noyau)
    start_time = time.time()
    clsvm.fit(datatrain,targettrain)
    elapsed = time.time()-start_time
    score = clsvm.score(datatest,targettest)
    tottime = time.time() - start_time
    testpred = clsvm.predict(datatest)
    rappel = recall_score(targettest,testpred,average='micro')
    erreur = zero_one_loss(targettest,testpred)
    cm = confusion_matrix(targettest,testpred)
    print(cm)
    cm_svm.append(cm)
    precision_svm.append(score)
    recall_svm.append(rappel)
    time_svm.append(elapsed)
    tot_time_svm.append(tottime)
    error_svm.append(erreur)
    print("SVM precision =", score,"time =",elapsed)


# In[15]:


print(error_svm)


# In[9]:


plt.plot([5000,10000,20000,30000,40000,50000,70000],precision_knn,'r+-', label="KNN")
plt.plot([5000,10000,20000,30000,40000,50000,70000],precision_nn,'b+-', label="MLP")
plt.plot([5000,10000,20000,30000,40000,50000,70000],precision_svm,'g+-', label="SVM")
plt.ylabel('precision')
plt.xlabel('size')
plt.legend()
plt.show()


# In[10]:


plt.plot([5000,10000,20000,30000,40000,50000,70000],recall_knn,'r+-', label="KNN")
plt.plot([5000,10000,20000,30000,40000,50000,70000],recall_nn,'b+-', label="MLP")
plt.plot([5000,10000,20000,30000,40000,50000,70000],recall_svm,'g+-', label="SVM")
plt.ylabel('recall')
plt.xlabel('size')
plt.legend()
plt.show()


# In[11]:


plt.plot([5000,10000,20000,30000,40000,50000,70000],time_knn,'r+-', label="KNN")
plt.plot([5000,10000,20000,30000,40000,50000,70000],time_nn,'b+-', label="MLP")
plt.plot([5000,10000,20000,30000,40000,50000,70000],time_svm,'g+-', label="SVM")
plt.ylabel('training time')
plt.xlabel('size')
plt.legend()
plt.show()


# In[13]:


plt.plot([5000,10000,20000,30000,40000,50000,70000],tot_time_knn,'r+-', label="KNN")
plt.plot([5000,10000,20000,30000,40000,50000,70000],tot_time_nn,'b+-', label="MLP")
plt.plot([5000,10000,20000,30000,40000,50000,70000],tot_time_svm,'g+-', label="SVM")
plt.ylabel('total time')
plt.xlabel('size')
plt.legend()
plt.show()


# In[14]:


plt.plot([5000,10000,20000,30000,40000,50000,70000],error_knn,'r+-', label="KNN")
plt.plot([5000,10000,20000,30000,40000,50000,70000],error_nn,'b+-', label="NN")
plt.plot([5000,10000,20000,30000,40000,50000,70000],error_svm,'g+-', label="SVM")
plt.ylabel('error')
plt.xlabel('size')
plt.show()

