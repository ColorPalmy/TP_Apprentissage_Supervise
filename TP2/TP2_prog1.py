#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, zero_one_loss, confusion_matrix
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
import time


# In[2]:


mnist = fetch_openml('mnist_784')


# In[3]:


data = mnist.data
target = mnist.target
#division de la base en données d'apprentissage (49000) et de test
datatrain, datatest, targettrain, targettest = train_test_split(data, target,train_size=0.70)
print(datatrain.shape)


# In[4]:


mlp = nn.MLPClassifier(hidden_layer_sizes=(50))
mlp.fit(datatrain,targettrain)
mlp.score(datatest,targettest)


# In[5]:


print(targettest[3])
print(mlp.predict(datatest)[3])


# In[6]:


print(precision_score(targettest,mlp.predict(datatest),average='micro'))


# In[7]:


precisions_couches = []
temps_couches = []
tt_couches = []
rappel_couches = []
erreur_couches = []
train_precisions_couches = []
for nb_c in [2,10,20,50,70,100]:
    hls = tuple(50 for i in range(nb_c))
    mlp = nn.MLPClassifier(hidden_layer_sizes=hls)
    st = time.time()
    start_time = time.process_time()
    mlp.fit(datatrain,targettrain)
    elapsed = time.process_time()-start_time
    duration = time.time()-st
    prec = mlp.score(datatest,targettest)
    testpred = mlp.predict(datatest)
    precisions_couches.append(prec)
    temps_couches.append(elapsed)
    tt_couches.append(duration)
    rap = recall_score(targettest,testpred,average='micro')
    err = zero_one_loss(targettest,testpred)
    rappel_couches.append(rap)
    erreur_couches.append(err)
    prec_train = mlp.score(datatrain,targettrain)
    train_precisions_couches.append(prec_train)
    print("precision =", prec, "temps =", elapsed,"time =",duration,"sur train =",prec_train, "pour", nb_c,"couches")


# In[8]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('number of layers')
ax1.set_ylabel('precision', color=color)
ax1.plot([2,10,20,50,70,100],precisions_couches,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('execution time', color=color)
ax2.plot([2,10,20,50,70,100],temps_couches,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[9]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('number of layers')
ax1.set_ylabel('precision', color=color)
ax1.plot([2,10,20,50,70,100],precisions_couches,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('execution time', color=color)
ax2.plot([2,10,20,50,70,100],tt_couches,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[10]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('number of layers')
ax1.set_ylabel('precision', color=color)
ax1.plot([2,10,20,50,70,100],precisions_couches,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('rappel', color=color)
ax2.plot([2,10,20,50,70,100],rappel_couches,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[11]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('number of layers')
ax1.set_ylabel('precision test', color=color)
ax1.plot([2,10,20,50,70,100],precisions_couches,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('precision train', color=color)
ax2.plot([2,10,20,50,70,100],train_precisions_couches,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[12]:


plt.plot([2,10,20,50,70,100],erreur_couches,'+-')
plt.ylabel('error')
plt.xlabel('number of layers')
plt.show()


# In[13]:


plt.plot([2,10,20,50,70,100],temps_couches,'+-')
plt.ylabel('training time')
plt.xlabel('number of layers')
plt.show()


# In[14]:


plt.plot([2,10,20,50,70,100],tt_couches,'+-')
plt.ylabel('training time')
plt.xlabel('number of layers')
plt.show()


# In[15]:


precisions_neurones = []
temps_neurones = []
tt_neurones = []
rappel_neurones = []
erreur_neurones = []
for nb_n in [5,10,20,30,40,50,60,70,80,90,100,150,200,300]:
    hls = tuple(nb_n for i in range(10))
    mlp = nn.MLPClassifier(hidden_layer_sizes=hls)
    st = time.time()
    start_time = time.process_time()
    mlp.fit(datatrain,targettrain)
    elapsed = time.process_time()-start_time
    duration = time.time() - st
    prec = mlp.score(datatest,targettest)
    testpred = mlp.predict(datatest)
    precisions_neurones.append(prec)
    temps_neurones.append(elapsed)
    tt_neurones.append(duration)
    rap = recall_score(targettest,testpred,average='micro')
    err = zero_one_loss(targettest,testpred)
    rappel_neurones.append(rap)
    erreur_neurones.append(err)
    prec_train = mlp.score(datatrain,targettrain)
    print("precision =", prec, "temps =", elapsed,"sur train =",prec_train, "pour", nb_n,"neurones")


# In[16]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('number of neurons')
ax1.set_ylabel('precision', color=color)
ax1.plot([5,10,20,30,40,50,60,70,80,90,100,150,200,300],precisions_neurones,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('rappel', color=color)
ax2.plot([5,10,20,30,40,50,60,70,80,90,100,150,200,300],rappel_neurones,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[17]:


plt.plot([5,10,20,30,40,50,60,70,80,90,100,150,200,300],temps_neurones,'+-')
plt.ylabel('training time')
plt.xlabel('number of neurons')
plt.show()


# In[18]:


plt.plot([5,10,20,30,40,50,60,70,80,90,100,150,200,300],tt_neurones,'+-')
plt.ylabel('training time')
plt.xlabel('number of neurons')
plt.show()


# In[19]:


plt.plot([5,10,20,30,40,50,60,70,80,90,100,150,200,300],erreur_neurones,'+-')
plt.ylabel('error')
plt.xlabel('number of neurons')
plt.show()


# In[20]:


tpl1 = tuple(60-i for i in range(50))
mlp = nn.MLPClassifier(hidden_layer_sizes=tpl1)
start_time = time.process_time()
mlp.fit(datatrain,targettrain)
elapsed = time.process_time() - start_time
prec = mlp.score(datatest,targettest)
print("50 couches: precision =",prec,"temps =",elapsed)
prec_train = mlp.score(datatrain,targettrain)
print(prec_train)


# In[21]:


pas = int(-49/20)
tpl2 = tuple(i for i in range(60,10,pas))
mlp = nn.MLPClassifier(hidden_layer_sizes=tpl2)
start_time = time.process_time()
mlp.fit(datatrain,targettrain)
elapsed = time.process_time() - start_time
prec = mlp.score(datatest,targettest)
print("20 couches: precision =",prec,"temps =",elapsed)
prec_train = mlp.score(datatrain,targettrain)
print(prec_train)


# In[22]:


cm = confusion_matrix(targettest,mlp.predict(datatest))
print(cm)


# In[23]:


tpl3 = tuple(i for i in range(60,13,3))
mlp = nn.MLPClassifier(hidden_layer_sizes=tpl3)
start_time = time.process_time()
mlp.fit(datatrain,targettrain)
elapsed = time.process_time() - start_time
prec = mlp.score(datatest,targettest)
print("20 couches: precision =",prec,"temps =",elapsed)
prec_train = mlp.score(datatrain,targettrain)
print(prec_train)


# In[24]:


tpl4 = tuple(i for i in range(10,60,5))
mlp = nn.MLPClassifier(hidden_layer_sizes=tpl4)
start_time = time.process_time()
mlp.fit(datatrain,targettrain)
elapsed = time.process_time() - start_time
prec = mlp.score(datatest,targettest)
print("10 couches: precision =",prec,"temps =",elapsed)
prec_train = mlp.score(datatrain,targettrain)
print(prec_train)


# In[25]:


nb_couches = 20
archi = tuple(50 for i in range(nb_couches))


# In[26]:


precisions_algos = []
rappel_algos = []
temps_algos = []
tt_algos = []
erreur_algos = []
for s in ["lbfgs","sgd","adam"]:
    mlp = nn.MLPClassifier(hidden_layer_sizes=archi,solver=s)
    st = time.time()
    start_time = time.process_time()
    mlp.fit(datatrain,targettrain)
    elapsed = time.process_time()-start_time
    duration = time.time()-st
    testpred = mlp.predict(datatest)
    prec = mlp.score(datatest,targettest)
    rap = recall_score(targettest,testpred,average='micro')
    err = zero_one_loss(targettest,testpred)
    precisions_algos.append(prec)
    rappel_algos.append(rap)
    temps_algos.append(elapsed)
    tt_algos.append(duration)
    erreur_algos.append(err)
    print("precision =", prec,"rappel =",rap,"erreur =",err,"temps =",elapsed,"pour",s)


# In[27]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('algorithm')
ax1.set_ylabel('precision', color=color)
ax1.plot(["lbfgs","sgd","adam"],precisions_algos,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('rappel', color=color)
ax2.plot(["lbfgs","sgd","adam"],rappel_algos,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[28]:


plt.plot(["lbfgs","sgd","adam"],temps_algos,'+-')
plt.ylabel('training time')
plt.xlabel('algorithm')
plt.show()


# In[29]:


plt.plot(["lbfgs","sgd","adam"],tt_algos,'+-')
plt.ylabel('training time')
plt.xlabel('algorithm')
plt.show()


# In[30]:


plt.plot(["lbfgs","sgd","adam"],erreur_algos,'+-')
plt.ylabel('error')
plt.xlabel('algorithm')
plt.show()


# In[31]:


precisions_activ = []
rappel_activ = []
temps_activ = []
tt_activ = []
erreur_activ = []
for a in ["identity", "logistic", "tanh", "relu"]:
    mlp = nn.MLPClassifier(hidden_layer_sizes=archi,activation=a)
    st = time.time()
    start_time = time.process_time()
    mlp.fit(datatrain,targettrain)
    elapsed = time.process_time()-start_time
    duration = time.time()-st
    testpred = mlp.predict(datatest)
    prec = mlp.score(datatest,targettest)
    rap = recall_score(targettest,testpred,average='micro')
    err = zero_one_loss(targettest,testpred)
    precisions_activ.append(prec)
    rappel_activ.append(rap)
    temps_activ.append(elapsed)
    tt_activ.append(duration)
    erreur_activ.append(err)
    print("precision =", prec,"rappel =",rap,"erreur =",err,"temps =",elapsed,"pour",a)


# In[32]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('activation function')
ax1.set_ylabel('precision', color=color)
ax1.plot(["identity", "logistic", "tanh", "relu"],precisions_activ,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('training time', color=color)
ax2.plot(["identity", "logistic", "tanh", "relu"],temps_activ,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[33]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('activation function')
ax1.set_ylabel('precision', color=color)
ax1.plot(["identity", "logistic", "tanh", "relu"],precisions_activ,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('training time', color=color)
ax2.plot(["identity", "logistic", "tanh", "relu"],tt_activ,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[34]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('activation function')
ax1.set_ylabel('precision', color=color)
ax1.plot(["identity", "logistic", "tanh", "relu"],precisions_activ,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('rappel', color=color)
ax2.plot(["identity", "logistic", "tanh", "relu"],rappel_activ,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[35]:


plt.plot(["identity", "logistic", "tanh", "relu"],temps_activ,'+-')
plt.ylabel('training time')
plt.xlabel('activation function')
plt.show()


# In[36]:


plt.plot(["identity", "logistic", "tanh", "relu"],tt_activ,'+-')
plt.ylabel('training time')
plt.xlabel('activation function')
plt.show()


# In[37]:


plt.plot(["identity", "logistic", "tanh", "relu"],erreur_activ,'+-')
plt.ylabel('error')
plt.xlabel('activation function')
plt.show()


# In[ ]:


precisions_regul = []
rappel_regul = []
temps_regul = []
tt_regul = []
erreur_regul = []
for a in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    mlp = nn.MLPClassifier(hidden_layer_sizes=archi,alpha=a)
    st = time.time()
    start_time = time.process_time()
    mlp.fit(datatrain,targettrain)
    elapsed = time.process_time()-start_time
    duration = time.time()-st
    testpred = mlp.predict(datatest)
    prec = mlp.score(datatest,targettest)
    rap = recall_score(targettest,testpred,average='micro')
    err = zero_one_loss(targettest,testpred)
    precisions_regul.append(prec)
    rappel_regul.append(rap)
    temps_regul.append(elapsed)
    tt_regul.append(duration)
    erreur_regul.append(err)
    print("precision =", prec,"rappel =",rap,"erreur =",err,"temps =",elapsed,"pour",a)


# In[ ]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('regularization')
ax1.set_xscale("log")
ax1.set_ylabel('precision', color=color)
ax1.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],precisions_regul,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('execution time', color=color)
ax2.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],temps_regul,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('regularization')
ax1.set_xscale("log")
ax1.set_ylabel('precision', color=color)
ax1.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],precisions_regul,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('execution time', color=color)
ax2.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],tt_regul,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('regularization')
ax1.set_xscale("log")
ax1.set_ylabel('precision', color=color)
ax1.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],precisions_regul,'+-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('rappel', color=color)
ax2.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],rappel_regul,'r+-',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()


# In[ ]:


plt.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],temps_regul,'+-')
plt.ylabel('training time')
plt.xlabel('regularization')
plt.xscale('log')
plt.show()


# In[ ]:


plt.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],tt_regul,'+-')
plt.ylabel('training time')
plt.xlabel('regularization')
plt.xscale('log')
plt.show()


# In[ ]:


plt.plot([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],erreur_regul,'+-')
plt.ylabel('error')
plt.xlabel('regularization')
plt.xscale('log')
plt.show()

