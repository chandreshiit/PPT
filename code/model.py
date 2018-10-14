#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:37:13 2018

@author: chandresh
"""

from collections import defaultdict, Counter
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# specify directory to load
directory = '/home/chandresh/ckm/data/movies review/review_polarity/txt_sentoken/'
inpfile    = 'processed_review.txt'
vocab_dict ='movie_dict.txt'


"""
Step 1. load the data
"""
data =[]
with open(directory+inpfile,'r') as fid:
    for doc in fid:
         tokens = doc.strip('\n')
         data.append(tokens)
"""
step 2. load the dictionary
"""
vocabulary=defaultdict(int)

with open(directory+vocab_dict,'r') as fid:
    words = fid.readlines()
    words = [w.strip('\n') for w in words]
    vocabulary = {w:i for i, w in enumerate(words)}
        
"""
step 3. Using tf-idf model for encoding documents
"""
## define data matrix X and label array y
#X = np.zeros((len(data),len(vocabulary)))
#pos_label = [1]*1000
#neg_label = [0]*1000
#
#y = np.array(pos_label+neg_label)
#
#for row,doc in enumerate(data):        
#    BoW    = Counter(tokens)
#    for  w, c in BoW.items():
#        #get the index from dict
#      X[row,  vocabulary[w]] = c
#
## clear data to free up space
#del vocabulary
#del data

# define data matrix X and label array y
pos_label = [1]*1000
neg_label = [0]*1000
y = np.array(pos_label+neg_label)
tfidf = TfidfVectorizer(ngram_range=(1,1), vocabulary=vocabulary)
X = tfidf.fit_transform(data)

# clear data to free up space
del vocabulary
del data

"""
step 4. Construct the model, remember logistic regression is a linear model
"""        
# step 1. initialize the model
n,d = X.shape
w = np.zeros((d,))
    
# step 2. shuffle the data

index = list(range(0,n))
random.shuffle(index)

X = X[index, :]
y = y[index]

# split into train/test in 2/3-1/3
Xtrain = X[:1500,:]
ytrain = y[:1500]

Xtest = X[1500:,:]
ytest = y[1500:]

del X
del y

"""
Using scikit in-built logistic regression
"""

#from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()
#
#logreg.fit(Xtrain, ytrain)
#logreg.score(Xtrain, ytrain)
#logreg.score(Xtest, ytest)
#y_hat = logreg.predict(Xtest)
#from sklearn.metrics import accuracy_score
#print("test accuracy:",accuracy_score(ytest,y_hat))

"""
custome logistic regression
"""
# some usuful functions
def sigmoid(x):
    # numerically stable sigmoid
    if x>= 0:
        z = np.exp(x)
        return 1 / (1 + z)    
    else:
        z =  np.exp(x)
        return z / (1 + z)
    
def loss(w,X,y):
    z = np.dot(X,w)
    z = np.array([ sigmoid(t) for t in z])
    # loss = -\sum_i^n(y^ilog(z) +(1-y)log(1-z))
    first_term = np.multiply(y,np.log(z))
    second_term = np.multiply(1-y,np.log(1-z))
    
    loss = -np.sum(first_term+second_term)/len(y)
    
    return loss
 
    
def gradient(x,X,y):
     z = np.dot(X,w)
     z = np.array([ sigmoid(t) for t in z])
     grad = np.multiply(z.reshape((len(z),)), X.T)
     grad = np.sum(grad,axis=1)
     return grad
    
# iterate  over data
# Batch Gradient descent
maxiter         = 100
learning_rate   = 0.1
print_step      = int(maxiter/10)

for it in range(maxiter):
    #calculate gradient of the objective
    grad = gradient(w,Xtrain,ytrain)
    # update
    w = w -learning_rate*grad.T
    
    # bookkeeping
    #print loss time to time
    if it%print_step == 0:
        print("loss: ",loss(w,Xtrain,ytrain))
        
