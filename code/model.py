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
    
# step 1. shuffle the data
n,d = X.shape
index = list(range(0,n))
random.shuffle(index)

X = X[index, :]
y = y[index]

# step 2. split into train/test in 2/3-1/3
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
##logreg.score(Xtrain, ytrain)
##logreg.score(Xtest, ytest)
#y_hat = logreg.predict(Xtest)
##print accuracy
#from sklearn.metrics import accuracy_score
#print("test accuracy:",accuracy_score(ytest,y_hat))
#
##print confusion matrix
#from sklearn.metrics import confusion_matrix
#from util import plot_confusion_matrix
#import matplotlib.pyplot as plt
## Compute confusion matrix
#cnf_matrix = confusion_matrix(ytest, y_hat)
#np.set_printoptions(precision=2)
## Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=np.unique(y_hat),
#                      title='Confusion matrix, without normalization')


"""
custome logistic regression
"""
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=10000, fit_intercept=True, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.w = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.w)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.w -= self.lr * gradient
            
            if(self.verbose == True and i % 1000 == 0):
                z = np.dot(X, self.w)
                h = self.__sigmoid(z)
                print('loss: ',self.__loss(h, y))
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.w))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

#Evaluation
model = LogisticRegression(lr=0.1, num_iter=10000)
Xtrain =Xtrain.toarray()
Xtest = Xtest.toarray()
model.fit(Xtrain, ytrain)
preds = model.predict(Xtest)
# accuracy
print('accuracy',(preds == ytest).mean())

    
