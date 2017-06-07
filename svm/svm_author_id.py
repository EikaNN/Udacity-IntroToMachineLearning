#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Slice the training set in order to speed up training
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel='rbf', C=10000)

start = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-start, 3), "s"

start = time()
pred = clf.predict(features_test)
print "Testing time:", round(time()-start, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print "The accuracy with C = 10000 is", accuracy

for position in [10, 26, 50]:
    print "prediction[", position, "] =", pred[position]

print "There are", np.count_nonzero(pred), "emails predicted to be in the 1 class"

#########################################################
