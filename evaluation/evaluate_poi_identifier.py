#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

np.random.seed(42)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print "The number of predicted POIs in the test set is", np.count_nonzero(pred)
print "The number of total people in the test set is", len(pred)

print "The number of true positives is", confusion_matrix(labels_test, pred)[1, 1]
print "The precision of the test data is", precision_score(labels_test, pred)
print "The recall of the test data is", recall_score(labels_test, pred)
