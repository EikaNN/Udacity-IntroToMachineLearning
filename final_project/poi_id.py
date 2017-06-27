#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features

# Financial features
features_list.append('salary')
features_list.append('deferral_payments')
features_list.append('total_payments')
features_list.append('loan_advances')
features_list.append('bonus')
features_list.append('restricted_stock_deferred')
features_list.append('deferred_income')
features_list.append('total_stock_value')
features_list.append('expenses')
features_list.append('exercised_stock_options')
features_list.append('other')
features_list.append('long_term_incentive')
features_list.append('restricted_stock')
features_list.append('director_fees')

# Email features
features_list.append('to_messages')
features_list.append('from_messages')
features_list.append('from_poi_to_this_person')
features_list.append('from_this_person_to_poi')
features_list.append('shared_receipt_with_poi')

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

outliers = ['TOTAL']

for outlier in outliers:
    del data_dict[outlier]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for feature in features_list:
    print "The feature {} has {} missing values"\
        .format(feature, [person[feature] for person in my_dataset.values()].count('NaN'))

for person in my_dataset.values():
    if 'NaN' not in (person['from_poi_to_this_person'], person['from_messages']):
        person['from_poi_ratio'] = \
            float(person['from_poi_to_this_person']) / float(person['from_messages'])
    else:
        person['from_poi_ratio'] = 0

for person in my_dataset.values():
    if 'NaN' not in (person['from_this_person_to_poi'], person['to_messages']):
        person['to_poi_ratio'] = \
            float(person['from_this_person_to_poi']) / float(person['to_messages'])
    else:
        person['to_poi_ratio'] = 0

features_list.append('from_poi_ratio')
features_list.append('to_poi_ratio')

features_list.remove('loan_advances')

features_list.remove('to_messages')
features_list.remove('from_messages')
features_list.remove('from_poi_to_this_person')
features_list.remove('from_this_person_to_poi')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

print "The number of data points is", len(labels)
print "The number of non-POIs is", labels.count(0)
print "The number of POIs is", labels.count(1)

import matplotlib.pyplot

x_feature = 'salary'
y_feature = 'deferral_payments'

for point, poi in zip(features, labels):
    x = point[features_list.index(x_feature) - 1]
    y = point[features_list.index(y_feature) - 1]
    color = 'red' if poi else 'blue'
    matplotlib.pyplot.scatter(x, y, c=color)

matplotlib.pyplot.xlabel(x_feature)
matplotlib.pyplot.ylabel(y_feature)
matplotlib.pyplot.savefig(x_feature + '_' + y_feature + '.png')

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [5, 10, None],
    'min_samples_split': [1, 2, 5, 10]
}

SVC = DecisionTreeClassifier(random_state=42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

from sklearn.grid_search import GridSearchCV

grid = GridSearchCV(SVC, param_grid, cv=10, scoring='f1', verbose=2, n_jobs=-1)

grid.fit(features_train, labels_train)

print "The score on the testing set is", grid.best_estimator_.score(features_test, labels_test)

clf = grid.best_estimator_

print "The best estimator found by GridSearch is"
print grid.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
