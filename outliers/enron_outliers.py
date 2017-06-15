#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
del data_dict['TOTAL']

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.savefig("data.png")

def isBandit(person):
    if 'NaN' in (person["bonus"], person["salary"]):
        return False

    return person["salary"] > 1e6 and person["bonus"] > 5e6

for person in data_dict.items():
    if isBandit(person[1]):
        print person[0], "is a bandit"


