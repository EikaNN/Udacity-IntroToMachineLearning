#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

m = len(enron_data)
print "There are {} data points (people) in the Enron dataset".format(m)

n = len(enron_data.itervalues().next())
print "There are {} features for each person".format(n)

print "The number of POIs in the Enron dataset is", \
    [person['poi'] for person in enron_data.values()].count(True)

poi_names = [line for line in open("../final_project/poi_names.txt", "r") if line.startswith('(')]
print "There are {} POIs in the poi_names.txt file".format(len(poi_names))

print "The total value of the stock belonging to James Prentice is", \
    enron_data['PRENTICE JAMES']['total_stock_value']

print "The number of messages from Wesley Colwell to persons of interest is", \
    enron_data['COLWELL WESLEY']['from_this_person_to_poi']

print "The value of stock options exercised by Jeffrey K Skilling is", \
    enron_data['SKILLING JEFFREY K']['exercised_stock_options']

for person in ['LAY KENNETH L', 'SKILLING JEFFREY K', 'FASTOW ANDREW S']:
    print "{} took home {}".format(person.title(), enron_data[person]['total_payments'])

print "A feature that doesn't have a well-defined values is denoted", \
    enron_data['LAY KENNETH L']['director_fees']

print "The number of persons that have a quatified salary is", \
    m - [person['salary'] for person in enron_data.values()].count('NaN')

print "The number of persons that have a known email address is", \
    m - [person['email_address'] for person in enron_data.values()].count('NaN')

unknown_payments = [person['total_payments'] for person in enron_data.values()].count('NaN')
print "The number of persons that have NaN for their total payments is", unknown_payments, \
    "which is about", round(100.0*unknown_payments/m, 1), "percent of the dataset"

print "The number of persons that are POI and have NaN for their total payment is", \
    [person['total_payments'] for person in enron_data.values() if person['poi']].count('NaN')
