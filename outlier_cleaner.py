#!/usr/bin/python
import numpy as np

to_include = "inclu"

def create_list(data_set, feature):
    feat_list = []
    for name in data_set:
        if isinstance(data_set[name][feature],int):
            feat_list.append(data_set[name][feature])


def univariate_outlier_cleaner(data_set,feature_list,threshold):
    for name in data_set:
        data_set[name][to_include] = True
    for feature in feature_list:
        if feature != 'poi':
            numbers = []

            for name in data_set:
                if isinstance(data_set[name][feature],int):
                    numbers.append(data_set[name][feature])

            median = np.median(numbers,axis=0)
            diff = (numbers - median)**2
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            #modified_z_score = 0.6745 * diff / med_abs_deviation

            for name in data_set:
                if isinstance(data_set[name][feature],int):
                    modified_z_score = 0.6745 * np.sqrt((data_set[name][feature] - median)**2) / med_abs_deviation
                    #print modified_z_score
                    if modified_z_score > threshold:
                        data_set[name][to_include] = False

    cleaned_data = {}

    for name in data_set:
        if data_set[name][to_include]:
            cleaned_data[name] = data_set[name]
    print len(cleaned_data)
    return cleaned_data

'''
def outlierCleaner(data_set, feature_list):
    false_list = {}
    cleaned_data = {}
    features_trans = []
    count = 0
    for name in data_set:
        item = []
        for feature in feature_list:
            if feature != 'poi':
                value = data_set[name][feature]
                item.append(value)
        if 'NaN' not in item:
            features_trans.append(item)
        else:
            count += 1
            #print item
    #print count
    outliers = is_outlier(features_trans, 4)

    for i in range(0, len(outliers)):
        if outliers[i] == False:
            cleaned_data.append(features_trans[i])
    print len(data_set)
    print len(features_trans)
    print len(cleaned_data)
    return cleaned_data
'''