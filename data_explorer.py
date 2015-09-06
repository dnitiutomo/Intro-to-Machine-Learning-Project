__author__ = 'dnitiutomo'

import pandas as pd
import numpy as np

def count_missing_data_points(data_set, feature_list):
    missing_feats=[]
    missing_half=0
    pois = 0
    for person in data_set:
        missing=0
        for feature in feature_list:
            if feature != "poi":
                x = data_set[person][feature]
                if not isinstance(data_set[person][feature],int):
                    missing+=1
            else:
                if data_set[person][feature]:
                    pois+=1
        if missing > 10:
            missing_half+=1
        missing_feats.append(missing)
    print "Average Num. of Missing Data Points per Person: ", np.mean(missing_feats)
    print "Number of Observations missing over half of data points: ", missing_half
    print "Num. of POIs: ", pois

def summarize_features(data_set, feature_list):
    df_values = pd.DataFrame.from_dict(data_set, orient="index")
    for feat in feature_list:
        if feat != 'poi':
            print feat
            col = df_values[df_values[feat] != 'NaN'][feat].astype(float)
            if len(col) < 20:
                print feat + " does not have sufficient obesrvations  "
            print col.describe()


def minmax_standardize_data(data_set, feature_list):
    print '-------------Min Max Standardizing Features----------------'
    df_values = pd.DataFrame.from_dict(data_set, orient="index")
    for feature in feature_list:
        if feature != 'poi':
            col = df_values[df_values[feature] != 'NaN'][feature].astype(float)
            min = np.min(col)
            max = np.max(col)
            for person in data_set:
                x = data_set[person][feature]
                if isinstance(data_set[person][feature],int):
                    data_set[person][feature] = (x - min) / (max - min)
                    #print data_set[person][feature]
    return data_set

def zscore_standardize_data(data_set, feature_list):
    print '-------------Z Score Standardizing Features----------------'
    df_values = pd.DataFrame.from_dict(data_set, orient="index")
    for feature in feature_list:
        if feature != 'poi':
            col = df_values[df_values[feature] != 'NaN'][feature].astype(float)
            mean = np.mean(col)
            stddev = np.std(col)
            for person in data_set:
                x = data_set[person][feature]
                if isinstance(data_set[person][feature],int):
                    data_set[person][feature] = (x - mean) / (stddev)
                    #print data_set[person][feature]
    return data_set

def add_message_ratios(data_dict):
    for name in data_dict:
        valueRec = 'NaN'
        valueSen = 'NaN'
        valueBonusRatio = 'NaN'
        valueExpenseRatio = 'NaN'
        valueSalaryRatio = 'NaN'
        if isinstance(data_dict[name]['from_poi_to_this_person'], int) and isinstance(data_dict[name]['to_messages'], int) and isinstance(data_dict[name]['from_this_person_to_poi'], int) and isinstance(data_dict[name]['from_messages'], int):
            valueRec = (float(data_dict[name]['from_poi_to_this_person']) / data_dict[name]['to_messages'])
            valueSen = (float(data_dict[name]['from_this_person_to_poi']) / data_dict[name]['from_messages'])

        if isinstance(data_dict[name]['total_payments'], int):
            if isinstance(data_dict[name]['bonus'], int):
                valueBonusRatio = (float(data_dict[name]['bonus'])) / data_dict[name]['total_payments']
            if isinstance(data_dict[name]['expenses'], int):
                valueExpenseRatio = (float(data_dict[name]['expenses'])) / data_dict[name]['total_payments']
            if isinstance(data_dict[name]['salary'], int):
                valueExpenseRatio = (float(data_dict[name]['salary'])) / data_dict[name]['total_payments']

        data_dict[name]['messages_recevied_poi_ratio'] = valueRec
        data_dict[name]['messages_sent_poi_ratio'] = valueSen
        data_dict[name]['bonus_ratio'] = valueBonusRatio
        data_dict[name]['expense_ratio'] = valueExpenseRatio
        data_dict[name]['salary_ratio'] = valueSalaryRatio
    return data_dict
