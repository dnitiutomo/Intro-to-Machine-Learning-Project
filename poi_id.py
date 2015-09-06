#!/usr/bin/python

import sys
sys.path.append("../tools/")
sys.path.append( "../choose_your_own/" )

import pickle
import matplotlib.pyplot as plt
import numpy as py
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from outlier_cleaner import univariate_outlier_cleaner
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import SelectKBest, f_classif
import data_explorer as expl


'''
Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
'''
#features_list = ['poi','salary']
#features_list = ['poi','salary', 'from_this_person_to_poi','from_poi_to_this_person']
#features_list = ['poi','salary','total_payments','expenses','restricted_stock']
#features_list = ['poi','salary','shared_receipt_with_poi','total_payments','expenses','total_stock_value']
'''
The following are the best features found from using the Decision Tree Classifier importances
'''
features_list = ['poi','exercised_stock_options','bonus','shared_receipt_with_poi','expenses','other']

'''
features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus',
                 'restricted_stock','shared_receipt_with_poi',
                 'total_stock_value','expenses','from_messages','other','from_this_person_to_poi',
                 'deferred_income','long_term_incentive','from_poi_to_this_person','restricted_stock_deferred',
                 'loan_advances','director_fees']
'''
'''
features_list = ['poi','total_payments','exercised_stock_options','bonus',
                 'restricted_stock','expenses','other','from_this_person_to_poi',
                 'deferred_income']
'''
'''
Load the dictionary containing the dataset
'''
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

'''
Task 2: Remove outliers
'''
#data_dict = univariate_outlier_cleaner(data_dict, features_list,5)
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

'''
Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
'''
#expl.count_missing_data_points(data_dict, features_list)
#expl.summarize_features(data_dict, features_list)
#data_dict = expl.minmax_standardize_data(data_dict, features_list)
data_dict = expl.zscore_standardize_data(data_dict, features_list)

#features_list.append('messages_recevied_poi_ratio')
features_list.append('messages_sent_poi_ratio')
#features_list.append('bonus_ratio')
#features_list.append('expense_ratio')
#features_list.append('salary_ratio')

data_dict = expl.add_message_ratios(data_dict)

my_dataset = data_dict

'''
Extract features and labels from dataset for local testing
'''
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

'''
Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
'''
### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Decision Tree
from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(criterion="entropy",max_features=2)
#clf = DecisionTreeClassifier(criterion="entropy",min_samples_split=4)
#clf = DecisionTreeClassifier(criterion="entropy",max_depth=5)
#clf = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=9)
#clf = DecisionTreeClassifier(criterion="gini",max_leaf_nodes=5)
#clf = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=7,min_samples_leaf=12)

from sklearn.grid_search import GridSearchCV
tuned_parameters = [{ 'min_samples_split':[5,10,15],
                     'max_depth':[5,10,15], 'min_samples_leaf':[5,10,12]}]
#clf = GridSearchCV(DecisionTreeClassifier(criterion="entropy"),tuned_parameters, cv=5,scoring='average_precision')

### ----------CHOSEN & TUNED ALGORITHM--------------
### Includes ZScoreStandardized features
clf = DecisionTreeClassifier(criterion="entropy",max_depth=5,min_samples_leaf=12)
### ------------------------------------------------

### Support Vector Machines
from sklearn.svm import SVC
#clf = SVC()

### KMeans Clustering - Note the only valid K-Means clustering is n_clusters = 2 as there are only 2 categories
#### Actual Kmeans clustering is not valid for this problem as it needs a supervised training algorithm, not
#### unsupervised
from sklearn.cluster import KMeans
#clf = KMeans(n_clusters=2)

### KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier()
#clf = KNeighborsClassifier(n_neighbors=3,weights='distance')
#clf = KNeighborsClassifier(n_neighbors=3,algorithm='brute')
#clf = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree',leaf_size=70)

### Logistic Regression
from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression()

### Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier()
#clf = GradientBoostingClassifier(learning_rate=0.05)
#clf = GradientBoostingClassifier(n_estimators=200)
#clf = GradientBoostingClassifier(max_depth=2)
#clf = GradientBoostingClassifier(min_samples_split=7)
#clf = GradientBoostingClassifier(min_samples_leaf=5)
#clf = GradientBoostingClassifier(subsample=0.25)
#clf = GradientBoostingClassifier(warm_start=True)
#clf = GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,warm_start=True)

tuned_parameters = [{ 'learning_rate':[0.1,0.5,1,2],
                     'n_estimators':[50,100,200,400], 'max_depth':[2,5,10]}]

#clf = GridSearchCV(GradientBoostingClassifier(warm_start=True),tuned_parameters)

### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()

'''
Conduct feature selection/transformation algorithms here
'''
#fs_filter = SelectKBest(k=7)
fs_filter = RandomizedPCA(n_components=3)
#clf = make_pipeline(fs_filter, clf)

'''
Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''
print len(my_dataset)
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)