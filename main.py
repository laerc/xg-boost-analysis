from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from scipy.io import arff

import numpy as np
import pandas as pd

def drop_column(data, columns=[]):
  return data.drop(columns=columns)

def boolean_columns(data, meta):
  columns = [x for x in meta]
  for column in columns:
    if(meta[column][1] == ('false', 'true')):
      col_values = [elem for elem in data[column]]
      data[column] = [0 if elem == b'false' else 1 for elem in col_values]

  return data
  

def normalize_data(data):
  min_max_scaler = MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(data)
  df = pd.DataFrame(x_scaled)
  return df

def split_column_in_segments(data, columns=[], n_segments=0):

  for column in columns:
    min_val = data[column].min()
    max_val = data[column].max()

    interval_length = (max_val - min_val) / n_segments
    col_values = data[column]
    data[column] = [ int((value - min_val - 1)/interval_length) for value in col_values]
    
  return data  

def categorical_to_score(data, cat_column):
  data[cat_column] = pd.factorize(data[cat_column])[0]
  return data

# Returns and array of predict and expected values
def train_and_test(n_splits, data, classifier, column_class='class', normalize=False):
  results = []

  kf = KFold(n_splits)
  for train_index, test_index in kf.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
  
    #convert to ta list, then remove the last element(the class)
    x_train = (train.loc[:, train.columns != column_class])
    y_train = (train[column_class])

    if(normalize == True):
      x_train = normalize_data(x_train)

    classifier = classifier.fit(x_train, y_train)
    
    #convert to ta list, then remove the last element(the class)
    x_test = (test.loc[:, test.columns != column_class])  
    y_test = (test[column_class])

    if(normalize == True):
      x_test = normalize_data(x_test)

    #predict
    y_predicted = classifier.predict(x_test)
    results.append([ y_test, y_predicted ])
  
  return results

def evalute_method(results, average):
  
  best_f1_score = 0.0
  best_acc_score = 0.0
  avg_f1_score = 0.0
  avg_acc_score = 0.0

  for execution in results:
    y_test = execution[0]
    y_predicted = execution[1]
    
    best_f1_score = max(best_f1_score, f1_score(y_test, y_predicted, average=average))
    best_acc_score = max(best_acc_score, accuracy_score(y_test, y_predicted))

    avg_f1_score += f1_score(y_test, y_predicted, average=average)
    avg_acc_score += accuracy_score(y_test, y_predicted)
  
  avg_f1_score /= float(len(results))
  avg_acc_score /= float(len(results))

  print('Max Accuracy: {} '.format(best_acc_score))
  print('Max F1 score: {} '.format(best_f1_score))

  print('Average Accuracy: {} '.format(avg_acc_score))
  print('Average F1 score: {} '.format(avg_f1_score))

  print('-------------------------------------------------------------------')
  return best_acc_score, best_f1_score


def solve(file_path, classifiers, column_class='class', cat_column='', columns_to_drop=[], 
          discrete_columns=[], n_splits=5, n_segments=5, average='macro', normalize=False):
 
  #read .arff from file
  data, meta = arff.loadarff(file_path)

  dataset = pd.DataFrame(data, columns=meta.names())
  
  data = dataset

  data = split_column_in_segments(data, discrete_columns, n_segments)
  
  #transform boolean string to integer
  boolean_columns(data,meta)

  #if there is column to drop
  if(len(columns_to_drop) > 0):
    data = drop_column(data, columns=columns_to_drop)

  #if there is a categorical column 
  if(cat_column != ''):
    data=categorical_to_score(data,cat_column=cat_column)

  #shuffle the data
  shuffled_data = shuffle(data)
  
  for classifier, classifier_name in classifiers:
    print(classifier_name) 
    #train and test classfier using K-fold
    results = train_and_test(n_splits, shuffled_data, classifier, column_class=column_class, normalize=normalize)

    evalute_method(results, average)


def main ():
  files_path = ['UCI/diabetes.arff', 'UCI/glass.arff', 'UCI/iris.arff', 'UCI/letter.arff', 'UCI/segment.arff', 'UCI/zoo.arff']
  file_drop_column = [[], [], [], [], [], ['animal']]
  file_cat_column = ['class', 'Type', 'class', 'class', 'class', 'type']
  file_discrete_column = [['age'], [], [], [], [], []]
  
  n_splits = 5

  #create classifier
  decision_tree = DecisionTreeClassifier()

  #create classifier
  random_forest = RandomForestClassifier(n_estimators=100)

  #create classifier
  xg_boost = XGBClassifier()

  for index in range(len(files_path)):
    file_path = files_path[index]
    drop_column = file_drop_column[index]
    cat_column = file_cat_column[index]
    discrete_column = file_discrete_column[index]
    print('================================== {} ====================================='.format(file_path))
    #solve for a classifier
    solve(file_path=file_path, classifiers=[(decision_tree, 'Decision Tree'), (random_forest, 'Random Forest'), (xg_boost, 'XG Boosting Tree')], 
          column_class=cat_column, cat_column=cat_column, columns_to_drop=drop_column, discrete_columns=discrete_column, n_splits=n_splits)

main()
