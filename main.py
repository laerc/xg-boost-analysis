from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.io import arff

import numpy as np
import pandas as pd

# Returns and array of predict and expected values
def train_and_test(n_splits, data, classifier):
  results = []

  kf = KFold(n_splits)
  for train_index, test_index in kf.split(data):
    train, test = data[train_index], data[test_index]
    
    #convert to ta list, then remove the last element(the class)
    x_train = [ (list(elem))[0:-1] for elem in train ]
    y_train = train['class']

    #train the model
    classifier = classifier.fit(x_train, y_train)
    
    #convert to ta list, then remove the last element(the class)
    x_test = [ (list(elem))[0:-1] for elem in test ]  
    y_test = test['class']

    #predict
    y_predicted = classifier.predict(x_test)
    results.append([ y_test, y_predicted ])
  
  return results

def evalute_method(results, average):
  for execution in results:
    y_test = execution[0]
    y_predicted = execution[1]

    print('Accuracy: {} '.format(accuracy_score(y_test, y_predicted)))
    print('F1 score: {} '.format(f1_score(y_test, y_predicted, average=average)))


def main ():
  file_path = 'UCI/iris.arff'
  n_splits = 10

  #create classifier
  decision_tree = DecisionTreeClassifier()

  #read .arff from file
  data, meta = arff.loadarff(file_path)

  #shuffle the data
  shuffled_data = np.random.shuffle(data)

  #train and test classfier using K-fold
  results = train_and_test(n_splits, data, decision_tree)

  evalute_method(results, average='micro')


main()
