# Stratified K Fold

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


iris = load_iris()
features = iris.data
label = iris.target

dt_clf = DecisionTreeClassifier(random_state=156)
skf = StratifiedKFold(n_splits=3)
n_iter=0
cv_accuracy = []

for train_index, test_index in skf.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(n_iter, accuracy, train_size, test_size)
    print("검증 세트 인덱스" + str(n_iter) + " " + str(test_index))
    cv_accuracy.append(accuracy)
    
print(np.round(cv_accuracy, 4))

# 간편한 교차 검증
from sklearn.model_selection import cross_val_score, cross_validate
scores = cross_val_score(dt_clf, features, label, scoring='accuracy', cv=3)
print('교차 검증별 정확도:', np.round(scores, 4))