# coding=utf-8
import json
import os

import sklearn.datasets
from joblib import dump
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# instantiate classifier and scaler
clf = svm.SVC(verbose=True)
scaler = StandardScaler()

# get training data
X, y = sklearn.datasets.load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)

# scale data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# evaluate the quality of the trained model
f1_metric = f1_score(y_test, y_pred, average='weighted')
print(f"f1 score: {round(f1_metric, 3)}")

# persist model
dump(clf, 'model.joblib')

# write metrics
if not os.path.exists("metrics"):
    os.mkdir("metrics")
with open("metrics/f1.metric", "w+") as f:
    json.dump(f1_metric, f)
