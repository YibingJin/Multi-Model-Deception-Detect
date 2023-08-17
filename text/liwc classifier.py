import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utilities import train_model
from sklearn.ensemble import RandomForestClassifier
x_train = pd.read_csv("./transcript_train.csv",header=None)
x_test = pd.read_csv("./transcript_test.csv",header=None)
y_train = pd.read_csv("./transcript_train_label.csv",header=None)
print(np.any(np.isnan(x_train)))
print(np.all(np.isfinite(x_train)))
print(np.where(np.isnan(x_train)))
y_test = pd.read_csv("./transcript_test_label.csv",header=None)
classifier = RandomForestClassifier(n_estimators=1000, random_state=None, max_features='auto', min_samples_split=2)
classifier.fit(x_train,y_train)
result=classifier.predict(x_test)
print('predict',result)
print('label',y_test)