import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from sklearn import model_selection, preprocessing, naive_bayes, svm

visual=pd.read_csv('./alldata.csv',header=None)
text=pd.read_csv('./newtext.csv',header=None)
visual=np.array(visual.values)
text=np.array(text.values)
allinfo=np.concatenate((text,visual),axis=1)
print("allinfo:",allinfo.shape)
print("visual:",visual.shape)
print("text:",text.shape)
label=pd.read_csv('./labels.csv',header=None)
encoder = preprocessing.LabelEncoder()
label = encoder.fit_transform(label)
print(label)
alldata = StandardScaler().fit_transform(allinfo)
train_x, test_x, train_y, test_y = model_selection.train_test_split(allinfo, label, test_size=0.15, random_state=None)
# predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr',kernel='linear')
# predictor.fit(train_x, train_y)
# result = predictor.predict(test_x)
# print("F-score: {0:.2f}".format(f1_score(result,test_y,average='micro')))

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split




def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 15, base=2)
    gamma_range = np.logspace(-9, 5, 15, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=10, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('精度为%s' % score)
    return score

svm_c(train_x,  test_x, train_y, test_y )