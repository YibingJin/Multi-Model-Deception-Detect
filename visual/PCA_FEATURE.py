import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
path='D:\PyCharm 2021.3.2\openface\\visual\\lie_visual\original'
files= os.listdir(path)
i=0
for file in files:
    i+=1
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)
    name = os.path.basename(file_path)
    print(name)
    x = StandardScaler().fit_transform(data)
    pca_data = PCA(n_components=19)
    principalComponents_data = pca_data.fit_transform(x)
    newdata = pd.DataFrame(principalComponents_data)
    i_str0 = str(i)
    i_str = "new"+name
    newdata.to_csv(i_str, header=None, index=None)

    print(newdata)
    print('Explained variation principal component: {}'.format(pca_data.explained_variance_ratio_.sum()))
    a = pca_data.explained_variance_ratio_.sum()
    df = pd.DataFrame(newdata.values.T)
    # print(df.info)
    # y = StandardScaler().fit_transform(df)
    # pca_data1 = PCA(n_components=18)
    # principalComponents_data1 = pca_data1.fit_transform(y)
    # finaldata = pd.DataFrame(principalComponents_data1)
    # print(finaldata.info)
    # print('Explained variation principal component: {}'.format(pca_data1.explained_variance_ratio_.sum()))
    # b = pca_data1.explained_variance_ratio_.sum()
    # c = a * b
    # print('overall explained component:{}'.format(c))
    # i_str0 = str(i)
    # i_str = "new"+name
    # filename = i_str + '.csv'
    # finaldata.to_csv(i_str,header=None, index=None)

############text-liwc pca
# PATH='./LIWC_RealLife.csv'
# df=pd.read_csv(PATH)
# df=df.drop('Filename',axis=1)
# df=df.drop('Segment',axis=1)
# df=df.drop('Deceptive Class',axis=1)
# x = StandardScaler().fit_transform(df)
# pca_data = PCA(n_components=27)
# principalComponents_data = pca_data.fit_transform(x)
#
# newdata=StandardScaler().fit_transform(principalComponents_data)
# newdata = pd.DataFrame(newdata)
# newdata.to_csv('newtext.csv', header=None, index=None)
# print('Explained variation principal component: {}'.format(pca_data.explained_variance_ratio_.sum()))