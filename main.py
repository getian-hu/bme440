import pandas as pd
import csv
from sklearn.decomposition import PCA
from sympy.physics.control.control_plots import plt
import numpy as np
import sklearn
from sklearn import svm, datasets
import pandas as pd
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


data = pd.read_csv("/Users/hugetian/Desktop/BME440/svm_rf7.csv",header=0)
data = data.replace(np.nan, 0)
print(data.shape)
print(data)
#590
data = data.drop("sample",axis=1)



pca = PCA()


#PCA part of dead patients data
data1 = data.iloc[1:589, 1:157].values
data1 =data1.T
pcadata1 = pca.fit_transform(data1)
print('original shape for data1:',data1.shape)
print('transformed shape for data1:',pcadata1.shape)
print(pca.explained_variance_ratio_.sum())

#return dimensionality
pcadata3 = pca.inverse_transform(pcadata1)
np.savetxt("data3.csv",pcadata3.T, delimiter = ',')

#PCA part of live patients data
data2 = data.iloc[590:8451,1:157].values
data2 = data2.T
pcadata2 = pca.fit_transform(data2)
print('original shape for data2:',data2.shape)
print('transformed shape for data2:',pcadata2.shape)
print(pca.explained_variance_ratio_.sum())
print(pcadata3.shape)

#return dimensionality
pcadata4 = pca.inverse_transform(pcadata2)
print(pcadata4.shape)
np.savetxt("data4.csv",pcadata4.T, delimiter = ',')


#dt = pd.read_csv("data3.csv")
#genedata= pd.read_csv("/Users/hugetian/Desktop/BME440/svm_rf7.csv",header=0)
#df = genedata.replace(np.nan, 0)
#dt = df.drop(columns=['sample','class'])
#a = dt.columns

#f1 = pd.DataFrame(pcadata3.T,columns=a)
#f2 =pd.DataFrame(pcadata4.T,columns=a)
#f = f2.append(f1)
#f.to_csv("datas" + ".csv")

#print(f.shape)
#f

#RFECV steps
genedata= pd.read_csv("file.csv",header=0)
df = genedata.replace(np.nan, 0)
x = df.iloc[:, 1:156].values
y = df.iloc[:,-1 ].values
print('\nInput Dataset shape: ', x.shape,
      ' Number of features: ', x.shape[1])
print(y.shape)
svm = SVC(kernel='linear')
rfecv = RFECV(estimator=svm,
              min_features_to_select=10,
              step=1,
              cv=StratifiedKFold(2),
              scoring='accuracy',
              verbose = 0,
              ).fit(x, y)
X_RFECV = rfecv.transform(x)
print("best genes number : %d" % rfecv.n_features_)
print("levels list : %s" % list(rfecv.ranking_))

indx_n=0
op_feature_list=[]
for i in rfecv.ranking_:
    indx_n=indx_n+1
    if i == 1:
        op_feature_list.append(df.columns[indx_n])
print ('the best gene list is：',op_feature_list)


f_name = df.columns[1:].values
f_idxs = np.argsort(rfecv.ranking_)
new_f_name = np.insert(f_name[f_idxs[0:]], 0, df.columns[0])
new_df = df[new_f_name]
new_df.to_csv('geneoutput.csv', index=None)