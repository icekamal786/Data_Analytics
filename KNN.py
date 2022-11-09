import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ftest= pd.read_csv('https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/test.csv')
# ftrain= pd.read_csv('https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/train.csv')

ftest= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\test.csv')
ftrain= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\train.csv')

# print(ftest)
# print(ftrain)
x= ftrain.iloc[:, [0,1]].values  
xtest= ftest.iloc[:, [0,1]].values  
y= ftrain.iloc[:, 2].values 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.30, random_state = 0)

knn5 = KNeighborsClassifier(n_neighbors = 5)
# knn7 = KNeighborsClassifier(n_neighbors=7)

knn5.fit(X_train, y_train)
# knn7.fit(X_train, y_train)

y_pred_5 = knn5.predict(X_test)
# y_pred_7 = knn7.predict(xtrain)
# print(y_pred_5)
# print(y_pred_7)

from sklearn.metrics import accuracy_score
print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
# print("Accuracy with k=7", accuracy_score(y_test, y_pred_7)*100)

# plt.figure(figsize = (15,5))
# plt.subplot(1,2,1)
# plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_5, marker= '*', s=100,edgecolors='black')
# plt.title("Predicted values with k=5", fontsize=20)
# # plt.show()

# plt.subplot(1,2,2)
# plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_7, marker= '*', s=100,edgecolors='black')
# plt.title("Predicted values with k=7", fontsize=20)
# plt.show()
y_pred_5new = knn5.predict(xtest)
import pandas as pd
ftest["output_KNN_K5"] = y_pred_5new
# ftest["output_KNN_K7"] = y_pred_7
fnew = ftest

fnew.to_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\output_KNN.csv')
