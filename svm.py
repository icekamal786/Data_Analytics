import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ftest= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\test.csv')
ftrain= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\train.csv')

print(ftest)
print(ftrain)
x= ftrain.iloc[:, [0,1]].values  
xtest= ftest.iloc[:, [0,1]].values  
y= ftrain.iloc[:, 2].values 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state = 0)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))