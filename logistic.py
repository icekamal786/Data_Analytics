import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ftest= pd.read_csv('https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/test.csv')
ftrain= pd.read_csv('https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/train.csv')

# ftest= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\test.csv')
# ftrain= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\train.csv')

x= ftrain.iloc[:, [0,1]].values  
y= ftrain.iloc[:, 2].values 
xtest= ftest.iloc[:, [0,1]].values 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state = 0)


model = LogisticRegression()

model.fit(X_train, y_train)

# a = model.classes_
# b = model.intercept_
# c = model.coef_
# print(a,b,c)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, model.predict(X_test))*100)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

y_pred_test = model.predict(xtest)


import pandas as pd
ftest["output_logi"] = y_pred_test
fnew = ftest
print(fnew)
fnew.to_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\output_logistic.csv')