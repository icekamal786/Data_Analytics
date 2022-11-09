import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ftest= pd.read_csv('C:\\Users\\admin\\Desktop\\pharmaPRO\\IIT\\test.csv')
ftrain= pd.read_csv('C:\\Users\\admin\\Desktop\\pharmaPRO\\IIT\\train.csv')
xtrain= ftrain.iloc[:, [0,1]].values  
ytrain= ftrain.iloc[:, 2].values 
xtest= ftest.iloc[:, [0,1]].values 

X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size= 0.03205, random_state = 0)

x = xtrain
y = ytrain

model = LogisticRegression(solver='liblinear', random_state=0)

model.fit(x, y)

a = model.classes_
b = model.intercept_
c = model.coef_
# print(a,b,c)

model.predict(xtest)

score = model.score(x, y)
print(score)

confusion_matrix(y, model.predict(x))

cm = confusion_matrix(y, model.predict(x))

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

print(classification_report(y, model.predict(x)))