from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#importing data
ftest = pd.read_csv(
    'https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/test.csv')
ftrain = pd.read_csv(
    'https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/train.csv')

# ftest= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\test.csv')
# ftrain= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\train.csv')

# Train test split 
x = ftrain.iloc[:, [0, 1]].values
xtest = ftest.iloc[:, [0, 1]].values
y = ftrain.iloc[:, 2].values
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# KNN model
knn5 = KNeighborsClassifier(n_neighbors=5)

# KNN fitting 
knn5.fit(X_train, y_train)

# KNN prediction
y_pred_5 = knn5.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_5)

# Accuracy score with K=5
print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)

# Plotting confusion matrix
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

# Plotting of predicted values 
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_5, marker= '*', s=100,edgecolors='black')
plt.title("Predicted values with k=5", fontsize=20)
plt.show()

# Exporting the values in csv file
y_pred_5new = knn5.predict(xtest)
ftest["output_KNN_K5"] = y_pred_5new
fnew = ftest
fnew.to_csv(
    'C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\output_KNN.csv')
