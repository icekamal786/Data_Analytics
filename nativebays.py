# load the iris dataset
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

ftest= pd.read_csv('https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/test.csv')
ftrain= pd.read_csv('https://raw.githubusercontent.com/icekamal786/Data_Analytics/main/train.csv')

# ftest= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\test.csv')
# ftrain= pd.read_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\train.csv')

# store the feature matrix (X) and response vector (y)

x= ftrain.iloc[:, [0,1]].values  
y= ftrain.iloc[:, 2].values 
xtest= ftest.iloc[:, [0,1]].values 

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)
  
# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred = gnb.predict(X_test)
  
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)

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

# making predictions on the testing set
y_pred_test = gnb.predict(xtest)


import pandas as pd
ftest["output_NativeB"] = y_pred_test
fnew = ftest
print(fnew)
fnew.to_csv('C:\\Users\\admin\\Desktop\\GIT\\OPEN IIT\\Data_Analytics\\output_NB.csv')