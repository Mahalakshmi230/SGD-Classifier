# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data
2. Split Dataset into Training and Testing Sets 
3. Train the Model using Stochastic Gradient Descent (SGD)
4. Make predictions and evaluate accuracy
5. Generate confusion matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MAHALAKSHMI R
RegisterNumber: 212223230116  
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#LOAD THE IRIS DATASET
iris = load_iris()

#creat a pandas dataframes
df= pd.DataFrame(data = iris.data,columns = iris.feature_names)
df['target'] = iris.target

#display the few rows of the dataset
print(df.head())

#split the data into features (x) and target(y)
X = df.drop('target', axis = 1)
y = df['target']

#split the
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#Create an SGD classifier with defualt parameters
sgd_clf = SGDClassifier(max_iter = 1000,tol = 1e-3)

#train the classifier on training data
sgd_clf.fit(X_train,Y_train)

#make predictions on testing data
y_pred = sgd_clf.predict(X_test)

#evaluate the classifier accuracy
accuracy = accuracy_score(Y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

#calculate the confusion matrix
cm = confusion_matrix(Y_test,y_pred)
print("Confusion matrix:")
print(cm)
```

## Output:
![Screenshot 2024-09-20 131504](https://github.com/user-attachments/assets/8f297674-357c-4b5c-a1dd-3a5771b85453)
![Screenshot 2024-09-20 131511](https://github.com/user-attachments/assets/c2d33602-bfa5-44dd-b26c-4aad46507ef3)

![Screenshot 2024-09-20 131517](https://github.com/user-attachments/assets/9de24399-fbed-4377-84fb-3fb9de373e18)

![Screenshot 2024-09-20 131524](https://github.com/user-attachments/assets/bfab7e44-c22c-4b90-9e03-9305ba3ef65d)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
