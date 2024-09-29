# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
```
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
```
## Algorithm
```
step1:start the program
step2:Import the standard libraries.
step3:Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
step4:Import LabelEncoder and encode the dataset.
step5:Import LogisticRegression from sklearn and apply the model on the dataset.
step6:Predict the values of array.
step7:Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
step8:Apply new unknown values
step9:End
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MONISH S
RegisterNumber:  212223040115
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
#load the california housing dataset
data = fetch_california_housing()
#use the first 3 features as inputs
X= data.data[:, :3] #features: 'Medinc','housage','averooms'
Y=np.column_stack((data.target,data.data[:, 6]))
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
#scale the features and target variables
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
#initialize the SGDRegressor
sgd = SGDRegressor(max_iter = 1000,tol = 1e-3)
#Use Multioutputregressor to handle multiple output varibles
multi_output_sgd = MultiOutputRegressor(sgd)
#train the model
multi_output_sgd.fit(x_train,y_train)
#predict on the test data
y_pred = multi_output_sgd.predict(x_test)
#inverse transform the prediction to get them back to the original scale
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
#evaluate the model using mean squared error
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
#optionally print some predictions
print("\npredictions:\n",y_pred[:5])
```
## Output:
```
Mean Squared Error: 2.5617706196019805
predictions: [[ 1.05253483 35.78240513]
             [ 1.49326644 35.79998002]
             [ 2.32258926 35.5349678 ]
             [ 2.71613475 35.49546887]
             [ 2.08993585 35.70601914]]
```


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
