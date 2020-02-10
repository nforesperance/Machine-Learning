# Simple Lineair Regression with 
# Cross Validation
# Training a model on part of a dataset and testing on another part
import pandas as pd;
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing


# loading and trining sample model
# here our X is 1 dimension

data = pd.read_table("eucalyptus.txt",header=0,sep =" ")
reg = LinearRegression(n_jobs=-1)
y= data["ht"]



# Incressing the dimension of X to 4 (3 basically the first is just ones)
# Creatind X = 1430 by 3 Matrix
X = np.c_[np.sqrt(data["circ"]), data["circ"], np.square(data["circ"])]
X = preprocessing.scale(X)

# Dividing data for cross validation
#Training Dataset
X_train = X[:-700]
y_train = y[:-700]

#Testing Dataset
X_test = X[-700:]
y_test = y[-700:]

# Train the model using the training sets
reg.fit(X_train,y_train)
print("\n  [b1, b2, b3]\n")
print(reg.coef_)
print(reg.score(X_test,y_test))
