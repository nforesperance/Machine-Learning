import pandas as pd
import quandl,math
from datetime import datetime
import numpy as np
from sklearn import preprocessing,model_selection,neighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace = True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

# Note: This section was used to create pickle and commented out
# so as to avoid training the model every time

# reg = LinearRegression(n_jobs=-1)
# reg.fit(X_train,y_train)
# with open ('breast_cancer.pickle','wb') as f:
#     pickle.dump(reg,f)

pickle_in = open("breast_cancer.pickle",'rb')
reg = pickle.load(pickle_in)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

example_mesures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,3,1,3,2,3,2,1]])
# example_mesures = example_mesures.reshape(1,-1)
prediction = clf.predict(example_mesures) #with single sample rember to put the param in [] or reshpae as above
print(prediction)