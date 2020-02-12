import pandas as pd
import numpy as np
from sklearn import model_selection,neighbors,svm


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace = True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

example_mesures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,3,1,3,2,3,2,1]])
# example_mesures = example_mesures.reshape(1,-1)
prediction = clf.predict(example_mesures) #with single sample rember to put the param in [] or reshpae as above
print(prediction)