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
import random
from collections import Counter
import warnings

def knn(data,predict,k=3):
    if len(data)>=k:
        warnings.warn("K less that voting groups")
    distances = []
    for group in data: 
        for pt in data[group]:
            euclidean_distance =  np.linalg.norm(np.array(pt)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = (Counter(votes).most_common(1)[0][1])/k


    return vote_result,confidence

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace = True)

# convert to float
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total =0
for group in test_set:
    for data in test_set[group]:
        vote, confidence = knn(train_set,data,k=5)
        if group == vote:
            correct +=1
        total +=1
print("Accuracy:",correct/total)


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
