from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset ={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}

new_faatures = [5,7]


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

    return vote_result
print(knn(dataset,new_faatures,k=3))

[[plt.scatter(ii[0],ii[1],s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_faatures[0],new_faatures[1],s=100, color='blue')
plt.show()