import numpy as np
import pandas as pd
from pprint import pprint
import json
from sklearn.utils import shuffle

x1 = [5,4,2,5,1,2]
x2 = [4,2,4,1,1,2]
y=np.array([0,1,0,3,1,3])

def entropy(array):
    H = 0
    values, counts = np.unique(array, return_counts=True)
    fractions = counts.astype("float")/len(array)
    for f in fractions:
        if f!=0.0:
            H -= f*np.log2(f)
    return H

def partion(a):
    return {c: (a==c).nonzero() [0] for c in np.unique(a) }

def information_gain(x, y):
    H = entropy(y)
    values, counts = np.unique(x, return_counts = True)
    fractions = counts.astype("float")/len(x)

    for p, v in zip(fractions,values):
        H -= p* entropy(y[x==v])
    return H

print(information_gain(x1, y))

def isPure(s):
    return len(set(s)) == 1

def recursiveSplit(x,y):
    if isPure(y) or len(y)==0:
        return y

    gain = np.array([information_gain(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    if np.all(gain<1e-6):
        return y

    sets = partion(x[:, selected_attr])
    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis = 0)
        x_subset = x.take(v, axis= 0)

        res["x_%d = %d" % (selected_attr, k)] = recursiveSplit(x_subset, y_subset)
    return res

X = np.array([x1, x2]).T
print(np.column_stack((X,y)))
tree = recursiveSplit(X,y)
pprint(tree)


seed = 256

data = pd.read_csv("FinalData.csv").values
data = shuffle(data, random_state=seed)

X = data[:,:-1]
y = data[:,-1]
tree= recursiveSplit(X,y)

def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

with open('data.json','w') as fl:
    json.dump(tree, fl, default=default)





