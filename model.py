from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import pandas as pd
import pyprind

from matplotlib import pyplot as plt

seed = 256

data = pd.read_csv("FinalData.csv").values
data = shuffle(data, random_state=seed)
print(data[:5,:5])

X = data[:,:-1]
y = data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.3)

n_splits = 10
kfold = KFold(n_splits=n_splits, random_state=seed)
model = DecisionTreeClassifier()
'''results = cross_val_score(model, X_train,y_train, cv=kfold)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
'''
model.fit(X_train, y_train)
prediction = model.predict(X_val)
report = classification_report(y_val, prediction)
print(report)


#---------------------
models = []
names = []
names = ["KNN","LogisticR","LDA", "Naive","SVM","TREE"]
models.append((names[0], KNeighborsClassifier()))
models.append((names[1], LogisticRegression()))
models.append((names[2], LinearDiscriminantAnalysis()))
models.append((names[3], GaussianNB()))
models.append((names[4], SVC()))
models.append((names[5], DecisionTreeClassifier()))
results = []

bar = pyprind.ProgBar(len(models), monitor=True)
for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
    bar.update()