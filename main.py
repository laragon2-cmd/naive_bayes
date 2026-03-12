import numpy as np
from extract_features import extract_features
from naive_bayes import NaiveBayes

X = extract_features("../TRAINING")

with open("../SPAMTrain.label") as f:
    y = np.array([int(line.split()[1]) for line in f])

model = NaiveBayes()

model.fit(X,y)

pred = model.predict(X)

error = np.mean(pred != y)

print("Misclassification rate:", error)

TP = np.sum((pred==1) & (y==1))
TN = np.sum((pred==0) & (y==0))
FP = np.sum((pred==1) & (y==0))
FN = np.sum((pred==0) & (y==1))

print("TP:",TP)
print("TN:",TN)
print("FP:",FP)
print("FN:",FN)