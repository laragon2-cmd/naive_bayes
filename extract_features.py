import os
import numpy as np

keywords = [
    "free","money","offer","click","win","prize","buy",
    "urgent","cash","credit","loan","limited","deal",
    "winner","cheap","discount","bonus","trial",
    "viagra","explosive"
]

def extract_features(folder):

    X = []

    files = sorted(os.listdir(folder))

    for file in files:

        with open(os.path.join(folder,file), encoding="latin1") as f:
            text = f.read().lower()

        row = []

        for word in keywords:
            row.append(1 if word in text else 0)

        X.append(row)

    return np.array(X)
