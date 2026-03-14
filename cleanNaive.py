import os
import numpy as np
from sklearn.model_selection import train_test_split


#features
keywords = [
    "free","money","offer","click","win","prize","buy",
    "urgent","cash","credit","loan","limited","deal",
    "winner","cheap","discount","bonus","trial",
    "viagra","explosive"
]

#extracting keywords function
def extract_features(folder):
    X = []
    files = sorted(os.listdir(folder))

    for file in files:
        if not file.endswith(".eml"):
            continue

        with open(os.path.join(folder, file), encoding="latin1") as f:
            text = f.read().lower()

        row = []

        for word in keywords:
            row.append(1 if word in text else 0)

        X.append(row)

    return np.array(X)


# Step 1 - Download SPAMTrain.label, TRAINING.zip, and TESTING.zip from Canvas
# Step 2 - Put these folders somewhere on your computer and point to them below

training_folder = "/Users/laragon/Downloads/TRAINING 2"
labels_file = "/Users/laragon/Downloads/SPAMTrain.label"

print("Training folder path:", training_folder)
print("Label file path:", labels_file)
print("Training folder exists:", os.path.exists(training_folder))
print("Label file exists:", os.path.exists(labels_file))
print()

# Create the binary feature matrix X from the training emails
X = extract_features(training_folder)

# Load the training labels
# According to the dataset readme:
# 1 = HAM
# 0 = SPAM
# The label file has the label in the first column
Y = np.loadtxt(labels_file, usecols=0, dtype=int)

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print("First 10 labels:", Y[:10])
print()

# Split the labeled training data into train/test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# i. Prior probabilities for each class
pi = np.array([
    np.mean(Y_train == 0),   # SPAM
    np.mean(Y_train == 1)    # HAM
])

print("Prior probabilities:", pi)
print()

# ii. Class conditional probabilities P(x_k = 1 | Y = j)
class_cond_probs = np.zeros((len(pi), X_train.shape[1]))

for k in range(len(pi)):
    class_count = np.sum(Y_train == k)

    for j in range(X_train.shape[1]):
        class_cond_probs[k, j] = np.sum(X_train[Y_train == k, j]) / class_count

print("Class conditional probabilities:")
print("P(X_k = 1 | Y = 0):", class_cond_probs[0])
print("P(X_k = 1 | Y = 1):", class_cond_probs[1])
print()

def naive_bayes_classifier(x, class_cond_probs, pi):
    K = len(pi)
    features = len(x)
    numerators = np.zeros(K)

    for k in range(K):
        density_f_kp = 1

        for j in range(features):
            p_hat = class_cond_probs[k, j]

            if x[j] == 1:
                density_f_kp *= p_hat
            else:
                density_f_kp *= (1 - p_hat)

        numerators[k] = pi[k] * density_f_kp

    denominator = np.sum(numerators)

    if denominator == 0:
        posterior = np.zeros(K)
    else:
        posterior = numerators / denominator

    predicted_class = np.argmax(posterior)

    return posterior, predicted_class

# iii. Posterior probabilities and predicted class for the first observation
posterior, predicted_class = naive_bayes_classifier(X_train[0], class_cond_probs, pi)

print("Posterior probabilities and predicted class for first observation:")
print(posterior)
print(predicted_class)
print()

# iv. Predicted class for every observation in the training set
predicted_classe_for_each_obs = np.array([
    naive_bayes_classifier(X_train[i], class_cond_probs, pi)[1]
    for i in range(X_train.shape[0])
])

print("Predicted class for every observation in training set:")
print(predicted_classe_for_each_obs)
print()

# v. Misclassification rate on the training set
nbayes_error_trainig = np.mean(Y_train != predicted_classe_for_each_obs)
print("Naive Bayes Error on training set:", nbayes_error_trainig)
print()

# v.i. Predicted class for every observation in the testing split
predicted_class_testing = np.array([
    naive_bayes_classifier(X_test[i], class_cond_probs, pi)[1]
    for i in range(X_test.shape[0])
])

print("Predicted class on testing split:")
print(predicted_class_testing)
print()

print("The actual classes on testing split:")
print(Y_test)
print()

# Misclassification rate on the testing split
nbayes_error_test = np.mean(Y_test != predicted_class_testing)
print("Naive Bayes Error on testing split:", nbayes_error_test)
print()

# vi. Consensus matrix: TP, TN, FP, FN
# Here we treat SPAM = 0 as the positive class
TP = np.sum((predicted_class_testing == 0) & (Y_test == 0))
TN = np.sum((predicted_class_testing == 1) & (Y_test == 1))
FP = np.sum((predicted_class_testing == 0) & (Y_test == 1))
FN = np.sum((predicted_class_testing == 1) & (Y_test == 0))

print("Consensus matrix counts:")
print("TP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)
