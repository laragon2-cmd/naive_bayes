import numpy as np

class NaiveBayes:

    def fit(self, X, y):

        self.labels = np.unique(y)
        n,p = X.shape

        self.priors = {}
        self.cond_prob = {}

        for label in self.labels:

            X_class = X[y == label]

            self.priors[label] = len(X_class) / n

            self.cond_prob[label] = (X_class.sum(axis=0) + 1) / (len(X_class) + 2)


    def predict(self, X):

        predictions = []

        for x in X:

            scores = {}

            for label in self.labels:

                log_prob = np.log(self.priors[label])

                cond = self.cond_prob[label]

                log_prob += np.sum(
                    x*np.log(cond) + (1-x)*np.log(1-cond)
                )

                scores[label] = log_prob

            predictions.append(max(scores, key=scores.get))

        return np.array(predictions)