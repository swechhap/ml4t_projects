import numpy as np


class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []

        for _ in range(self.bags):
            self.learners.append(
                learner(**kwargs)
            )

    def author(self):
        return "SPANDEY343"

    def study_group(self):
        return "SPANDEY343"

    def add_evidence(self, X, Y):

        n = X.shape[0]

        for learner in self.learners:

            indices = np.random.choice(
                n,
                size=n,
                replace=True
            )

            X_sample = X[indices]
            Y_sample = Y[indices]

            learner.add_evidence(X_sample, Y_sample)

    def query(self, Xtest):

        predictions = np.zeros((self.bags, Xtest.shape[0]))

        for i, learner in enumerate(self.learners):
            predictions[i] = learner.query(Xtest)

        return np.mean(predictions, axis=0)