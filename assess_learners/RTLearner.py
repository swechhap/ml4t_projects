import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "SPANDEY343"

    def study_group(self):
        return "SPANDEY343"

    def add_evidence(self, X, Y):
        self.tree = self.build_tree(X, Y)

    def build_tree(self, X, Y):

        # Stop if node is small enough
        if X.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(Y), -1, -1]])

        # Stop if all Y values are the same
        if np.all(Y == Y[0]):
            return np.array([[-1, Y[0], -1, -1]])

        best_feature = np.random.randint(0, X.shape[1])

        split_value = np.median(X[:, best_feature])

        left_mask = X[:, best_feature] <= split_value
        right_mask = X[:, best_feature] > split_value

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.array([[-1, np.mean(Y), -1, -1]])

        left_tree = self.build_tree(
            X[left_mask],
            Y[left_mask]
        )

        right_tree = self.build_tree(
            X[right_mask],
            Y[right_mask]
        )

        # Create root node
        root = np.array([
            [best_feature, split_value, 1, left_tree.shape[0] + 1]
        ])

        return np.vstack((root, left_tree, right_tree))

    def query(self, Xtest):

        predictions = np.zeros(Xtest.shape[0])

        for i in range(Xtest.shape[0]):

            node = 0

            while self.tree[node, 0] != -1:

                feature = int(self.tree[node, 0])
                split_value = self.tree[node, 1]

                if Xtest[i, feature] <= split_value:
                    node += int(self.tree[node, 2])
                else:
                    node += int(self.tree[node, 3])

            predictions[i] = self.tree[node, 1]

        return predictions