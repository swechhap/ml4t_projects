""""""  		  	   		 		  		  		  		    	 		 		   		 		  
"""  		  	   		 		  		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 		  		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 		  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 		  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 		  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 		  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 		  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 		  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 		  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 		  		  		  		    	 		 		   		 		  
or edited.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 		  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 		  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 		  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Student Name: Swechha Pandey (replace with your name)  		  	   		 		  		  		  		    	 		 		   		 		  
GT User ID: spandey343 (replace with your User ID)  		  	   		 		  		  		  		    	 		 		   		 		  
GT ID: 904262734 (replace with your GT ID)  		  	   		 		  		  		  		    	 		 		   		 		  
"""  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import warnings  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		 		  		  		  		    	 		 		   		 		  
    your own correct DTLearner from Project 3.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		 		  		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		 		  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 		  		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 		  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		 		  		  		  		    	 		 		   		 		  
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 		  		  		  		    	 		 		   		 		  
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None


    def author(self):
            """
            :return: The GT username of the student
            :rtype: str
            """
            return "spandey343"

    def study_group(self):
        return "spandey343"
  		  	   		 		  		  		  		    	 		 		   		 		  
    def add_evidence(self, X,Y):
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		 		  		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 		  		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        """

        self.tree = self.build_tree(X, Y)

    def build_tree(self, X, Y):
        # Leaf: few enough samples left
        if X.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(Y), -1, -1]])

        # Leaf: every Y is identical, nothing left to learn
        if np.all(Y == Y[0]):
            return np.array([[-1, Y[0], -1, -1]])

        # Absolute correlation of each feature with Y
        correlations = np.zeros(X.shape[1])
        y_std = np.std(Y)
        for i in range(X.shape[1]):
            x = X[:, i]
            x_std = np.std(x)
            if x_std == 0 or y_std == 0:
                correlations[i] = 0
            else:
                correlation = np.mean((x - np.mean(x)) * (Y - np.mean(Y))) / (x_std * y_std)
                correlations[i] = abs(correlation)

        # Try features from most to least correlated and use the first one whose
        # median actually divides the data. (If a feature has many repeated values,
        # its median can equal its max, leaving the right side empty.)
        best_feature = None
        for f in np.argsort(-correlations):
            split_value = np.median(X[:, f])
            left_mask = X[:, f] <= split_value
            right_mask = ~left_mask
            if left_mask.any() and right_mask.any():
                best_feature = f
                break

        # No feature can split the data: make a leaf
        if best_feature is None:
            return np.array([[-1, np.mean(Y), -1, -1]])

        left_tree = self.build_tree(X[left_mask], Y[left_mask])
        right_tree = self.build_tree(X[right_mask], Y[right_mask])

        root = np.array([[best_feature, split_value, 1, left_tree.shape[0] + 1]])

        return np.vstack((root, left_tree, right_tree))

    def query(self, Xtest):
        """  		  	   		 		  		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 		  		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 		  		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 		  		  		  		    	 		 		   		 		  
        """
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
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 		  		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		 		  		  		  		    	 		 		   		 		  
