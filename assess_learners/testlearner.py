""""""  		  	   		 		  		  		  		    	 		 		   		 		  
"""  		  	   		 		  		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
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
"""  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import math
import sys
import time
import numpy as np
import DTLearner as dt
import matplotlib.pyplot as plt
import RTLearner as rt
import BagLearner as bl

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "spandey343"


def study_group():
    return "spandey343"

def rmse(actual, predicted):
    return math.sqrt(np.mean((actual - predicted) ** 2))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    data = np.genfromtxt(
        sys.argv[1],
        delimiter=",",
        skip_header=1
    )
    np.random.seed(904262734)
    # Ignore date column
    data = data[:, 1:]

    # Random 60/40 train/test split
    indices = np.random.permutation(data.shape[0])

    train_rows = int(0.6 * data.shape[0])

    train_indices = indices[:train_rows]
    test_indices = indices[train_rows:]

    train_x = data[train_indices, :-1]
    train_y = data[train_indices, -1]

    test_x = data[test_indices, :-1]
    test_y = data[test_indices, -1]

    print(f"Test X shape: {test_x.shape}")
    print(f"Test Y shape: {test_y.shape}")

    # ---------------------------------------------------------
    # DTLEARNER EXPERIMENT
    # ---------------------------------------------------------

    leaf_sizes = np.arange(1, 51)

    dt_train_rmse = np.zeros(len(leaf_sizes))
    dt_test_rmse = np.zeros(len(leaf_sizes))
    dt_train_time = np.zeros(len(leaf_sizes))

    for i, leaf_size in enumerate(leaf_sizes):

        learner = dt.DTLearner(
            leaf_size=leaf_size,
            verbose=False
        )

        start = time.perf_counter()

        learner.add_evidence(train_x, train_y)

        dt_train_time[i] = time.perf_counter() - start

        train_pred = learner.query(train_x)
        test_pred = learner.query(test_x)

        dt_train_rmse[i] = rmse(train_y, train_pred)
        dt_test_rmse[i] = rmse(test_y, test_pred)

    print("\nDTLearner experiment")

    for i, leaf_size in enumerate(leaf_sizes):
        print(
            f"leaf_size={leaf_size}: "
            f"train RMSE={dt_train_rmse[i]:.6f}, "
            f"test RMSE={dt_test_rmse[i]:.6f}, "
            f"time={dt_train_time[i]:.6f}"
        )

    # DT RMSE plot
    plt.figure()

    plt.plot(
        leaf_sizes,
        dt_train_rmse,
        label="In-sample RMSE"
    )

    plt.plot(
        leaf_sizes,
        dt_test_rmse,
        label="Out-of-sample RMSE"
    )

    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Experiment 1: DTLearner Overfitting vs Leaf Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_rmse_vs_leaf_size.png")
    plt.close()

    # DT training time plot
    plt.figure()

    plt.plot(
        leaf_sizes,
        dt_train_time
    )

    plt.xlabel("Leaf Size")
    plt.ylabel("Training Time (seconds)")
    plt.title("DTLearner: Training Time vs Leaf Size")
    plt.grid(True)
    plt.savefig("dt_training_time_vs_leaf_size.png")
    plt.close()

    # ---------------------------------------------------------
    # RTLEARNER EXPERIMENT
    # ---------------------------------------------------------

    rt_train_rmse = np.zeros(len(leaf_sizes))
    rt_test_rmse = np.zeros(len(leaf_sizes))
    rt_train_time = np.zeros(len(leaf_sizes))

    for i, leaf_size in enumerate(leaf_sizes):

        learner = rt.RTLearner(
            leaf_size=leaf_size,
            verbose=False
        )

        start = time.perf_counter()

        learner.add_evidence(train_x, train_y)

        rt_train_time[i] = time.perf_counter() - start

        train_pred = learner.query(train_x)
        test_pred = learner.query(test_x)

        rt_train_rmse[i] = rmse(train_y, train_pred)
        rt_test_rmse[i] = rmse(test_y, test_pred)

    print("\nRTLearner experiment")

    for i, leaf_size in enumerate(leaf_sizes):
        print(
            f"leaf_size={leaf_size}: "
            f"train RMSE={rt_train_rmse[i]:.6f}, "
            f"test RMSE={rt_test_rmse[i]:.6f}, "
            f"time={rt_train_time[i]:.6f}"
        )

    # RT RMSE plot
    plt.figure()

    plt.plot(
        leaf_sizes,
        rt_train_rmse,
        label="In-sample RMSE"
    )

    plt.plot(
        leaf_sizes,
        rt_test_rmse,
        label="Out-of-sample RMSE"
    )

    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("RTLearner: RMSE vs Leaf Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("rt_rmse_vs_leaf_size.png")
    plt.close()

    # RT training time plot
    plt.figure()

    plt.plot(
        leaf_sizes,
        rt_train_time
    )

    plt.xlabel("Leaf Size")
    plt.ylabel("Training Time (seconds)")
    plt.title("RTLearner: Training Time vs Leaf Size")
    plt.grid(True)
    plt.savefig("rt_training_time_vs_leaf_size.png")
    plt.close()

    # ---------------------------------------------------------
    # BAGGED DTLEARNER EXPERIMENT
    # Fixed number of bags, varying leaf size
    # ---------------------------------------------------------

    bags = 20

    bag_dt_train_rmse = np.zeros(len(leaf_sizes))
    bag_dt_test_rmse = np.zeros(len(leaf_sizes))


    for i, leaf_size in enumerate(leaf_sizes):
        learner = bl.BagLearner(
            learner=dt.DTLearner,
            kwargs={"leaf_size": leaf_size},
            bags=bags,
            boost=False,
            verbose=False
        )

        learner.add_evidence(train_x, train_y)

        train_pred = learner.query(train_x)
        test_pred = learner.query(test_x)

        bag_dt_train_rmse[i] = rmse(train_y, train_pred)
        bag_dt_test_rmse[i] = rmse(test_y, test_pred)

    print("\nBagged DTLearner experiment")

    for i, leaf_size in enumerate(leaf_sizes):
        print(
            f"leaf_size={leaf_size}: "
            f"train RMSE={bag_dt_train_rmse[i]:.6f}, "
            f"test RMSE={bag_dt_test_rmse[i]:.6f}"
        )

    # Bagged DT RMSE plot
    plt.figure()

    plt.plot(
        leaf_sizes,
        bag_dt_train_rmse,
        label="In-sample RMSE"
    )

    plt.plot(
        leaf_sizes,
        bag_dt_test_rmse,
        label="Out-of-sample RMSE"
    )


    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title(f"Experiment 2: Bagged DTLearner Overfitting vs Leaf Size")
    plt.legend()
    plt.grid(True)

    plt.savefig("bagged_dt_rmse_vs_leaf_size.png")
    plt.close()

    # ---------------------------------------------------------
    # EXPERIMENT 3: DTLEARNER VS RTLEARNER
    # Training time and tree size
    # ---------------------------------------------------------

    comparison_leaf_sizes = np.array([1, 5, 10, 20, 30, 40, 50])

    dt_times = np.zeros(len(comparison_leaf_sizes))
    rt_times = np.zeros(len(comparison_leaf_sizes))

    dt_nodes = np.zeros(len(comparison_leaf_sizes))
    rt_nodes = np.zeros(len(comparison_leaf_sizes))

    print("\nExperiment 3: DTLearner vs RTLearner")

    for i, leaf_size in enumerate(comparison_leaf_sizes):
        # -------------------------
        # DTLearner
        # -------------------------
        learner = dt.DTLearner(
            leaf_size=leaf_size,
            verbose=False
        )

        start = time.perf_counter()

        learner.add_evidence(train_x, train_y)

        dt_times[i] = time.perf_counter() - start
        dt_nodes[i] = learner.tree.shape[0]

        # -------------------------
        # RTLearner
        # -------------------------
        learner = rt.RTLearner(
            leaf_size=leaf_size,
            verbose=False
        )

        start = time.perf_counter()

        learner.add_evidence(train_x, train_y)

        rt_times[i] = time.perf_counter() - start
        rt_nodes[i] = learner.tree.shape[0]

        print(
            f"leaf_size={leaf_size}: "
            f"DT time={dt_times[i]:.6f}, "
            f"RT time={rt_times[i]:.6f}, "
            f"DT nodes={int(dt_nodes[i])}, "
            f"RT nodes={int(rt_nodes[i])}"
        )

    # Experiment 3 - Training Time
    plt.figure()

    plt.plot(
        comparison_leaf_sizes,
        dt_times,
        marker="o",
        label="DTLearner"
    )

    plt.plot(
        comparison_leaf_sizes,
        rt_times,
        marker="o",
        label="RTLearner"
    )

    plt.xlabel("Leaf Size")
    plt.ylabel("Training Time (seconds)")
    plt.title("Experiment 3: Training Time Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig("experiment3_training_time.png")
    plt.close()

    # Experiment 3 - Tree Size
    plt.figure()

    plt.plot(
        comparison_leaf_sizes,
        dt_nodes,
        marker="o",
        label="DTLearner"
    )

    plt.plot(
        comparison_leaf_sizes,
        rt_nodes,
        marker="o",
        label="RTLearner"
    )

    plt.xlabel("Leaf Size")
    plt.ylabel("Number of Tree Nodes")
    plt.title("Experiment 3: Tree Size Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig("experiment3_tree_size.png")
    plt.close()