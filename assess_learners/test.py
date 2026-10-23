# import numpy as np
# import DTLearner as dt
# import RTLearner as rt
# import BagLearner as bl
#
# X = np.array([
#     [1.0, 10.0],
#     [2.0, 20.0],
#     [3.0, 30.0],
#     [4.0, 40.0]
# ])
#
# Y = np.array([10.0, 20.0, 30.0, 40.0])
#
# # learner = dt.DTLearner(leaf_size=1, verbose=False)
# # learner = rt.RTLearner(leaf_size=1, verbose=False)
# learner = bl.BagLearner(
#     learner=dt.DTLearner,
#     kwargs={"leaf_size": 1},
#     bags=5,
#     boost=False,
#     verbose=False
# )
# # learner.fit(X, Y)
# # print(learner.query(X))
# learner.add_evidence(X, Y)
# #
# # print("TREE:")
# # print(learner.tree)
#
# print("PREDICTIONS:")
# print(learner.query(X))
import numpy as np
import InsaneLearner as il

leaf_sizes = np.arange(1, 51)

train_rmse = np.zeros(len(leaf_sizes))
test_rmse = np.zeros(len(leaf_sizes))

for i, leaf_size in enumerate(leaf_sizes):
    learner = dt.DTLearner(
        leaf_size=leaf_size,
        verbose=False
    )

    learner.add_evidence(train_x, train_y)

    train_pred = learner.query(train_x)
    test_pred = learner.query(test_x)

    train_rmse[i] = rmse(train_y, train_pred)
    test_rmse[i] = rmse(test_y, test_pred)

print("\nDTLearner experiment")

for i, leaf_size in enumerate(leaf_sizes):
    print(
        f"leaf_size={leaf_size}: "
        f"train RMSE={train_rmse[i]:.6f}, "
        f"test RMSE={test_rmse[i]:.6f}"
    )
# X = np.array([
#     [1.0, 10.0],
#     [2.0, 20.0],
#     [3.0, 30.0],
#     [4.0, 40.0]
# ])
#
# Y = np.array([10.0, 20.0, 30.0, 40.0])
#
# learner = il.InsaneLearner(verbose=False)
#
# learner.add_evidence(X, Y)
#
# print("PREDICTIONS:")
# print(learner.query(X))