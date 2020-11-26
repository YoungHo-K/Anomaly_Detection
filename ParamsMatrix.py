import numpy as np

start_prob = np.array([1.0, 0.0])

trans_prob = np.array([[0.2, 0.8],
                       [0.8, 0.2]])

mean_array = np.zeros(27)
covar_array = np.ones(27)

mean_ = np.array([mean_array, mean_array])

covar_ = np.array([[1, 1]]).T


# covariance parameter per state
# model.covars_ = np.array( [ 1,2,3 ] ).T  # spherical
# model.covars_ = np.array( [ [1,1],[1,2],[1,1] ] )  # diag
# model.covars_ = np.tile(np.identity(2), (3, 1, 1)) # covariance_type ="full"
# model.covars_ = np.array( [ [1,0], [0,1 ] ] )  # tied (p.s.d.m)
