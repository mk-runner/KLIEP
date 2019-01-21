# KLIEP
A density ratio estimator package for python using the KLIEP algorithm.

The DensityRatioEstimator class implements the Kullback-Leibler Importance Estimation Procedure by Sugiyama et al. Estimator uses likelihood cross validation (LCV) to tune the num_params and sigma parameters.

# Usage
from kliep import KLIEP
model = KLIEP()
weight, _ = model.fit(X_train.T, X_test.T)
