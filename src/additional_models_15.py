"""Fifteen additional estimators; fixed candidates, validation-only selection."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (BayesianRidge, ARDRegression, HuberRegressor,
    OrthogonalMatchingPursuit, PassiveAggressiveRegressor, Ridge)
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.preprocessing import SplineTransformer
from sklearn.random_projection import GaussianRandomProjection

SEED = 20261022
MODELS = ['bayesian_ridge', 'ard_regression', 'huber_regression', 'linear_svr',
          'extra_trees', 'hist_gradient_boosting', 'adaboost', 'bagging_trees',
          'decision_tree', 'rbf_random_features', 'laplacian_nystroem',
          'spline_additive_ridge', 'orthogonal_matching_pursuit',
          'passive_aggressive', 'random_projection_ridge']

def make_model(name, candidate, p):
    c = candidate
    if name == 'bayesian_ridge':
        return BayesianRidge(max_iter=300, alpha_1=[1e-6, .01][c], lambda_1=[1e-6, .01][c])
    if name == 'ard_regression':
        return ARDRegression(max_iter=100, threshold_lambda=[1e3, 1e4][c])
    if name == 'huber_regression':
        return HuberRegressor(epsilon=[1.2, 1.5][c], alpha=10., max_iter=300, tol=1e-4)
    if name == 'linear_svr':
        return LinearSVR(C=[.01, .1][c], epsilon=.05, loss='squared_epsilon_insensitive',
                         dual=False, max_iter=3000, random_state=SEED)
    if name == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=100, max_depth=[6, 10][c],
            min_samples_leaf=80, max_features=.7, n_jobs=4, random_state=SEED)
    if name == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=[7, 15][c],
            learning_rate=.04, min_samples_leaf=100, l2_regularization=10,
            early_stopping=False, random_state=SEED)
    if name == 'adaboost':
        return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=[2, 3][c],
            min_samples_leaf=100), n_estimators=60, learning_rate=.05,
            loss='linear', random_state=SEED)
    if name == 'bagging_trees':
        return BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=[6, 10][c],
            min_samples_leaf=80), n_estimators=60, max_samples=.8, max_features=1.,
            bootstrap=True, n_jobs=4, random_state=SEED)
    if name == 'decision_tree':
        return DecisionTreeRegressor(max_depth=[3, 5][c], min_samples_leaf=100, random_state=SEED)
    if name == 'rbf_random_features':
        return make_pipeline(RBFSampler(gamma=[.1/p, 1./p][c], n_components=256,
            random_state=SEED), Ridge(alpha=10.))
    if name == 'laplacian_nystroem':
        return make_pipeline(Nystroem(kernel='laplacian', gamma=[.1/p, 1./p][c],
            n_components=128, random_state=SEED, n_jobs=4), Ridge(alpha=10.))
    if name == 'spline_additive_ridge':
        return make_pipeline(SplineTransformer(n_knots=[3, 5][c], degree=2,
            include_bias=False, extrapolation='linear'), Ridge(alpha=100.))
    if name == 'orthogonal_matching_pursuit':
        return OrthogonalMatchingPursuit(n_nonzero_coefs=[10, 30][c])
    if name == 'passive_aggressive':
        return PassiveAggressiveRegressor(C=[.001, .01][c], epsilon=.05,
            max_iter=1000, tol=1e-4, early_stopping=False, random_state=SEED,
            average=True)
    if name == 'random_projection_ridge':
        return make_pipeline(GaussianRandomProjection(n_components=[32, 64][c],
            random_state=SEED), Ridge(alpha=100.))
    raise ValueError(name)

def fit(name, candidate, x, y):
    model = make_model(name, candidate, x.shape[1])
    model.fit(x, y)
    return model

def predict(name, model, x):
    return np.asarray(model.predict(x)).ravel()
