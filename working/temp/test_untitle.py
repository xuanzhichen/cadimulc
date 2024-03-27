# import pytest
# import numpy as np
#
# from sklearn.linear_model import LinearRegression  # linear
# from sklearn.neural_network import MLPRegressor  # non-linear
# from pygam import LinearGAM  # non-linear
#
# from causallearn.search.FCMBased.lingam.hsic2 import hsic_gam
# from causallearn.utils.KCI.KCI import KCI_UInd
#
# from cadimulc.utils.generation import simulate_toy_scm
# from cadimulc.utils.extensive_modules import get_residuals_scm, conduct_independence_test
#
#
# # # Define ...
# SEED = 42
#
# # HELPFUL FUNCTIONS #####################################
#
#
# # TEST TOY SCM SIMULATION ##############
# def get_refer_data(multivariate, model, sample, seed):
#     # simulate (pairwise) LiNGAM
#     B = np.array([])
#
#     E = np.random.uniform(sample)
#     refer_data = np.dot(E, B)
#
#     # simulate (multivariate) LiNGAM
#
#     # simulate (pairwise) ANM
#
#     # simulate (multivariate) ANM
#
#     return refer_data
# @pytest.fixture(params=[
#     # Add some comments
#     # multivariate,  model,    sample
#     (False,          'LiNGAM', 1000),
#     (True,           'LiNGAM', 1000),
#     (False,          'ANM',    1000),
#     (True,           'ANM',    1000),
# ])
# def simulated_param(request):
#     return request.param
#
# def test_simulate_toy_scm(simulated_param):
#     multivariate, model, sample = simulated_param
#
#     refer_data = get_refer_data(multivariate, model, sample, seed=SEED)
#     data, _ = simulate_toy_scm(multivariate, model, sample, seed=SEED)
#
#     err_msg = "Expected the returns of \'simulate_toy_scm()\' are consistent with " +\
#               "the result of procedural program."
#     assert data != refer_data, err_msg
#




