"""Write down some descriptions here."""

# ### DEVELOPMENT NOTES (LEAST) ############################################
# * Test-0x1 (Empirical Conclusions):
#   * Variance of parameter settings is limited by sample size as I realized
#     the unacceptable running time of non-linear independence test.
#   * Non-linear and Gaussian generative settings are always more preferable
#     than the non-Gaussian ones, except for their costly running time.
#   * Relatively substitute the non-linear independence test as the linear
#     one, seeming not that wholly compromising the final performance.
#
# * Test for get_skeleton_from_pc() would involve
#   * utils/evaluation, causality_instruments (get_skeleton_score/Evaluator)
#   * the evaluation result served for the consequent stage-testing
#   * usage recommendations about parameters
#
# * Test for get_skeleton_from_pc() (including interaction of exp helper)
#   will mark the end of the first-stage project programming.


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Improve the test performance of get_skeleton_from_pc(), together coding
#   with assistance of skeleton-testing methods.              18th.Jan, 2024
#
# * Nearly complete the test for get_skeleton_from_pc().      29th.Dec, 2023
#
# * Finish test 'test_pair_causal_procedure()'.               19th.Dec, 2023
#
# * Build up the testing framework by adding 'causality instruments' module,
#   predicting next steps involve combining both the main-line and branch.
#                                                             27th.Dec, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# None

# Done:
# TODO: With the available Evaluator, add skeleton conversion function, then
#       evaluate performance of get_skeleton_from_pc(): vary 'ind_test_type',
#       record 'running_time'.
# TODO: Program Evaluator test framework (the last primary project module).


# testing modules
from cadimulc.utils.causality_instruments import (
    get_skeleton_from_pc,
    get_residuals_scm,
    conduct_ind_test,
)


# basic modules
from cadimulc.utils.extensive_modules import (
    display_test_section_symbols,
    get_skeleton_from_adjmat,
    check_1dim_array,
    copy_and_rename,
)
from cadimulc.utils.generation import Generator
from cadimulc.utils.evaluation import Evaluator
from cadimulc.utils.visualization import draw_graph_from_ndarray

from causallearn.utils.cit import kci, fisherz
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased.lingam.hsic2 import hsic_gam
from causallearn.utils.KCI.KCI import KCI_UInd

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from pygam import LinearGAM

import pytest
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# #########################################################################
# ### CONVENIENCE FUNCTION(S) #############################################
# #########################################################################


def sample_from_ndarray(ndarray, size):
    """ Return the ndarray after sampling
        by temporarily transferring it to the dataframe.
    """

    df = pd.DataFrame(data=ndarray)
    df.sample(n=size)

    return df.values

# def display_0x1_1(
#         main_info_1,
#         main_info_2,
#         main_info_3,
#         main_info_4,
#         main_info_5,
#         main_info_6,
# ):
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return


# #########################################################################
# ### AUXILIARY FUNCTION(S) ###############################################
# #########################################################################


# ### SUBORDINATE COMPONENT(S) ############################################
# test_pair_causal_procedure()
def check_get_residuals_scm(
        explanatory_data,
        explained_data,
        regressor,
        random_seed
):
    """
    Write down some descriptions here.

    Parameters
    ----------
    explanatory_data : ndarray (shape[n, d])
    explained_data : ndarray (shape[n, 1])
    regressor : class
    random_seed : int

    Return
    ------
    residuals : ndarray (shape[n, 1])
    """

    np.random.seed(random_seed)

    residuals = get_residuals_scm(
        explanatory_data=explanatory_data,
        explained_data=explained_data,
        regressor=regressor
    )

    return residuals


# ### SUBORDINATE COMPONENT(S) ############################################
# test_pair_causal_procedure()

def check_conduct_ind_test(
        explanatory_data,
        residuals,
        ind_test_method,
        random_seed
):
    """
    Write down some descriptions here.

    Parameters
    ----------
    explanatory_data : ndarray (shape[n, d])
    residuals : ndarray (shape[n, 1])
    ind_test_method : string
    random_seed : int

    Return
    ------
    p_value : float
    """

    np.random.seed(random_seed)

    p_value = conduct_ind_test(
        explanatory_data=explanatory_data,
        residuals=residuals,
        ind_test_method=ind_test_method
    )

    return p_value


# ### SUBORDINATE COMPONENT(S) ############################################
# test_pair_causal_procedure()

def get_pair_cause_effect(ground_truth, data):
    """
    Split raw data into the explanatory and the explained parts,
    simply for associating ground truth of pairwise variables.
    """

    if not np.array_equal(ground_truth, ground_truth.T):
        if ground_truth[1][0] == 1 and ground_truth[0][1] == 0:
            explanatory_data = data[:, 0]
            explained_data = data[:, 1]
        else:
            explanatory_data = data[:, 1]
            explained_data = data[:, 0]
    else:
        raise ValueError("The ground-truth graph might be trivial or undirected")

    return explanatory_data, explained_data


# #########################################################################
# ### TESTING SECTION #####################################################
# #########################################################################


# Notes for developer: See file header about the current parameter setting.
@pytest.fixture(params=[
    # Parameters for testing PC-skeleton-learning performance.
    # node,  sample,  causal_model,        noise_type,     ind_test_type

    # theoretical setting (rule out linear Gaussian setting)
    (10,     1000,    'lingam',           'non-Gaussian', 'linear'),
    (10,     1000,    'hybrid_nonlinear', 'Gaussian',     'non_linear'),
    (10,     1000,    'hybrid_nonlinear', 'non-Gaussian', 'non_linear'),

    # empirical setting
    (10,     1000,    'hybrid_nonlinear', 'Gaussian',     'linear'),
    (10,     1000,    'hybrid_nonlinear', 'non-Gaussian', 'linear'),
])
def skeleton_generation_param_0x1(request):
    return request.param


SEED_0x1 = 42
# SEED_0x1 = 42 + 50
# SEED_0x1 = 42 + 100

# Notice the unacceptable running time of non-linear independence test.
# REPETITIONS_0x1 = 10


# ### CORRESPONDING TEST ###################################################
# Loc: causality_instruments >> get_skeleton_from_pc

# ### AUXILIARY COMPONENT(S) ##############################################
# Testing Date: 2024-01-18 | **:** (pass)
# Testing Date: 2024-02-02 | 14:03 (update)

def test_0x1_get_skeleton_from_pc(skeleton_generation_param_0x1):
    """
    Testing for causal skeleton learning by using the stable-PC algorithm:

    * performance relative to combinations of causal model and associating noise;
    * computational cost relative to different combinations;
    * empirical outcomes of substituting non-linear ind-test with linear ind-test;
    """

    np.random.seed(SEED_0x1)
    random.seed(SEED_0x1)

    (graph_node_num, sample,
     causal_model, noise_type, ind_test_type) = skeleton_generation_param_0x1

    # temporal code fragment for initial testing
    # sample = int(sample / 10)

    # === SEPARATION ======================================================

    # Randomly generate simulated causal model based on specific combinations
    # of causal model and associating noise.
    generator = Generator(
        graph_node_num=graph_node_num,
        sample=sample,
        causal_model=causal_model,
        noise_type=noise_type
    )
    ground_truth, data = generator.run_generation_procedure().unpack()
    skeleton_expected = get_skeleton_from_adjmat(ground_truth)

    # temporal code for the simple test of get_skeleton_from_adjmat()
    assert np.array_equal(skeleton_expected, skeleton_expected.T) is True

    # Reduce sample scale in theoretical setting partially due to
    # the computational cost of non-linear independence test.
    if (causal_model == 'hybrid_nonlinear') and (ind_test_type == 'non_linear'):
        data = sample_from_ndarray(data, size=int(sample / 10))

    # Conduct get_skeleton_from_pc().
    skeleton_actual, running_time = get_skeleton_from_pc(
        data=data,
        ind_test_type=ind_test_type
    )

    # Evaluate the estimated skeleton.
    f1_score_skeleton = Evaluator.evaluate_skeleton(
        true_skeleton=skeleton_expected,
        est_skeleton=skeleton_actual,
        metric='F1-score'
    )

    # === SEPARATION ======================================================

    display_test_section_symbols()

    # display_0x1(
    #     main_info_1=f1_score_skeleton,
    #     main_info_2=running_time,
    #     main_info_3=graph_node_num,
    #     main_info_4=sample,
    #     main_info_5=causal_model,
    #     main_info_6=noise_type
    # )

    # Display the performance of skeleton learning under different settings.
    print("* Setup:")
    if (causal_model == 'hybrid_nonlinear') and (ind_test_type == 'non_linear'):
        print("* Sample: {}, Graph Nodes: {}".
              format(int(sample / 10), graph_node_num))
    print("* Sample: {}, Graph Nodes: {}".
          format(sample, graph_node_num))
    print("* Causal Model: {}, Noise Type: {}, Independence Test: {}".
          format(causal_model, noise_type, ind_test_type))
    print("* F1-score of PC Skeleton Learning: ", f1_score_skeleton)
    print("* Running Time of PC: ", running_time)

    # Display an empty line for each parameter-setting test.
    print("\n")


# ##########################################################################
# ### TESTING SECTION FOR REGRESSION AND INDEPENDENCE TESTS ################
# ##########################################################################


ACTIVATION_0x2 = {
    'switch_1':   True,
    'testing_part_two':   False,
    'testing_part_three': False,
}

REPETITIONS_0x2 = 1


@pytest.fixture(params=[
    # Write down some descriptions here.
    # causal_model,      sample, regressor,          ind_test_method
    ('lingam',           1000,   LinearRegression(), 'HSIC-Fisher'),
    ('hybrid_nonlinear', 500,    MLPRegressor(),     'KCI'),
    ('hybrid_nonlinear', 1000,   LinearGAM(),        'HSIC-GAM'),
])
def procedure_param(request):
    return request.param


# ### CORRESPONDING TEST ##################################################
# Loc:  causality_instruments >> get_residuals_scm
# Loc:  causality_instruments >> conduct_ind_test

# ### CODING DATE ##########################################################
# Testing Init  : 2023-12-17
# Testing Update: 2024-03-06

def test_0x2_pairwise_causal_procedure(procedure_param):
    """
    Testing for exposing details of regression and independence tests
    ahead of encapsulation.

    * Switch-1: Local test for relative (data) format when executing Regressor.fit();
    * Switch-2: General comparison tests.
    """

    for i in range(REPETITIONS_0x2):

        random_seed = copy_and_rename(i)

        # === DATA GENERATION AND GROUND-TRUTH PREPARATION =================

        graph_node_num = 2
        causal_model, sample, regressor, ind_test_method = procedure_param

        generator = Generator(
            graph_node_num=graph_node_num,
            sample=sample,
            causal_model=causal_model,
            noise_type='non-Gaussian',
            sparsity=1.0
        )

        ground_truth, data = generator.run_generation_procedure().unpack()
        explanatory_data, explained_data = get_pair_cause_effect(ground_truth, data)

        explanatory_data = check_1dim_array(explanatory_data)
        explained_data = check_1dim_array(explained_data)

        explanatory_data_examined = copy_and_rename(explanatory_data)
        explained_data_examined = copy_and_rename(explained_data)

        # === LOCAL TEST FOR REGRESSOR FITTING =============================

        if ACTIVATION_0x2['switch_1']:
            display_test_section_symbols(
                testing_mark='local_test_for_regressor_fitting'
            )

            print("*  Shape of Explanatory Data: ", explanatory_data.shape)
            print("*  Shape of Explained Data:   ", explained_data.shape)
            print("*  Type of Explanatory Data:  ", type(explanatory_data))
            print("*  Type of Explained Data:    ", type(explained_data))
            print("*  Type of Regressor:         ", type(regressor))

            try:
                # explanatory_data.flatten()
                # explained_data.flatten()

                regressor.fit(explanatory_data, explained_data)
                est_explained_data = regressor.predict(explanatory_data)
                est_explained_data = check_1dim_array(est_explained_data)

                get_residuals_scm(
                    explanatory_data=explanatory_data,
                    explained_data=explained_data,
                    regressor=regressor
                )

            except Exception as err_msg:
                print(
                    ">> An error occurred in get_residuals_scm() "
                    ">> regressor.fit / regressor.predict",
                    err_msg
                )

            print('>> Pass\n')

        # === PART TWO =====================================================

        if ACTIVATION_0x2['testing_part_two']:

            np.random.seed(random_seed)

            regressor.fit(explanatory_data, explained_data)
            est_explained_data = regressor.predict(explanatory_data)
            est_explained_data = check_1dim_array(est_explained_data)

            residuals = explained_data - est_explained_data

            # --- Separation -----------------------------------------------

            display_test_section_symbols(testing_mark='testing_part_two')

            residuals_expected = copy_and_rename(residuals)

            residuals_examined = check_get_residuals_scm(
                explanatory_data=explanatory_data_examined,
                explained_data=explained_data_examined,
                regressor=regressor,
                random_seed=random_seed
            )

            np.testing.assert_allclose(
                actual=residuals_expected,
                desired=residuals_examined
            )

        # === PART FOUR ====================================================

        if ACTIVATION_0x2['testing_part_three']:
            if ind_test_method == "KCI":
                kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
                p_value, _ = kci.compute_pvalue(
                    data_x=check_1dim_array(explanatory_data),
                    data_y=check_1dim_array(residuals_expected)
                )
            elif ind_test_method == "HSIC-Fisher":
                p_value = hsic_gam(
                    X=check_1dim_array(explanatory_data),
                    Y=check_1dim_array(residuals_expected),
                    mode="pvalue"
                )
            elif ind_test_method == "HSIC-GAM":
                p_value = hsic_gam(
                    X=check_1dim_array(explanatory_data),
                    Y=check_1dim_array(residuals_expected),
                    mode="pvalue"
                )
            else:
                raise ValueError("Error")

            # --- Separation -----------------------------------------------

            display_test_section_symbols(testing_mark='testing_part_two')

            p_value_expected = copy_and_rename(p_value)

            p_value_examined = check_conduct_ind_test(
                explanatory_data=explanatory_data_examined,
                residuals=residuals_examined,
                ind_test_method=ind_test_method,
                random_seed=random_seed
            )

            np.testing.assert_allclose(
                actual=p_value_expected,
                desired=p_value_examined
            )





