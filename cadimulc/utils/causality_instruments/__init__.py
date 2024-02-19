"""Write down some descriptions here."""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * Related files: test_causality_instruments.py and causality_instruments


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Program the framework of get_skeleton_from_pc().          27th.Dec, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: Program and test get_skeleton_from_pc()
#
# Done:
#

from causallearn.search.FCMBased.lingam.hsic2 import hsic_gam
from causallearn.utils.KCI.KCI import KCI_UInd
from causallearn.utils.cit import kci, fisherz
from causallearn.search.ConstraintBased.PC import pc

from sklearn.utils import check_array

from cadimulc.utils.extensive_modules import check_1dim_array

import networkx as nx
import copy as cp


# ### CORRESPONDING TEST ##################################################
# test_hybrid_algorithms.py > test_pair_causal_procedure()
def get_residuals_scm(explanatory_data, explained_data, regressor):
    """
    - pairwise or multiple
    - linear or non-linear (GAM or MLP)

    Parameters
    ----------
    explanatory_data : ndarray (shape[n, d])
    explained_data : ndarray (shape[n, 1])
    regressor : object

    Return
    ------
    residuals : ndarray (shape[n, 1])
    """

    explanatory_data = check_1dim_array(explanatory_data)
    explained_data = check_1dim_array(explained_data)

    err_msg = 'reg could not fit / current passed regressor module involves...'
    try:
        regressor.fit(explanatory_data, explained_data)
        est_explained_data = regressor.predict(explanatory_data)
        est_explained_data = check_1dim_array(est_explained_data)

    except Exception as err_msg:
        print("An error occurred:", err_msg)

    residuals = explained_data - est_explained_data

    return residuals


# ### CORRESPONDING TEST ##################################################
# test_hybrid_algorithms.py > test_pair_causal_procedure()
def conduct_ind_test(explanatory_data, residuals, ind_test_method):
    """
    - functional-based context (conditional ind_test)
    - explanatory variable(s) and explained variable's residuals
    - pairwise variables

    Parameters
    ----------
    explanatory_data : ndarray (shape[n, d])
    residuals : ndarray (shape[n, 1])
    ind_test_method : string

    Return
    ------
    p_value : float
    """

    if ind_test_method == "kernel_hsic":
        kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
        p_value, _ = kci.compute_pvalue(
            data_x=check_1dim_array(explanatory_data),
            data_y=check_1dim_array(residuals)
        )
    elif ind_test_method == "HSIC-Fisher":
        p_value = hsic_gam(
            X=check_1dim_array(explanatory_data),
            Y=check_1dim_array(residuals),
            mode="pvalue"
        )
    elif ind_test_method == "HSIC-GAM":
        p_value = hsic_gam(
            X=check_1dim_array(explanatory_data),
            Y=check_1dim_array(residuals),
            mode="pvalue"
        )
    else:
        raise ValueError("Error")

    return p_value


# ### CORRESPONDING TEST ###################################################
# Loc: test_causality_instruments.py >> test_0x1_get_skeleton_from_pc

def get_skeleton_from_pc(
    data,
    alpha=0.3,
    ind_test_type='linear',
):
    """
    Obtain the causal skeleton learned by the stable PC-algorithm.

    Parameters
    ----------
    data : ndarray or dataframe
    alpha : float (default: 0.3)
    ind_test_type : string (default: linear)

    Returns
    -------
    skeleton, running_time
        Numpy array of estimated ``skeleton`` along with ``running_time``.
    """

    data = check_array(data)

    if ind_test_type == 'linear':
        # Fisher-Z conditional independence test as defaulting for linearity
        indep_test = fisherz
    else:
        # kernel conditional independence test as defaulting for non-linearity
        # massive computational cost
        indep_test = kci

    # Conduct the stable PC-algorithm implemented by causal-learn.
    causal_graph = pc(
        data=data,
        alpha=alpha,
        indep_test=indep_test,
        show_progress=False
    )

    causal_graph.to_nx_skeleton()

    skeleton = nx.to_numpy_array(causal_graph.nx_skel)
    running_time = round(causal_graph.PC_elapsed, 3)

    return skeleton, running_time
