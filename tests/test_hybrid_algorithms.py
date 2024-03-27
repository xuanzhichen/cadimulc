"""Test for the framework and auxiliary modules of Nonlinear-MLC algorithm."""

# On Maintenance: Xuanzhi CHEN <xuanzhichen.42@gmail.com>

# Source: ../cadimulc/hybrid_algorithms/hybrid_algorithms.py


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * Test-0x5:
#   * There might be a bug: Compelling reinitialization of pairwise regressor.
#   * The odds are very likely that searching round times are less than 2.
#
# * Test-0x3:
#   * MLC-LiNGAM has been strengthened by additionally combining a loop for
#     searching the most exogenous and the most leaf variables.
#   * Testing shows outcomes of up-down search mirror bottom-up search.
#
# * Test-0x4:
#   * Simulating LiNGAM in cadimulc maybe subjective, thus a baseline is needed.
#   * Average performance on the general setting: 0.3 (baseline: 0.1)
#   * Average performance with a priori of causal skeleton: [0.50, 0.55].
#
# * Some ideas that might be able to add for the test:
#   * the ADJ method
#   * p-value variations with hidden confounders (latent Generator)
#   * Nonlinear-MLC:
#       * framework of stage-2, orientation of stage-2
#       * clique search of stage-3,
#       * ground-truth of a given skeleton,
#       * the whole framework (given the instructions)
#
# * Expect to reuse the current building logic for subsequent refactorings:
#   * nonlinearMLC -> generator -> visualization -> regression + ind-test
#   * developing phrases: stage-1 > 2 > 3 > skeleton > (evaluation)


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Build testing flow relative to MLC-LiNGAM.                30th.Jan, 2024
#
# * Finish test 'test_pair_causal_procedure()'.               19th.Dec, 2023
#
# * Get more familiar with testing logic and ready to test.   17th.Dec, 2023
#
# * The two file 'nonlinearmlc.py' and 'test_hybrid_algorithms.py' should be
#   top-level during the refactorings of the project          13th.Dec, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: Add comments for associating files and prepare for the next phrase
#       until stage-1 of Nonlinear-MLC have been built and tested.
# TODO: Add readability flag for iterative tests. (Optional)
#
# Done:
# _TODO: Initialize reconstruction of Nonlinear-MLC stage-1 and stage-2.
# _TODO: Finish test 'test_pair_causal_procedure()'.
# _TODO: Add function type "LiNGAM" into Generator (Hybrid-Nonlinear).
# _TODO: Programming for functions: get_pair_cause_effect() and check_*(),
#       and check the final procedure.


# testing modules
from cadimulc.hybrid_algorithms import HybridFrameworkBase
from cadimulc.hybrid_algorithms import NonlinearMLC, MLCLiNGAM
from cadimulc.hybrid_algorithms.hybrid_algorithms import GraphPatternManager

# basic modules
from causallearn.search.FCMBased.lingam import BottomUpParceLiNGAM
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM

from cadimulc.utils.causality_instruments import (
    get_residuals_scm,
    conduct_ind_test,
)
from cadimulc.utils.extensive_modules import (
    copy_and_rename,
    display_test_section_symbols,
    cancel_test_duplicates,
    get_skeleton_from_adjmat,
    get_medium_num,
    check_1dim_array
)
from cadimulc.utils.generation import Generator
from cadimulc.utils.evaluation import Evaluator
from cadimulc.utils.visualization import draw_graph_from_ndarray

import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import random
import pytest


# ##########################################################################
# ### CONVENIENCE FUNCTION(S) ##############################################
# ##########################################################################


def simulate_single_case_causal_model(
    linear_setting=True,
    graph_node_num=10,
    sample=1000,
    random_seed=42
):
    """ Simulate fixed-parameter causal models for single-case testing.

    Parameters
    ----------
    linear_setting : bool
        Default as True and False for 'hybrid_nonlinear' setting.
    graph_node_num : int
        Write down some descriptions here.
    sample : int
        Write down some descriptions here.
    random_seed : int
        Default as 42.

    Returns
    -------
    ground_truth, data : ndarray
        Numpy array of ``ground truth`` of the simulated causal graph,
        and numpy array of `data` of the simulated dataset.
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    causal_model = 'lingam' if linear_setting is True else 'hybrid_nonlinear'
    noise_type = 'non-Gaussian' if linear_setting is True else 'Gaussian'

    generator = Generator(
        graph_node_num=graph_node_num,
        sample=sample,
        causal_model=causal_model,
        noise_type=noise_type
    )
    ground_truth, data = generator.run_generation_procedure().unpack()

    return ground_truth, data


def adjust_list_counting_from_one(_list) -> list:
    """
    Adjust each of the int-type variables
    (e.g. variable := variable + 1) for printing.
    """

    _list_temp = []
    for element in _list:
        _list_temp.append(element + 1)

    return _list_temp


def adjust_nested_list_counting_from_one(_lists) -> list:
    """
    The nexted-list version of 'adjust_list_counting_from_one'.

    This is common when printing the elements of maximal cliques which
    are typically in form of nested lists.
    """

    # Specify _lists is an entity of nested list.
    _lists_temp = []
    for _list in _lists:
        _list_temp = []
        for element in _list:
            _list_temp.append(element + 1)
        _lists_temp.append(_list_temp)

    return _lists_temp

# ##########################################################################
# ### AUXILIARY TESTING FUNCTION(S) ########################################
# ##########################################################################


# # ### SUBORDINATE COMPONENT(S) ###########################################
# # test_get_skeleton_from_pc()
# def check_get_skeleton_from_pc():
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return


# ### SUBORDINATE COMPONENT(S) #############################################
# test_stage_one_learning()
# def check_stage_one_learning():
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return


# ##########################################################################
# ### TEST SECTION FOR HYBRID FRAMEWORK BASE ###############################
# ##########################################################################


# ### CORRESPONDING TEST ###################################################
# Loc: hybrid_framework.py >> _causal_skeleton_learning

# ### AUXILIARY COMPONENT(S) ###############################################
# Testing Date: 2024-01-28 | 16:16 (Pass)
# --------------------------------------------------------------------------
# Function:     None

def test_0x1_causal_skeleton_learning():
    """
    Testing for the correct initialization of HybridFrameworkBase and
    the foremost encapsulation of get_skeleton_from_pc.
    """

    random_seed = 42

    # Randomly generate simulated causal model in a general setting.
    ground_truth, data = simulate_single_case_causal_model(
        linear_setting=True,
        random_seed=random_seed
    )

    display_test_section_symbols()

    print("* Ground Truth Adjacency Matrix: \n", ground_truth)

    display_test_section_symbols()

    # Test the correct initialization of HybridFrameworkBase.
    mlc_lingam = MLCLiNGAM()
    mlc_lingam._dataset = data
    model = mlc_lingam

    # nonlinear_mlc = NonlinearMLC()
    # nonlinear_mlc._dataset = data
    # model = nonlinear_mlc

    # Test the foremost encapsulation of get_skeleton_from_pc.
    model._causal_skeleton_learning(data)

    draw_graph_from_ndarray(
        array=ground_truth,
        testing_text='ground_truth'
    )
    draw_graph_from_ndarray(
        array=model.skeleton_,
        testing_text='estimation_skeleton'
    )
    draw_graph_from_ndarray(
        array=model.skeleton_,
        graph_type=True,
        testing_text='estimation_adjacency_matrix'
    )

    plt.show()

    print("* Stage-1 Running Time: ", model.stage1_time_)


# ##########################################################################
# ### TESTING SECTION FOR GRAPH PATTERN MANAGER ############################
# ##########################################################################


SEED_0x2 = 42
# SEED_0x2 = 42 + 50
# SEED_0x2 = 42 + 100


# ### CORRESPONDING TEST ###################################################
# Loc:  hybrid_algorithms.py
#       >> GraphPatternManager >> recognize_maximal_cliques_pattern
# Loc:  hybrid_algorithms.py
#       >> GraphPatternManager >> get_undetermined_cliques
# Loc:  hybrid_algorithms.py
#       >> GraphPatternManager >> check_newly_determined

# ### CODING DATE ##########################################################
# Testing Init  : 2024-02-05
# Testing Update: 2024-03-04

def test_0x2_graph_pattern_manager():
    """
    Testing for the three of Class-methods as to the cliques pattern management.

    * default: recognize the maximal-cliques pattern over a causal skeleton;
    * get undetermined cliques over a partial causal skeleton;
    * check newly determined cliques over a partial causal skeleton.
    """

    # ================== INITIALIZE THE CAUSAL GRAPH =======================

    # Randomly generate simulated causal model in a general setting.
    # Notes: the subsequent graph modifications are based on SEED_0x2
    ground_truth, data = simulate_single_case_causal_model(
        graph_node_num=5,
        random_seed=SEED_0x2
    )

    # Obtain the causal skeleton from ground truth.
    causal_skeleton = get_skeleton_from_adjmat(adjacency_matrix=ground_truth)

    # ==================== RECOGNIZE MAXIMAL CLIQUES ========================

    display_test_section_symbols(testing_mark='recognize_maximal_cliques')

    maximal_cliques = GraphPatternManager.recognize_maximal_cliques_pattern(
        causal_skeleton=causal_skeleton
    )

    # Display the maximal cliques searching result.

    # maximal_cliques_temp = []
    # for maximal_clique in maximal_cliques:
    #     maximal_clique_temp = []
    #     for node in maximal_clique:
    #         maximal_clique_temp.append(node + 1)
    #     maximal_cliques_temp.append(maximal_clique_temp)
    # print("* Maximal Cliques Pattern: ", maximal_cliques_temp)

    print(
        "* Maximal Cliques Pattern: ",
        adjust_nested_list_counting_from_one(maximal_cliques)
    )

    # Display the associating causal skeleton.
    draw_graph_from_ndarray(
        array=causal_skeleton,
        testing_text="skeleton"
    )
    plt.show()

    # # additional testing code fragment
    # # Remove the trivial graph from maximal_cliques.
    # for clique in maximal_cliques_temp:
    #     if len(clique) == 1:
    #         print("Trivial Graph (Node): ", clique)
    #         maximal_cliques_temp.remove(clique)
    #         print("* Maximal Cliques Removing Trivial Graphs: ",
    #               maximal_cliques_temp)
    #
    # print("* Maximal Cliques Removing Trivial Graphs (Repeat): ",
    #       maximal_cliques_temp)

    # ===================== GET UNDETERMINED CLIQUES =======================

    display_test_section_symbols(testing_mark='get_undetermined_cliques')

    # Initialize a graph pattern manager for subsequent tests.
    graph_pattern_manager = GraphPatternManager(init_graph=causal_skeleton)

    # # Orient the causal skeleton randomly.
    # for i in range(graph_pattern_manager.managing_adjacency_matrix.shape[0]):
    #     for j in range(graph_pattern_manager.managing_adjacency_matrix.shape[1]):
    #         if i <= j:
    #             if (graph_pattern_manager.managing_adjacency_matrix[i][j] == 1) and (
    #                     1 == graph_pattern_manager.managing_adjacency_matrix[j][i]):
    #                 orienting = np.random.randint(low=0, high=1)
    #                 if orienting == 1:
    #                     # Partially orient the causal skeleton
    #                     # (notice: avoid acyclic graph).
    #                     graph_pattern_manager.managing_adjacency_matrix[i][j] = 1
    #                     graph_pattern_manager.managing_adjacency_matrix[j][i] = 0

    # Orient the causal skeleton based on existing maximal cliques (SEED_0x2).
    determined_clique = maximal_cliques[0] if len(maximal_cliques) > 0 else []
    for i in determined_clique:
        for j in determined_clique[i:]:
            # Partially orient the causal skeleton
            # (notice: avoid acyclic graph).
            graph_pattern_manager.managing_adjacency_matrix[i][j] = 1
            graph_pattern_manager.managing_adjacency_matrix[j][i] = 0

    # Display the undetermined cliques searching result.
    undetermined_cliques = graph_pattern_manager.get_undetermined_cliques(
        maximal_cliques=maximal_cliques
    )

    # undetermined_cliques_temp = []
    # for undetermined_clique in undetermined_cliques:
    #     undetermined_clique_temp = []
    #     for node in undetermined_clique:
    #         undetermined_clique_temp.append(node + 1)
    #     undetermined_cliques_temp.append(undetermined_clique_temp)
    # print("* Undetermined Cliques: ", undetermined_cliques_temp)

    print(
        "* Undetermined Cliques: ",
        adjust_nested_list_counting_from_one(undetermined_cliques)
    )

    # Display the associating partial skeleton (adjacency matrix).
    draw_graph_from_ndarray(
        array=graph_pattern_manager.managing_adjacency_matrix,
        testing_text="partial_skeleton"
    )
    plt.show()

    # ===================== CHECK NEWLY DETERMINED =========================

    display_test_section_symbols(testing_mark='check_newly_determined')

    # Notes: Not necessary to rewrite an instance
    # if the following tests are dependent.

    # # Rewrite the graph pattern manager for another test.
    # graph_pattern_manager = GraphPatternManager(init_graph=causal_skeleton)

    graph_pattern_manager.store_last_managing_adjacency_matrix()

    # Display whether there exists newly determined cliques (before modification)
    newly_determined = graph_pattern_manager.check_newly_determined(
        last_undetermined_cliques=undetermined_cliques
    )
    print("* Newly Determined (Before Modification): ", newly_determined)

    # Test checking newly determined cliques.
    # Randomly orient the causal skeleton.
    # for i in range(graph_pattern_manager.managing_adjacency_matrix.shape[0]):
    #     for j in range(graph_pattern_manager.managing_adjacency_matrix.shape[1]):
    #         if i <= j:
    #             if (graph_pattern_manager.managing_adjacency_matrix[i][j] == 1) and (
    #                     1 == graph_pattern_manager.managing_adjacency_matrix[j][i]):
    #                 graph_pattern_manager.managing_adjacency_matrix[i][j] = 1
    #                 graph_pattern_manager.managing_adjacency_matrix[j][i] = 0
    #
    #                 # Mildly modify at least one edge for testing.
    #                 break

    # Orient the causal skeleton based on existing maximal cliques (SEED_0x2).
    determined_clique = maximal_cliques[1] if len(maximal_cliques) > 0 else []
    for i in determined_clique:
        for j in determined_clique[i:]:
            if (graph_pattern_manager.managing_adjacency_matrix[i][j] == 1) and (
                   1 == graph_pattern_manager.managing_adjacency_matrix[j][i]
            ):
                graph_pattern_manager.managing_adjacency_matrix[i][j] = 1
                graph_pattern_manager.managing_adjacency_matrix[j][i] = 0

                # Mildly modify at least one edge for testing.
                break

    # Display whether there exists newly determined cliques (after modification)
    newly_determined = graph_pattern_manager.check_newly_determined(
        last_undetermined_cliques=undetermined_cliques
    )
    print("* Newly Determined (After Modification): ", newly_determined)

    # Display the undetermined cliques searching result again.
    undetermined_cliques = graph_pattern_manager.get_undetermined_cliques(
        maximal_cliques=maximal_cliques
    )

    # undetermined_cliques_temp = []
    # for undetermined_clique in undetermined_cliques:
    #     undetermined_clique_temp = []
    #     for node in undetermined_clique:
    #         undetermined_clique_temp.append(node + 1)
    #     undetermined_cliques_temp.append(undetermined_clique_temp)
    # print("* Undetermined Cliques: ", undetermined_cliques_temp)

    print(
        "* Undetermined Cliques: ",
        adjust_nested_list_counting_from_one(undetermined_cliques)
    )

    # Display the associating one-edge-modifying causal skeleton.
    draw_graph_from_ndarray(
        array=graph_pattern_manager.managing_adjacency_matrix,
        testing_text="one_edge_modified"
    )
    plt.show()


# ##########################################################################
# ### TEST SECTION FOR MLC-LINGAM ##########################################
# ##########################################################################


ACTIVATION_0x3 = {
    'general_setting': True,
    'skeleton_priori': False,
    'up_down_search': False,
    'bottom_up_search': True
}

SEED_0x3 = 42
# SEED_0x3 = 42 + 50
# SEED_0x3 = 42 + 100


# ### CORRESPONDING TEST ###################################################
# Loc: hybrid_algorithms.py >> MLC-LiNGAM >> stage_two_learning

# ### AUXILIARY COMPONENT(S) ###############################################
# Testing Date: 2024-01-29 | 16:33 (Pass)
# --------------------------------------------------------------------------
# Function: None

def test_0x3_procedure_stage_two_learning():
    """
    Testing for exposing details amidst stage-II learning, and encapsulation:

    * data flow in stage-II learning;
    * behaviors of identifying partial causal order...
    * particularly the stability of bottom-up search;
    """

    # Randomly generate simulated causal model in a general setting.
    # linear and non-Gaussian setting
    ground_truth, data = simulate_single_case_causal_model(
        linear_setting=True,
        random_seed=SEED_0x3
    )

    # Initialize mlc-lingam ahead of stage-ii learning.
    mlc_lingam = MLCLiNGAM()
    mlc_lingam._dataset = data
    mlc_lingam._dim = data.shape[1]

    if ACTIVATION_0x3['general_setting']:
        mlc_lingam._stage_1_learning(data)

    if ACTIVATION_0x3['skeleton_priori']:
        mlc_lingam._skeleton = get_skeleton_from_adjmat(ground_truth)
        mlc_lingam._adjacency_matrix = get_skeleton_from_adjmat(ground_truth)

    display_test_section_symbols(testing_mark='relative_causal_graph')

    draw_graph_from_ndarray(
        array=ground_truth,
        testing_text='ground_truth'
    )
    draw_graph_from_ndarray(
        array=mlc_lingam._skeleton,
        testing_text='causal_skeleton'
    )
    plt.show()

    display_test_section_symbols(testing_mark='relative_causal_discovery')

    # Listing structural procedure clips of Stage-II learning in MLC-LiNGAM
    # initialization
    adjacent_set = GraphPatternManager.find_adjacent_set(
        causal_skeleton=mlc_lingam._skeleton
    )
    graph_pattern_manager = GraphPatternManager(
        init_graph=mlc_lingam._skeleton
    )

    _X = cp.copy(mlc_lingam._dataset)
    _x = np.arange(mlc_lingam._dim)
    k_head = []
    k_tail = []
    regressor = LinearRegression()
    ind_test_method = 'kernel_hsic'

    search_round = 0

    if ACTIVATION_0x3['up_down_search']:

        print("* Structural Procedure of Exogenous Variable Searching Starts ...")

        # identify exogenous variables
        repeat = True

        while repeat:

            # additional code fragments for testing
            search_round += 1
            print("* Search Round: ", search_round)
            k_head_temp = [(item + 1) for item in k_head]
            print("    * K-head List: ", k_head_temp)

            if len(k_head) == (len(_x) - 1):
                # additional code fragments for testing
                print("* End Searching Round")
                print("\n")
                k_head_temp = [(item + 1) for item in k_head]
                print("* K-head List: ", k_head_temp)

                break

            p_values_x_all = {}
            for x_i in (set(_x) - set(k_head)):
                print("    * Candidate Exogenous Var: ", x_i + 1)

                adjacent_set_i = adjacent_set[x_i]

                # additional code fragments for testing
                adjacent_set_i_temp = []
                for adjacent_variable in (adjacent_set[x_i] - set(k_head)):
                    adjacent_set_i_temp.append(adjacent_variable + 1)
                adjacent_set_i_temp = set(adjacent_set_i_temp)
                print("    * Selective Adjacent Set: ", adjacent_set_i_temp)

                if len(adjacent_set_i) == 0:
                    print("    * Continue Next Searching Round")

                    k_head.append(x_i)
                    continue

                adjacent_set_i = adjacent_set_i - set(k_head)

                if len(adjacent_set_i) == 0:
                    print("    * Continue Next Searching Round")

                    k_head.append(x_i)
                    continue

                p_values_x_i = []
                print("        * Perform Regression and Independence Test")

                for x_j in adjacent_set_i:
                    residuals = get_residuals_scm(
                        explanatory_data=_X[:, x_i],
                        explained_data=_X[:, x_j],
                        regressor=regressor
                    )

                    # Cancel duplicates raised by controlled testing.
                    cancel_test_duplicates()

                    p_value = conduct_ind_test(
                        explanatory_data=_X[:, x_i],
                        residuals=residuals,
                        ind_test_method=ind_test_method
                    )
                    p_values_x_i.append(p_value)

                if np.min(p_values_x_i) >= mlc_lingam.pc_alpha:
                    p_values_x_all[x_i] = np.min(p_values_x_i)

            if len(p_values_x_all.values()) == 0:
                print("        * Not Exogenous Variable Anymore")
                print("        * End Searching Round")
                print("\n")

                repeat = False

            else:
                p_value_max = cp.copy(mlc_lingam.pc_alpha)
                x_exogenous = None

                for x_i, p_value in p_values_x_all.items():
                    if p_value > p_value_max:
                        p_value_max = p_value
                        x_exogenous = x_i

                # Cancel duplicates raised by controlled testing.
                cancel_test_duplicates()

                print("\n")
                print("    * Suggest {} Is the Most Exogenous".format(x_exogenous + 1))
                print("    * Update K-Head List and Replace the Remaining with Residuals")
                print("    * Continue Next Searching Round")
                print("\n")

                repeat = True
                k_head.append(x_exogenous)

                adjacent_set_exo = adjacent_set[x_exogenous] - set(k_head)

                for x_j in adjacent_set_exo:
                    supplanting_residuals = get_residuals_scm(
                        explanatory_data=_X[:, x_exogenous],
                        explained_data=_X[:, x_j],
                        regressor=regressor
                    )

                    _X[:, x_j] = supplanting_residuals.ravel()

        # additional code fragments for testing
        k_head_temp = [(item + 1) for item in k_head]
        print("* K-head List: ", k_head_temp)

    if ACTIVATION_0x3['bottom_up_search']:
        if len(k_head) < (mlc_lingam._dim - 2):

            print("* Structural Procedure of Leaf Variable Searching Starts ...")

            # identify leaf variables
            repeat = True
            while repeat:

                # additional code fragments for testing
                search_round += 1
                print("* Search Round: ", search_round)
                k_head_temp = [(item + 1) for item in k_head]
                print("    * K-head List: ", k_head_temp)
                k_tail_temp = [(item + 1) for item in k_tail]
                print("    * K-tail List: ", k_tail_temp)

                if len(k_head) + len(k_tail) == (len(_x) - 1):
                    # additional code fragments for testing
                    print("* End Searching Round")
                    print("\n")
                    k_head_temp = [(item + 1) for item in k_head]
                    print("    * K-head List: ", k_head_temp)
                    k_tail_temp = [(item + 1) for item in k_tail]
                    print("    * K-tail List: ", k_tail_temp)

                    break

                # repeat = False

                p_values_x_all = {}
                for x_j in (set(_x) - (set(k_head) | set(k_tail))):

                    print("    * Candidate Leaf Var: ", x_j + 1)

                    adjacent_set_j = adjacent_set[x_j] - (set(k_head) | set(k_tail))

                    # additional code fragments for testing
                    adjacent_set_j_temp = []
                    for adjacent_variable in adjacent_set_j:
                        adjacent_set_j_temp.append(adjacent_variable + 1)
                    adjacent_set_j_temp = set(adjacent_set_j_temp)
                    print("    * Selective Adjacent Set: ", adjacent_set_j_temp)

                    if len(adjacent_set_j) == 0:

                        print("    * Continue Next Searching Round")

                        # k_tail.insert(0, x_j)
                        k_head.append(x_j)
                        continue

                    print("    * Perform Regression and Independence Test")

                    residuals = get_residuals_scm(
                        explanatory_data=_X[:, list(adjacent_set_j)],
                        explained_data=_X[:, x_j],
                        regressor=regressor
                    )

                    # Cancel duplicates raised by controlled testing.
                    cancel_test_duplicates()

                    p_value = conduct_ind_test(
                        explanatory_data=_X[:, list(adjacent_set_j)],
                        residuals=residuals,
                        ind_test_method=ind_test_method
                    )
                    p_values_x_j = copy_and_rename(p_value)

                    # print("        * P-Value: ", np.min(p_values_x_j))

                    if p_values_x_j >= mlc_lingam.pc_alpha:
                        p_values_x_all[x_j] = p_values_x_j

                if len(p_values_x_all.values()) == 0:
                    print("        * Not Leaf Variable Anymore")
                    print("        * End Searching Round")
                    print("\n")

                    repeat = False

                else:
                    p_value_max = cp.copy(mlc_lingam.pc_alpha)
                    x_leaf = None

                    for x_j, p_value in p_values_x_all.items():
                        if p_value > p_value_max:
                            p_value_max = p_value
                            x_leaf = x_j

                    print("\n")
                    print("    * Suggest {} Is the Most Leaf".format(x_leaf + 1))
                    print("    * Update K-Tail List and Ignore the Leaf Variable In the Following")
                    print("    * Continue Next Searching Round")
                    print("\n")

                    repeat = True
                    k_tail.insert(0, x_leaf)

    # Construct partial causal order.
    causal_order = []
    _x_temp = cp.copy(_x)

    for x_i in k_head:
        causal_order.append(x_i + 1)
        _x_temp = _x_temp[_x_temp != x_i]

    for x_j in k_tail:
        _x_temp = _x_temp[_x_temp != x_j]

    _x_temp2 = []
    for temp in _x_temp:
        _x_temp2.append(temp + 1)

    causal_order.append(_x_temp2)

    for x_j in k_tail:
        causal_order.append(x_j + 1)

    display_test_section_symbols(testing_mark='result')

    print("* Partial Causal Order: ", causal_order)

    # Construct partial causal graph.
    graph_pattern_manager.identify_partial_causal_order(
        k_head=k_head,
        k_tail=k_tail
    )

    mlc_lingam._adjacency_matrix = graph_pattern_manager.managing_adjacency_matrix
    mlc_lingam._parents_set = graph_pattern_manager.managing_parents_set

    draw_graph_from_ndarray(
        array=mlc_lingam._adjacency_matrix,
        testing_text='estimated_graph'
    )
    plt.show()


ACTIVATION_0x4 = {
    'single_case': True,
    'repetitive_cases': False,
    'repetitive_setting': {
        'general_setting': False,
        'skeleton_priori': True,
    },
    'baseline_comparison': True
}

REPETITIONS_0x4 = 10


# ### CORRESPONDING TEST ###################################################
# Loc: hybrid_algorithms.py >> MLC-LiNGAM >> stage_two_learning

# ### AUXILIARY COMPONENT(S) ###############################################
# Testing Date: 2024-01-30 | 16:14 (Pass)
# --------------------------------------------------------------------------
# Function: None

def test_0x4_performance_stage_two_learning():
    """
    Testing for numerical and empirical results limited on stage-II learning:

    * single complete performance of calling to MLC-LiNGAM;
    * average performance on the general setting;
    * average performance on the priori of causal skeleton;
    * comparison test against the baseline method ParceLiNGAM.
    """

    # Test a single case given the same input in test 0x3 (general setting).
    if ACTIVATION_0x4['single_case']:

        display_test_section_symbols(testing_mark='single_case')

        # Randomly generate simulated causal model in a general setting.
        # linear and non-Gaussian setting
        ground_truth, data = simulate_single_case_causal_model(
            linear_setting=True,
            random_seed=SEED_0x3
        )

        if ACTIVATION_0x4['baseline_comparison']:
            # Perform parce-lingam causal discovery.
            parce_lingam = BottomUpParceLiNGAM()
            parce_lingam.fit(data)

            adjacency_matrix_baseline = (
                (parce_lingam.adjacency_matrix_ > 0.01).astype(int)
            )

            draw_graph_from_ndarray(
                array=adjacency_matrix_baseline,
                testing_text='estimated_graph_baseline'
            )
            plt.show()

        # Initialize mlc-lingam ahead of stage-ii learning.
        mlc_lingam = MLCLiNGAM()
        mlc_lingam._dataset = data
        mlc_lingam._dim = data.shape[1]

        # Conduct the stage-i learning.
        mlc_lingam._stage_1_learning(data)

        # Perform the stage-ii learning.
        mlc_lingam._stage_2_learning()

        draw_graph_from_ndarray(
            array=mlc_lingam._adjacency_matrix,
            testing_text='estimated_graph'
        )
        plt.show()

    # Test average performance on repetitive cases.
    if ACTIVATION_0x4['repetitive_cases']:

        display_test_section_symbols(testing_mark='repetitive_cases')

        i = 0
        step = 50
        f1_score_avg = 0
        f1_score_list = []

        if ACTIVATION_0x4['baseline_comparison']:
            f1_score_avg_baseline = 0

        while i < REPETITIONS_0x4:

            try:
                random_seed = copy_and_rename(i + step)
                np.random.seed(random_seed)
                random.seed(random_seed)

                i += 1

                # Randomly generate simulated causal model
                # in a general setting: linear and non-Gaussian setting
                generator = Generator(
                    graph_node_num=5,
                    sample=1000,
                    causal_model='lingam',
                    sparsity=0.7
                )
                ground_truth, data = (
                    generator.run_generation_procedure().unpack()
                )

                if ACTIVATION_0x4['baseline_comparison']:
                    # Perform parce-lingam causal discovery.
                    parce_lingam = BottomUpParceLiNGAM()
                    parce_lingam.fit(data)

                    adjacency_matrix_baseline = (
                        (parce_lingam.adjacency_matrix_ > 0.01).astype(int)
                    )

                    f1_score_avg_baseline = Evaluator.f1_score_pairwise(
                        true_graph=ground_truth,
                        est_graph=adjacency_matrix_baseline
                    )
                    f1_score_avg_baseline += f1_score_avg_baseline

                # Perform mlc-lingam causal discovery.
                mlc_lingam = MLCLiNGAM()
                mlc_lingam._dataset = data
                mlc_lingam._dim = data.shape[1]

                if ACTIVATION_0x4['repetitive_setting']['general_setting']:
                    mlc_lingam._stage_1_learning()

                if ACTIVATION_0x4['repetitive_setting']['skeleton_priori']:
                    mlc_lingam._skeleton = get_skeleton_from_adjmat(
                        adjacency_matrix=ground_truth
                    )
                    mlc_lingam._adjacency_matrix = (
                        cp.copy(mlc_lingam._skeleton)
                    )

                mlc_lingam._stage_2_learning()

                f1_score = Evaluator.f1_score_pairwise(
                    true_graph=ground_truth,
                    est_graph=mlc_lingam._adjacency_matrix
                )
                f1_score_avg += f1_score
                f1_score_list.append(f1_score)

                print("* Case-{}: Pass with F1-Score: {}".
                      format(i, f1_score))
                print("\n")

            except Exception as err_msg:
                print("* Case-{}: An Error Occurred: {}".
                      format(i, err_msg))
                print("\n")

        print("* Average F-1 Score: ", f1_score_avg / REPETITIONS_0x4)
        print("* Medium  F-1 Score: ", get_medium_num(f1_score_list))

        if ACTIVATION_0x4['baseline_comparison']:
            print("* Average F-1 Score of Baseline: ",
                  f1_score_avg_baseline / REPETITIONS_0x4)


# ##########################################################################
# ### TESTING SECTION FOR NONLINEAR-MLC ####################################
# ##########################################################################


ACTIVATION_0x5 = {
    'general_setting': True,
    'skeleton_priori': False,
}

# SEED_0x5 = 42
# SEED_0x5 = 42 + 50
SEED_0x5 = 42 + 100


# ### CORRESPONDING TEST ###################################################
# Loc: hybrid_algorithms.py >> NonlinearMLC >> fit
# Loc: hybrid_algorithms.py
#      >> GraphPatternManager >> identify_directed_causal_pair

# ### CODING DATE ##########################################################
# Testing Stabled: 2024-02-16
# Testing Updated: 2024-03-06

def test_0x5_procedure_fitting():
    """
    Testing for exposing details amidst the fitting procedure ahead of encapsulation:

    * test data flow in clique-based causal inference
      (_clique_based_causal_inference);
    * test identifying behaviors for directed causal pairs by the graph manager.
    """

    # =========== DATA GENERATION AND GROUND-TRUTH PREPARATION =============

    # Randomly generate simulated causal models in a general setting.
    # e.g. hybrid non-linear and Gaussian noise setting
    ground_truth, dataset = simulate_single_case_causal_model(
        linear_setting=False,
        random_seed=SEED_0x5
    )

    # Initialize nonlinear-mlc ahead of fitting procedure.
    nonlinear_mlc = NonlinearMLC()

    # _TODO: Specify the arguments that are necessary to be initialized.
    nonlinear_mlc._dataset = dataset
    nonlinear_mlc._dim = dataset.shape[1]
    if ACTIVATION_0x3['general_setting']:
        nonlinear_mlc._causal_skeleton_learning(dataset)
    if ACTIVATION_0x3['skeleton_priori']:
        nonlinear_mlc._skeleton = get_skeleton_from_adjmat(ground_truth)
        nonlinear_mlc._adjacency_matrix = get_skeleton_from_adjmat(ground_truth)

    display_test_section_symbols(testing_mark='corresponding_causal_skeleton')

    # Cancel duplicates raised by controlled testing.
    cancel_test_duplicates()

    draw_graph_from_ndarray(
        array=ground_truth,
        testing_text='ground_truth'
    )
    draw_graph_from_ndarray(
        array=nonlinear_mlc._skeleton,
        testing_text='causal_skeleton'
    )
    plt.show()

    display_test_section_symbols(testing_mark='corresponding_causal_discovery')

    # ================== LIST STRUCTURAL PROCEDURE CLIPS ===================

    # --------------- SETUP CLIQUE-BASED INFERENCE FRAMEWORK ---------------

    # Recognize the maximal-clique pattern based on the causal skeleton.
    maximal_cliques = GraphPatternManager.recognize_maximal_cliques_pattern(
        causal_skeleton=nonlinear_mlc._skeleton
    )

    # Initialize a graph pattern manager for subsequent learning.
    graph_pattern_manager = GraphPatternManager(
        init_graph=nonlinear_mlc._skeleton
    )

    print(
        "* Whole Patterns of Maximal Cliques: ",
        adjust_nested_list_counting_from_one(maximal_cliques)
    )

    print("* Structural Procedure of Clique-Based Inference Starts...")

    # Perform the nonlinear-mlc causal discovery.
    continue_search = True
    search_round = 0
    while continue_search:

        search_round += 1
        print("* Search Round: ", search_round)

        # Obtain cliques that remain at least one edge undetermined.
        undetermined_maximal_cliques = (
            graph_pattern_manager.get_undetermined_cliques(maximal_cliques)
        )

        print(
            "   * Undetermined Maximal Cliques: ",
            adjust_nested_list_counting_from_one(undetermined_maximal_cliques)
        )

        # End if all edges over the cliques have been determined.
        if len(undetermined_maximal_cliques) == 0:
            print("* End the Searching Round\n")
            break

        # --------------- DIVE INTO CLIQUE-BASED INFERENCE  ----------------

        determined_pairs = []

        # Temporally store the adjacency matrix ahead of starting a search round.
        graph_pattern_manager.store_last_managing_adjacency_matrix()

        # Conduct non-linear causal inference based on each maximal clique unit.
        for undetermined_maximal_clique in undetermined_maximal_cliques:

            print(
                "   * Undetermined Maximal Clique Unit: ",
                adjust_list_counting_from_one(undetermined_maximal_clique)
            )

            undetermined_pairs = []

            # Cancel duplicates raised by controlled testing.
            cancel_test_duplicates()

            # Get undetermined pairs within a clique.
            for i in undetermined_maximal_clique:
                for j in undetermined_maximal_clique[
                         undetermined_maximal_clique.index(i) + 1:
                         ]:
                    if (nonlinear_mlc._adjacency_matrix[i][j] == 1) and (
                            nonlinear_mlc._adjacency_matrix[j][i] == 1
                    ):
                        undetermined_pairs.append([i, j])

            print(
                "   * Undetermined Pairs: ",
                adjust_nested_list_counting_from_one(undetermined_pairs)
            )

            # Conduct pairwise non-linear regression and independence tests.
            for pair in undetermined_pairs:

                print(
                    "       * For Undetermined Pair: ",
                    adjust_list_counting_from_one(pair)
                )

                determined = False

                p_value_max = nonlinear_mlc.pc_alpha
                causation = copy_and_rename(pair)

                # Unravel the pairwise inferred directions respectively.
                pair_temp = cp.copy(pair)
                pair_temp.reverse()
                pair_reversed = copy_and_rename(pair_temp)

                for cause, effect in zip(pair, pair_reversed):

                    print(
                        "           * For Hypothetical Direction: ",
                        (cause + 1), "->", (effect + 1)
                    )

                    # ========== Empirical Regressor Construction ==========

                    # initialization of explanatory-and-explained variables
                    explanatory_vars = set()
                    explained_var = set()

                    # Cancel duplicates raised by controlled testing.
                    cancel_test_duplicates()

                    # basic explanatory-and-explained variables: cause-effect
                    explanatory_vars.add(cause)
                    explained_var.add(effect)  # namely the effect variable

                    # Add explanatory variables to strengthen empirical regression:

                    # determined parent-relations amidst the algorithm memory
                    explanatory_vars = explanatory_vars | set(nonlinear_mlc._parents_set[effect])

                    # undetermined connections within the maximal clique
                    explanatory_vars = explanatory_vars | (
                            set(undetermined_maximal_clique) - {effect}
                    )

                    print(
                        "               * Explanatory Variables: ",
                        adjust_list_counting_from_one(list(explanatory_vars))
                    )

                    print(
                        "               * Explain Variable: ",
                        adjust_list_counting_from_one(list(explained_var))
                    )

                    # Regress the effect variable on empirical explanatory variables
                    # (in an attempt to cancel unobserved confounding).

                    explanatory_data = cp.copy(
                        nonlinear_mlc._dataset[:, list(explanatory_vars)]
                    )

                    # namely the data with respect to the effect variable
                    explained_data = cp.copy(
                        nonlinear_mlc._dataset[:, list(explained_var)]
                    )

                    # Notes for developer:
                    # Bug: Reinitialize regressor instance for fitting pairwise data.
                    # Error info: Specify regression for X's feature = 1.

                    # Comment original code and add an IF branch of source code.

                    # # regressing residuals via fitting SCMs
                    # residuals = get_residuals_scm(
                    #     explanatory_data=explanatory_data,
                    #     explained_data=explained_data,
                    #     regressor=nonlinear_mlc.regressor
                    # )

                    # regressing residuals via fitting SCMs
                    if explanatory_data.shape[1] == 1:
                        explanatory_data = check_1dim_array(explanatory_data)
                        explained_data = check_1dim_array(explained_data)

                        regressor = LinearGAM()
                        regressor.fit(explanatory_data, explained_data)
                        est_explained_data = regressor.predict(explanatory_data)
                        est_explained_data = check_1dim_array(est_explained_data)

                        residuals = explained_data - est_explained_data

                    else:
                        residuals = get_residuals_scm(
                            explanatory_data=explanatory_data,
                            explained_data=explained_data,
                            regressor=nonlinear_mlc.regressor
                        )

                    # Cancel duplicates raised by controlled testing.
                    cancel_test_duplicates()

                    # Remove effects of parent-relations from the cause variable
                    # (in an attempt to cancel unobserved confounding).

                    cause_parents = list(nonlinear_mlc._parents_set[cause])

                    # Cancel duplicates raised by controlled testing.
                    cancel_test_duplicates()

                    if len(cause_parents) > 0:

                        # Cancel duplicates raised by controlled testing.
                        cancel_test_duplicates()

                        # Notes for developer:
                        # Bug: Reinitialize regressor instance for fitting pairwise data.
                        # Error info: Specify regression for X's feature = 1.

                        # Comment original code and add an IF branch of source code.

                        # cause_data = get_residuals_scm(
                        #     explanatory_data=nonlinear_mlc._dataset[:, cause_parents],
                        #     explained_data=nonlinear_mlc._dataset[:, cause],
                        #     regressor=nonlinear_mlc.regressor
                        # )

                        if len(cause_parents) == 1:
                            explanatory_data = check_1dim_array(
                                cp.copy(nonlinear_mlc._dataset[:, cause_parents])
                            )
                            explained_data = check_1dim_array(
                                cp.copy(nonlinear_mlc._dataset[:, cause])
                            )

                            regressor = LinearGAM()
                            regressor.fit(explanatory_data, explained_data)
                            est_explained_data = regressor.predict(explanatory_data)
                            est_explained_data = check_1dim_array(est_explained_data)

                            cause_residuals = explained_data - est_explained_data
                            cause_data = copy_and_rename(cause_residuals)

                        else:
                            cause_data = get_residuals_scm(
                                explanatory_data=nonlinear_mlc._dataset[:, cause_parents],
                                explained_data=nonlinear_mlc._dataset[:, cause],
                                regressor=nonlinear_mlc.regressor
                            )

                    else:
                        cause_data = cp.copy(nonlinear_mlc._dataset[:, cause])

                    # ================== Independence Test =================

                    # Conduct the independence test
                    # between the cause variable and regressing residuals.
                    p_value = conduct_ind_test(
                        explanatory_data=cause_data,
                        residuals=residuals,
                        ind_test_method=nonlinear_mlc.ind_test
                    )

                    print("                * P-value: ", round(p_value, 3))

                    # One single inferred causal direction is determined given the
                    # maximal p-value exceeding the threshold of the significant level.
                    if p_value > p_value_max:
                        determined = True

                        p_value_max = p_value
                        causation = (cause, effect)

                if determined:
                    determined_pairs.append(causation)

                    print(
                        "           * Accept the Hypothetical Direction: ",
                        (causation[0] + 1), "->", (causation[1] + 1)
                    )

                # code fragment for testing
                else:
                    print(
                        "           * Cannot Determine the Direction of Undetermined Pair: ",
                        adjust_list_counting_from_one(pair)
                    )

        # ---------------- DIVE OUT CLIQUE-BASED INFERENCE  ----------------

        print(
            "* Update Current Determined Paris: ",
            adjust_nested_list_counting_from_one(determined_pairs)
        )

        # Orient the determined causal directions
        # after a search round over maximal cliques.
        graph_pattern_manager.identify_directed_causal_pair(
            determined_pairs=determined_pairs
        )

        # Update the causal adjacency matrix and parent-relations set
        # after a search round over maximal cliques.
        nonlinear_mlc._adjacency_matrix = (
            graph_pattern_manager.managing_adjacency_matrix
        )
        nonlinear_mlc._parents_set = (
            graph_pattern_manager.managing_parents_set
        )

        # Display the change related to the adjacency matrix immediately.
        draw_graph_from_ndarray(
            array=nonlinear_mlc._adjacency_matrix,
            testing_text='partial_adjacency_matrix'
        )

        # Check if new causal relations have been determined
        # after the last round searching
        newly_determined = (
            graph_pattern_manager.check_newly_determined(
                last_undetermined_cliques=undetermined_maximal_cliques
            )
        )

        print("* Newly Determined: ", newly_determined)

        # End if none of new causal relation advancing the further search.
        if not newly_determined:
            continue_search = False
            print("* End the Searching Round\n")
        else:
            print("* Continue the Next Searching Round\n")

    plt.show()


ACTIVATION_0x6 = {

}
REPETITIONS_0x6 = 10


# ### CORRESPONDING TEST ###################################################
# Loc: hybrid_algorithms.py >> NonlinearMLC >> fit
# Loc: hybrid_algorithms.py >> NonlinearMLC >> _clique_based_causal_inference

# ### AUXILIARY COMPONENT(S) ###############################################
# Testing Date: 2024-__-__ | xx:xx (pass)
# --------------------------------------------------------------------------
# Function: TBD

def test_0x6_performance_fitting():
    """
    Testing for numerical and empirical results limited on
    the maximal-clique-based fitting procedure:

    * single complete performance of calling to Nonlinear-MLC;
    * average performance on the general setting;
    * average performance on the priori of causal skeleton;
    """

    # Start the code line

    return


# # ### CORRESPONDING TEST ###################################################
# # Loc: hybrid_algorithms.py >> NonlinearMLC >> fit()
#
# # ### AUXILIARY COMPONENT(S) ##############################################
# # Code: 0x8
# # Function: TBD
# def test_output_general_setting_nonlinear_mlc():
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return
#
#
# # ### CORRESPONDING TEST ###################################################
# # Loc: hybrid_algorithms.py >> NonlinearMLC >> fit()
#
# # ### AUXILIARY COMPONENT(S) ##############################################
# # Code: 0x9
# # Function: TBD
# def test_output_given_skeletal_ground_truth_nonlinear_mlc():
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return
#
#
# # ### CORRESPONDING TEST ###################################################
# # Loc: hybrid_algorithms.py >> MLC-LiNGAM >> fit()
#
# # ### AUXILIARY COMPONENT(S) ##############################################
# # Code: 0xA
# # Function: TBD
# def test_output_general_setting_mlc_lingam():
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return
#
#
# # ### CORRESPONDING TEST ###################################################
# # Loc: hybrid_algorithms.py >> MLC-LiNGAM >> fit()
#
# # ### AUXILIARY COMPONENT(S) ##############################################
# # Code: 0xB
# # Function: TBD
# def test_output_given_skeletal_ground_truth__mlc_lingam():
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return
#
#
# # ### CORRESPONDING TEST ###################################################
# # Loc: hybrid_framework.py >> display_info
#
# # ### AUXILIARY COMPONENT(S) ##############################################
# # Code: 0xC
# # Function: TBD
# def test_display_info():
#     """
#     Write down some descriptions here.
#     """
#
#     # Start the code line
#
#     return
