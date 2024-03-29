# Source: cadimulc/utils/evaluation.py


# ### DEVELOPMENT NOTES (LEAST) ###########################################################
# * evaluate_skeleton() in test_evaluation.py could serve as a simple template for testing
#   tasks that require typical repetition and flexible interruption.


# ### DEVELOPMENT PROGRESS (LEAST) ########################################################
# * Testing code for pairwise and skeletal evaluation were done.             18th.Jan, 2024
#
# * Fixed a bug (in comparison test) of precision calculation.               13th.Jan, 2024
#
# * Improved testing structures with repetitive and segmented coding.        12th.Jan, 2024


# ### TO-DO LIST (LEAST) ##################################################################
# Done:
# _TODO: Add a supplemental testing of 'evaluation_skeleton()'.
# _TODO: Inspect the precision calculation for the strong faults by random seed no.6.
# _TODO: Add repetitive and segmented coding style (preparation).


from cadimulc.utils.evaluation import Evaluator

from cadimulc.utils.generation import Generator
from cadimulc.utils.visualization import draw_graph_from_ndarray
from cadimulc.utils.extensive_modules import (
    display_test_section_symbols,
    copy_and_rename,
    cancel_test_duplicates
)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy as cp
import random


REPETITIONS_0x1 = 10
ACTIVATION_0x1 = {
    # 'display_parameters_of_metric_calculation': True,
    'display_parameters_of_metric_calculation': False,

    # 'display_operations_to_simulate_an_estimated_graph': True,
    'display_operations_to_simulate_an_estimated_graph': False,
}


# ### CODING DATE #########################################################################
# Testing Stabled: 2024-01-11
# Testing Updated: 2024-03-29

def test_evaluation_pairwise():
    for i in range(REPETITIONS_0x1):
        random_seed = copy_and_rename(i)

        np.random.seed(random_seed)
        random.seed(random_seed)

        # if random_seed == 6:
        #     ACTIVATION_0x1['testing_part_one'] = True
        #     ACTIVATION_0x1['testing_part_two'] = True
        #
        # else:
        #     ACTIVATION_0x1['testing_part_one'] = False
        #     ACTIVATION_0x1['testing_part_two'] = False

        try:
            # ====================== PARAMETERS OF METRIC CALCULATION =====================
            # Initialize the ground-truth and the estimated graph.

            generator = Generator(
                graph_node_num=10,
                sample=1000,
            )
            generator._generate_dag(sparsity=0.7)
            true_graph = generator.dag
            est_graph = cp.copy(true_graph)

            num_directed_pairs, _ = Evaluator.get_pairwise_info(true_graph)
            directed_pairs = Evaluator.get_directed_pairs(true_graph)

            # Manually modify the parameters for metric calculation.

            num_list = np.random.randint(
                low=0,
                high=(num_directed_pairs - 1) / 2,
                size=3
            )
            num_reversed = num_list[0]
            num_bi_directed = num_list[1]
            num_missing = num_list[2]

            num_est_pairs = num_directed_pairs - (num_missing + num_bi_directed)
            tp = num_est_pairs - num_reversed

            if ACTIVATION_0x1['display_parameters_of_metric_calculation']:

                display_test_section_symbols()

                assert num_directed_pairs == len(directed_pairs)

                print("* num_directed_pairs: ", num_directed_pairs)
                print("* directed_pairs: \n", directed_pairs)
                print("* num_reversed:    ", num_reversed)
                print("* num_bi_directed: ", num_bi_directed)
                print("* num_missing:     ", num_missing)
                print("* tp:              ", tp)
                print("* num_est_pairs:   ", num_est_pairs)

                draw_graph_from_ndarray(true_graph, testing_text='true_graph')

                plt.show()

            # ================= OPERATIONS TO SIMULATE AN ESTIMATED GRAPH =================

            for _ in range(num_reversed):
                directed_pair = directed_pairs.pop()
                child = directed_pair[1]
                parent = directed_pair[0]

                est_graph[child][parent] = 0
                est_graph[parent][child] = 1

            for _ in range(num_bi_directed):
                directed_pair = directed_pairs.pop()
                child = directed_pair[1]
                parent = directed_pair[0]

                est_graph[parent][child] = 1

            for _ in range(num_missing):
                directed_pair = directed_pairs.pop()
                child = directed_pair[1]
                parent = directed_pair[0]

                est_graph[child][parent] = 0

            if ACTIVATION_0x1['display_operations_to_simulate_an_estimated_graph']:
                display_test_section_symbols()

                draw_graph_from_ndarray(est_graph, testing_text='est_graph')
                plt.show()

            # =================== TESTING FOR EVALUATION CALCULATION ======================

            # setup for the expected result
            precision_expected = round((tp / num_est_pairs), 3)
            recall_expected = round((tp / num_directed_pairs), 3)

            # calling for the testing module
            precision_actual = Evaluator.precision_pairwise(
                true_graph=true_graph,
                est_graph=est_graph
            )
            recall_actual = Evaluator.recall_pairwise(
                true_graph=true_graph,
                est_graph=est_graph
            )

            # comparison testing

            assert precision_actual == precision_expected
            print('\nCase-{}: Pass (Precision)'.format(i))

            assert recall_actual == recall_expected
            print('\nCase-{}: Pass (Recall)'.format(i))

        except Exception as err_msg:
            print("\nCase-{}: An error occurred:".format(i), err_msg)


REPETITIONS_0x2 = 10
ACTIVATION_0x2 = {
    # 'display_parameters_of_metric_calculation': True,
    'display_parameters_of_metric_calculation': False,

    # 'display_true_and_redundant_edges_given_the_ground':   True,
    'display_true_and_redundant_edges_given_the_ground': False,

    # 'jump_into_implementation_of_skeleton_evaluation':  True,
    'jump_into_implementation_of_skeleton_evaluation':  False,
}


# ### CODING DATE #########################################################################
# Testing Stabled: 2024-01-11
# Testing Updated: 2024-03-29

def test_evaluation_skeleton():
    i = 0
    while i < REPETITIONS_0x2:
        random_seed = copy_and_rename(i)
        np.random.seed(random_seed)
        random.seed(random_seed)

        i += 1

        # ================ CHANGE PROGRAM FLOW FOR SINGLE-CASE CHECK OUT ==================

        # Development notes: Flow of try-except was interrupted for single-case check out.

        if ACTIVATION_0x2['display_true_and_redundant_edges_given_the_ground']:
            testing_case = 4
            i = testing_case
            random_seed_reset = testing_case - 1

            np.random.seed(random_seed_reset)
            random.seed(random_seed_reset)

            ACTIVATION_0x2['display_true_and_redundant_edges_given_the_ground'] = True
            ACTIVATION_0x2['jump_into_implementation_of_skeleton_evaluation'] = True

    # try:

        # ======================== PARAMETERS OF METRIC CALCULATION =======================

        # Initialize the ground-truth (skeleton) and the estimated skeleton.
        generator = Generator(
            graph_node_num=5,
            sample=1000,
        )

        undigraph_np = generator._get_undigraph(
            graph_node_num=generator.graph_node_num,
            sparsity=0.7
        )
        true_skeleton = copy_and_rename(undigraph_np)
        true_skeleton_nx = nx.from_numpy_array(true_skeleton)
        complete_graph_nx = nx.complete_graph(n=generator.graph_node_num)
        all_edges = list(complete_graph_nx.edges())
        true_edges = list(true_skeleton_nx.edges())

        redundant_edges = cp.copy(all_edges)
        for true_edge in true_edges:
            redundant_edges.remove(true_edge)

        if ACTIVATION_0x2['display_true_and_redundant_edges_given_the_ground-truth']:

            display_test_section_symbols()

            draw_graph_from_ndarray(
                true_skeleton,
                testing_text='true_skeleton'
            )

            print("* True Edges: \n", true_edges)
            print("* Redundant Edges: \n", redundant_edges)

            plt.show()

        # Warning: Parameters for the operating-number cannot be larger than the edge-number.
        # num_list = np.random.randint(
        #     low=0,
        #     high=(len(true_edges) - 1),
        #     size=3
        # )

        if (len(redundant_edges) - 1) <= 0:
            num_adding = 0
        else:
            num_adding = np.random.randint(
                low=0,
                high=len(redundant_edges) - 1
            )

        num_missing = np.random.randint(
            low=0,
            high=(len(true_edges) - 1)
        )

        # Warning: the typically wrong code fragment
        # tp_skeleton_expected = len(true_edges) - (num_adding + num_missing)

        tp_skeleton_expected = len(true_edges) - num_missing

        # ================== OPERATIONS TO SIMULATE AN ESTIMATED SKELETON =================

        est_skeleton = cp.copy(undigraph_np)
        for _ in range(num_adding):
            redundant_edge = redundant_edges.pop()
            end_i = redundant_edge[1]
            end_j = redundant_edge[0]

            est_skeleton[end_i][end_j] = 1
            est_skeleton[end_j][end_i] = 1

        true_edges_temp = copy_and_rename(true_edges)
        for _ in range(num_missing):
            true_edge = true_edges_temp.pop()
            end_i = true_edge[1]
            end_j = true_edge[0]

            est_skeleton[end_i][end_j] = 0
            est_skeleton[end_j][end_i] = 0

        est_skeleton_nx = nx.from_numpy_array(est_skeleton)
        est_edges = list(est_skeleton_nx.edges())

        if ACTIVATION_0x2['display_parameters_of_metric_calculation']:

            display_test_section_symbols()

            draw_graph_from_ndarray(
                est_skeleton,
                testing_text='est_skeleton'
            )

            print("* Estimated Edges: \n", est_edges)

            print("* Expected TP of Skeleton: {}, "
                  "Number of Estimated Edges: {}, "
                  "Number of True Edges: {}". format(
                    tp_skeleton_expected,
                    len(est_edges),
                    len(true_edges)))

            print("* Number of Adding Edges: {},"
                  "  Number of Missing Edges: {}".
                  format(num_adding, num_missing))

            plt.show()

        # ===================== TESTING FOR EVALUATION CALCULATION ========================

        # setup for the expected result
        precision_expected = round(
            tp_skeleton_expected / len(est_edges), 3
        )
        recall_expected = round(
            tp_skeleton_expected / len(true_edges), 3
        )

        if precision_expected + recall_expected > 0:
            f1_score_expected = round(
                    2 * ((precision_expected * recall_expected) /
                         (precision_expected + recall_expected)),
                    3
            )
        else:
            f1_score_expected = float(0)

        # calling for the testing module
        f1_score_actual = Evaluator.evaluate_skeleton(
            true_skeleton=true_skeleton,
            est_skeleton=est_skeleton,
            metric='F1-score'
        )

        if ACTIVATION_0x2['jump_into_implementation_of_skeleton_evaluation']:

            display_test_section_symbols()

            # true_skeleton_nx = nx.from_numpy_array(true_skeleton)
            est_skeleton_nx = nx.from_numpy_array(est_skeleton)

            true_edges = list(true_skeleton_nx.edges())
            est_edges = list(est_skeleton_nx.edges())

            tp_skeleton_actual = 0
            for est_edge in est_edges:
                if est_edge in true_edges:
                    tp_skeleton_actual += 1

            print("* Actual TP of Skeleton: {}, "
                  "Number of Estimated Edges: {}, "
                  "Number of True Edges: {}".format(
                    tp_skeleton_actual,
                    len(est_edges),
                    len(true_edges)))

            if len(est_edges) > 0:
                precision_actual = round(tp_skeleton_actual / len(est_edges), 3)
            else:
                precision_actual = float(0)

            cancel_test_duplicates()

            if len(true_edges) > 0:
                recall_actual = round(tp_skeleton_actual / len(true_edges), 3)

                # code fragment for temporal testing
                # print(" % Actual Recall: ", recall_actual)
            else:
                recall_actual = float(0)

            cancel_test_duplicates()

            if precision_actual + recall_actual > 0:
                f1_score_actual = round(
                    2 * ((precision_actual * recall_actual) /
                         (precision_actual + recall_actual)),
                    3
                )
            else:
                f1_score_actual = float(0)

            print("* Expected Precision: {},"
                  "  Expected Recall: {},"
                  "  Expected F1-score: {}".
                  format(precision_expected,
                         recall_expected,
                         f1_score_expected))

            print("* Actual Precision: {},"
                  "  Actual Recall: {},"
                  "  Actual F1-score: {}".
                  format(precision_actual,
                         recall_actual,
                         f1_score_actual))

            # code fragment for temporal testing
            # print("* Result of Actual Recall == 0 ?")
            # print("* Actual TP of Skeleton: {},"
            #       "  Number of True Edges: {}, ".
            #       format(tp_skeleton_actual, len(true_edges)))

        # comparison testing
        assert f1_score_actual == f1_score_expected
        print("\nCase-{}: Passed".format(i))

    # except Exception as err_msg:
    #     print("\nCase-{}: An error occurred:".format(i), err_msg)

        if ACTIVATION_0x2['display_parameters_of_metric_calculation']:
            break
        else:
            continue





