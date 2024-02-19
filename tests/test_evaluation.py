"""Write down some descriptions here."""

# ### DEVELOPMENT NOTES (LEAST) ############################################
# * None


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Testing-code for the skeleton evaluator is done.          18th.Jan, 2024
#
# * Testing-code for the pairwise evaluator is done.          15th.Jan, 2024
#
# * Fix the bug of computing comparison of precision.         13th.Jan, 2024
#
# * Improve testing structure with repetitive and segmented nature, and
#   computation of precision fails to passing the test.       12th.Jan, 2024


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# None
#
# Done:
# TODO: Prepare for the supplemental testing of 'evaluate_skeleton()'.
# TODO: Testing for "recall" and "f1 score" is completed.
# TODO: Manage and inspect for the strong fault (random seed no.6).
# TODO: Add repetitive and segmented nature for testing code (preparation).


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy as cp

from cadimulc.utils.extensive_modules import (
    display_test_section_symbols,
    copy_and_rename
)
from cadimulc.utils.visualization import draw_graph_from_ndarray

from cadimulc.utils.generation import Generator
from cadimulc.utils.evaluation import Evaluator


REPETITIONS_0x1 = 10
ACTIVATION_0x1 = {
    'testing_part_one':   False,
    'testing_part_two':   False,
    # 'testing_part_three': False,
    # 'testing_part_four':  False,
}


# Code: 0x1
def test_evaluation_pairwise():
    """
    Write down some descriptions here.
    """

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

            # ==================================================================

            generator = Generator(
                graph_node_num=10,
                sample=42,
            )
            generator._generate_dag(sparsity=0.7)
            true_graph = generator.dag

            est_graph = cp.copy(true_graph)
            num_directed_pairs, _ = Evaluator.get_pairwise_info(true_graph)
            directed_pairs = Evaluator.get_directed_pairs(true_graph)

            # ------------------------------------------------------------------

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

            if ACTIVATION_0x1['testing_part_one']:
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

            # ------------------------------------------------------------------

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

            if ACTIVATION_0x1['testing_part_two']:
                display_test_section_symbols()

                draw_graph_from_ndarray(est_graph, testing_text='est_graph')
                plt.show()

            # ==================================================================

            precision_expected = round((tp / num_est_pairs), 3)

            precision_actual = Evaluator.precision_pairwise(
                true_graph=true_graph,
                est_graph=est_graph
            )

            assert precision_actual == precision_expected

            print('\nCase-{}: Pass (Precision)'.format(i))

            # ------------------------------------------------------------------

            recall_expected = round((tp / num_directed_pairs), 3)

            recall_actual = Evaluator.recall_pairwise(
                true_graph=true_graph,
                est_graph=est_graph
            )

            assert recall_actual == recall_expected

            print('\nCase-{}: Pass (Recall)'.format(i))

        except Exception as err_msg:
            print("\nCase-{}: An error occurred:".format(i), err_msg)


# def test_evaluation_pair_order():
#     pass

REPETITIONS_0x2 = 10
ACTIVATION_0x2 = {
    # 'testing_part_two':   True,
    'testing_part_two':   False,

    # 'testing_part_one':   True,
    'testing_part_one':   False,

    # 'testing_part_three': True,
    'testing_part_three': False,

    # 'testing_part_four':  True,
    'testing_part_four':  False,
}


# Code: 0x2
def test_evaluation_skeleton():
    """
    Write down some descriptions here.
    """

    i = 0
    while i < REPETITIONS_0x1:
        random_seed = copy_and_rename(i)
        np.random.seed(random_seed)
        random.seed(random_seed)

        i += 1

        # ==================================================================

        if ACTIVATION_0x2['testing_part_two']:
            testing_case = 4
            i = testing_case
            random_seed_reset = testing_case - 1

            np.random.seed(random_seed_reset)
            random.seed(random_seed_reset)

            ACTIVATION_0x2['testing_part_one'] = True
            ACTIVATION_0x2['testing_part_three'] = True
            ACTIVATION_0x2['testing_part_four'] = True

        # ==================================================================

    # try:

        # --------------------------------------------------------------

        generator = Generator(
            graph_node_num=5,
            sample=42,
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

        if ACTIVATION_0x2['testing_part_one']:
            display_test_section_symbols()

            draw_graph_from_ndarray(
                true_skeleton,
                testing_text='true_skeleton'
            )
            plt.show()

            print("* True Edges: \n", true_edges)

            print("* Redundant Edges: \n", redundant_edges)

        # --------------------------------------------------------------

        # The operating-number cannot be larger than the edge-number.
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

        # typically wrong code fragment
        # tp_skeleton_expected = len(true_edges) - (num_adding + num_missing)

        tp_skeleton_expected = len(true_edges) - num_missing

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

        if ACTIVATION_0x2['testing_part_three']:

            display_test_section_symbols()

            draw_graph_from_ndarray(
                est_skeleton,
                testing_text='est_skeleton'
            )
            plt.show()

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

        # --------------------------------------------------------------

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

        # --------------------------------------------------------------

        f1_score_actual = Evaluator.evaluate_skeleton(
            true_skeleton=true_skeleton,
            est_skeleton=est_skeleton,
            metric='F1-score'
        )

        if ACTIVATION_0x2['testing_part_four']:
            display_test_section_symbols()

            # duplicated code fragment
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

            if len(true_edges) > 0:
                recall_actual = round(tp_skeleton_actual / len(true_edges), 3)

                # # temporally testing code
                # print(" % Actual Recall: ", recall_actual)
            else:
                recall_actual = float(0)

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

            # # temporally testing code
            # print("* Result of Actual Recall == 0 ?")
            # print("* Actual TP of Skeleton: {},"
            #       "  Number of True Edges: {}, ".
            #       format(tp_skeleton_actual, len(true_edges)))

        # --------------------------------------------------------------

        assert f1_score_actual == f1_score_expected
        print("\nCase-{}: Passed".format(i))

    # except Exception as err_msg:
    #     print("\nCase-{}: An error occurred:".format(i), err_msg)

        # ==================================================================

        if ACTIVATION_0x2['testing_part_two']:
            break
        else:
            continue





