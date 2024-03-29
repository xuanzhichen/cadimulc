"""The causal graph evaluation corresponding to confusion matrix"""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License

# Testing: cadimulc/tests/test_evaluation.py


# ### DEVELOPMENT PROGRESS (LEAST) ########################################################
# * Programming of the f1 score calculation (skeleton evaluation) was done. Back to:
#   * Tests of causality instruments: test_get_skeleton_from_pc()
#   * Experiment helper: get_skeleton_score()
#                                                                             15th.Jan, 2024
#
# * Programming of the precision and recall calculation was done.             13th.Jan, 2024


# ### TO-DO LIST (LEAST) ##################################################################
# Required (Optional):
# TODO: Causal-order evaluation methods for MLC-LiNGAM,
#       which might be used in tutorials. (Optional)
#
# Done:
# _TODO: Add a supplemental method of 'evaluate_skeleton()'.
# _TODO: Compared to causal-order evaluation, build the causal-pair evaluation.


from numpy import ndarray

import networkx as nx


# ### CODING DATE #########################################################################
# Module Init   : 2024-01-10
# Module Update : 2024-03-29

class Evaluator(object):
    """
    Given an instance as to causal discovery, the `Evaluator` defines the **classification
    errors** between an actual graph and a predicted graph, which is corresponding to,
    in the field of machine learning,
    the **four of the categories** within a **confusion matrix**.

    * **TP (True Positives)**: The number of the estimated directed pairs that
      are consistent with the true causal pairs. Namely, TP qualifies the
      correct estimation of causal relations.

    * **FP (False Positives)**: The number of the estimated directed pairs that
      do not present in the true causal pairs.

    * **TN: (True Negatives)**: The number of the unestimated directed pairs
      that are consistent with the true causal pairs. TN reflects
      the correct prediction of unpresented causal relations.

    * **FN (False Negatives)**: The number of the unestimated directed pairs
      that do present in the true causal pairs.

    !!! warning "The only assessment of directed causal relations"
        The `Evaluator` focuses on the assessment of estimated directed pairs (TP)
        extracted from an adjacency matrix, treating the rest as unpresented pairs (FP)
        relative to the ground-truth.

        In other words, `Evaluator` in CADIMULC does not explicitly consider bi-directed
        pairs or undirected pairs.
    """

    @staticmethod
    def precision_pairwise(true_graph: ndarray, est_graph: ndarray) -> float:
        """
        **Causal pair precision** refers to the proportion of the correctly estimated
        directed pairs in all the **estimated directed pairs** (EDP):

        $$
            Precision = TP \ / \  (TP + FP) = TP \ / \ EDP.
        $$

        The higher the precision, the larger the amount of the causal pairs,
        compared to EDP, that are identified,
        without considering the amount of **unestimated** pairs.

        Parameters:
            true_graph: True causal graph, namely the ground-truth.
            est_graph: Estimated causal graph, namely the empirical causal graph.
            learned from data.

        Returns:
            precision: Precision of the "causal discovery task".
        """

        _, dict_directed_parent = Evaluator.get_pairwise_info(true_graph)
        est_pairs = Evaluator.get_directed_pairs(est_graph)
        num_est_pairs = len(est_pairs)

        if num_est_pairs > 0:
            tp = 0

            for est_pair in est_pairs:
                child = est_pair[1]
                parent = est_pair[0]

                if parent in dict_directed_parent[child]:
                    tp += 1

            precision = round((tp / num_est_pairs), 3)

        else:
            precision = float(0)

        return precision

    @staticmethod
    def recall_pairwise(true_graph: ndarray, est_graph: ndarray) -> float:
        """
        **Causal pair recall** refers to the proportion of correctly estimated directed
        pairs in all **true causal pairs** (TCP):

        $$
            Recall = TP \ / \  (TP + FN) = TP \ / \  TCP
        $$

        The higher the recall, the larger the amount of the causal pairs,
        compared to TCP,
        that are identified, without considering the amount of **incorrectly** estimated pairs.

        Parameters:
            true_graph: True causal graph, namely the ground-truth.
            est_graph: Estimated causal graph, namely the empirical causal graph.

        Returns:
            recall: Recall of the "causal discovery task".
        """

        num_directed_pairs, dict_directed_parent = (
            Evaluator.get_pairwise_info(true_graph))

        est_pairs = Evaluator.get_directed_pairs(est_graph)
        num_est_pairs = len(est_pairs)

        if num_directed_pairs == 0:
            recall = float(1)

        elif num_est_pairs == 0:
            recall = float(0)

        else:
            tp = 0

            for est_pair in est_pairs:
                child = est_pair[1]
                parent = est_pair[0]

                if parent in dict_directed_parent[child]:
                    tp += 1

            recall = round((tp / num_directed_pairs), 3)

        return recall

    @staticmethod
    def f1_score_pairwise(true_graph: ndarray, est_graph: ndarray) -> float:
        """
        **Causal pair F1-score**, the concordant mean of the precision and recall,
        represents the global measurement of causal discovery, bring together the
        advantages from both the precision and recall.

        $$
            F1 = (2 * Precision * Recall)\  / \ (Precision + Recall).
        $$

        Parameters:
            true_graph: True causal graph, namely the ground-truth.
            est_graph: Estimated causal graph, namely the empirical causal graph.

        Returns:
            f1_score: F1-score of the "causal discovery task".
        """

        precision = Evaluator.precision_pairwise(true_graph, est_graph)
        recall = Evaluator.recall_pairwise(true_graph, est_graph)

        if (precision + recall) != 0:
            f1_score = round(
                (2 * precision * recall) / (precision + recall), 3
            )

        else:
            f1_score = float(0)

        return f1_score

    @staticmethod
    def get_directed_pairs(graph: ndarray) -> list[list]:
        """
        Extract directed pairs from a graph.

        Parameters:
            graph: An adjacency bool matrix representing the causation among variables.

        Returns:
            direct_pairs:
                A list whose elements are in form of [parent, child], referring to the
                causation parent -> child.
        """

        dim = graph.shape[0]
        directed_pairs = []

        for j in range(dim):
            for i in range(dim):
                if graph[i][j] == 1 and graph[j][i] == 0:
                    directed_pairs.append([j, i])

        return directed_pairs

    @staticmethod
    def get_pairwise_info(graph: ndarray) -> (int, dict):
        """
        Obtain information related to a given directed graph:
        (1) number of the directed pairs; (2) parents-child pairing relationships.

        Parameters:
            graph: An adjacency bool matrix representing the causation among variables.

        Returns:
            `num_directed_pairs` as the number of directed pairs and `directed_parent_dict`
             as the dictionary representing the parent-child pairing relationships.
        """

        directed_parent_dict = {}
        num_directed_pairs = 0
        dim = graph.shape[0]

        for i in range(dim):
            directed_parent_dict[i] = set()

        for i in range(dim):
            for j in range(dim):
                if graph[i][j] == 1:
                    directed_parent_dict[i].add(j)
                    num_directed_pairs += 1

        return num_directed_pairs, directed_parent_dict

    @staticmethod
    def evaluate_skeleton(
            true_skeleton: ndarray,
            est_skeleton: ndarray,
            metric: str
    ) -> float:
        """
        The `evaluate_skeleton` method evaluates a network skeleton based on an assigned
        metric. To this end, available metrics mirroring to the causal pair evaluation
         are list as the following:

        * **Skeleton Precision** = TP (of the estimated skeleton) / all estimated edges.
        * **Skeleton Recall** = TP (of the estimated skeleton) / all true edges.
        * **Skeleton F1-score** = (2 * Precision * Recall) / (Precision + Recall).

        Parameters:
            true_skeleton: True causal skeleton, namely the ground-truth.
            est_skeleton: Estimated causal skeleton, namely the empirical causal skeleton.
            metric: selective metrics from `['Precision', 'Recall', or 'F1-score']`.

        Returns:
            The evaluating value of the causal skeleton in light of the assigned metric.
        """

        true_skeleton_nx = nx.from_numpy_array(true_skeleton)
        est_skeleton_nx = nx.from_numpy_array(est_skeleton)

        true_edges = list(true_skeleton_nx.edges())
        est_edges = list(est_skeleton_nx.edges())

        tp_skeleton = 0
        for est_edge in est_edges:
            if est_edge in true_edges:
                tp_skeleton += 1

        if len(est_edges) > 0:
            precision = round(tp_skeleton / len(est_edges), 3)
        else:
            precision = float(0)

        if len(true_edges) > 0:
            recall = round(tp_skeleton / len(true_edges), 3)
        else:
            recall = float(0)

        if precision + recall > 0:
            f1_score = round(
                2 * ((precision * recall) / (precision + recall)), 3
            )
        else:
            f1_score = float(0)

        if metric == 'Precision':
            return precision

        elif metric == 'Recall':
            return recall

        elif metric == 'F1-score':
            return f1_score

        else:
            raise ValueError("Please input established metric types: "
                             "'Precision', 'Recall', or 'F1-score'.")



