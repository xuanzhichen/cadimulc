"""Write down some descriptions here."""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * None


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Complete programming of computing f1 score. Back into :
#   * tests of causality instruments: test_get_skeleton_from_pc()
#   * experiment helper: get_skeleton_score (evaluate skeleton)
#                                                             15th.Jan, 2024
#
# * Complete programming of computing precision and recall.   13th.Jan, 2024


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: Complete pair-order evaluation methods for MLC-LiNGAM, it might be
#       used in tutorial sections (Optional).
#
# Done:
# TODO: Add supplemental method of 'evaluate_skeleton()'.
# TODO: Complete the rest of evaluation-metric coding (pairwise version).


import networkx as nx


class Evaluator(object):
    """
    Write down some descriptions here.

    * TP (True Positives): The number of the estimated directed pairs that
      are consistent with the true causal pairs. Namely, TP qualifies the
      correct estimation of (truly) causal relationships.

    * FP (False Positives): The number of the estimated directed pairs that
      do not present in the true causal pairs.

    * TN: (True Negatives): The number of the unestimated directed pairs
      that are consistent with the true causal pairs. Namely, TN reflects
      the correct prediction of non-existential causal relationships.

    * FN (False Negatives): The number of the unestimated directed pairs
      that do present in the true causal pairs.
    """

    @staticmethod
    def precision_pairwise(true_graph, est_graph):
        """
        Precision refers to the proportion of correctly estimated directed
        pairs in all estimated directed pairs.

        * The higher the precision, the stronger the correct recognition
          among the individual causal pairs.
        * Precision = TP / (TP + FP) = TP / all estimated directed pairs.

        **Notice**: *precision_pairwise()* only focus on the assessment of
        directed pairs (not bi-directed pairs or undirected pairs).

        Parameters
        ----------
        true_graph : ndarray
        est_graph : ndarray

        Return
        ------
        precision : float
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
    def recall_pairwise(true_graph, est_graph):
        """
        Recall refers to the proportion of correctly estimated directed
        pairs in all true causal pairs.

        * A higher recall score means that the estimated causal graph as a
          whole is more likely to full-cover the true causal relation.
        * Recall = TP / (TP + FN) = TP / all true causal pairs.

        **Notice**: *recall_pairwise()* only focus on the assessment of
        directed pairs (not bi-directed pairs or undirected pairs).

        Parameters
        ----------
        true_graph : ndarray
        est_graph : ndarray

        Return
        ------
        recall : float
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
    def f1_score_pairwise(true_graph, est_graph):
        """
        F1 score amounts to the concordant mean of the precision and recall.

        * F1 score represents the global measurement of causal discovery,
          bring together advantages of both the precision and recall.
        * F1 score = (2 * Precision * Recall) / (Precision + Recall).

        **Notice**: *f1_score_pairwise()* only focus on the assessment of
        directed pairs (not bi-directed pairs or undirected pairs).

        Parameters
        ----------
        true_graph : ndarray
        est_graph : ndarray

        Return
        ------
        f1_score : float
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
    def get_directed_pairs(graph):
        """
        Write down some descriptions here.

        **Notice**: *get_directed_pairs()* only focus on the assessment of
        directed pairs (not bi-directed pairs or undirected pairs).

        Parameters
        ----------
        graph : ndarray

        Return
        ------
        direct_pairs : list
        """

        dim = graph.shape[0]
        directed_pairs = []

        for j in range(dim):
            for i in range(dim):
                if graph[i][j] == 1 and graph[j][i] == 0:
                    directed_pairs.append([j, i])

        return directed_pairs

    @staticmethod
    def get_pairwise_info(graph):
        """
        Write down some descriptions here.

        **Notice**: *get_pairwise_info()* only focus on the assessment of
        directed pairs (not bi-directed pairs or undirected pairs).

        Parameters
        ----------
        graph : ndarray

        Return
        ------
        num_directed_pairs : int
        dict_directed_parent : dictionary (key: int; value: set)
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

    # @staticmethod
    # def precision_pair_order(true_dag, est_dag):
    #     """
    #     Write down some descriptions here.
    #
    #     Parameters
    #     ----------
    #     true_dag : ndarray
    #     est_dag : ndarray
    #
    #     Return
    #     ------
    #     precision : float
    #     """
    #
    #     precision = 0
    #
    #     return precision
    #
    # @staticmethod
    # def recall_pair_order(true_dag, est_dag):
    #     """
    #     Write down some descriptions here.
    #
    #     Parameters
    #     ----------
    #     true_dag : ndarray
    #     est_dag : ndarray
    #
    #     Return
    #     ------
    #     recall : float
    #     """
    #
    #     recall = 0
    #
    #     return recall
    #
    # @staticmethod
    # def f1_score_pair_order(true_dag, est_dag):
    #     """
    #     Write down some descriptions here.
    #
    #     Parameters
    #     ----------
    #     true_dag : ndarray
    #     est_dag : ndarray
    #
    #     Return
    #     ------
    #     f1_score : float
    #     """
    #
    #     f1_score = 0
    #
    #     return f1_score

    @staticmethod
    def evaluate_skeleton(true_skeleton, est_skeleton, metric):
        """
        Write down some descriptions here.

        * Precision = TP (of the estimated skeleton) / all estimated edges.
        * Recall = TP (of the estimated skeleton) / all true edges.
        * F1 score = (2 * Precision * Recall) / (Precision + Recall).

        **Notice**: *evaluate_skeleton()* is a supplemental method builded
        in Evaluator() since evaluating the estimated skeleton becomes
        practical for hybrid algorithms (skeleton + orientation) in *cadimulc*.

        Parameters
        ----------
        true_skeleton : ndarray
        est_skeleton : ndarray
        metric : string
            'Precision', 'Recall', or 'F1-score'.

        Return
        ------
        precision : float
        recall : float
        f1_score : float
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
            raise ValueError("Please input established metric type: "
                             "'Precision', 'Recall', or 'F1-score'.")



