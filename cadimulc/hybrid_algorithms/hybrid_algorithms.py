"""Non-linear causal discovery with multiple latent confounders."""

# Author:  Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License

# Testing: ../tests/test_hybrid_algorithms.py


# ### DEVELOPMENT NOTES (LEAST) ###########################################################
# * To be compatible with the docs theme, docstring standards are slightly modified. e.g.
#   Attributes are specified as private parameters, but commented in the same Class in
#   'generation.py'; herein attributes are loaded from the base but explicitly displayed.
#
# * The fact that the procedure of identifying leaf variables in MLC-LiNGAM comes to be
#   necessary hints that the exogenous identification may get "stuck" by latent confounding.
#
# * Implementation of the time-series-Nonlinear-MLC version are migrated and commented in
#   the file `../paper_2023/nonlinearmlc.py`, acting as a static snapshot for reproduction.


# ### DEVELOPMENT PROGRESS (LEAST) ########################################################
# * Programming of the hybrid algorithms were (virtually) done.              15th.Feb, 2024
#
# * Main implementations of the algorithms were completed, adding a graph pattern manager to
#   specify the maximal-clique recognition and reduce duplicate code.        29th.Jan, 2024
#
# * Reconstructed the initial frameworks of Nonlinear-MLC and MLC-LiNGAM in one file, adding
#   a base-class and some static-methods.                                    21st.Jan, 2024
#
# * Started externally with adding a plotting function and internally with the data stream
#   interactions. Encapsulating the modules is urgent when going further in
#   coding the "stages learning".                                            01st.Nov, 2023


# ### GENERAL TO-DO LIST (LEAST) ##########################################################
# Required (Optional):
# TODO:  For (theoretically) determining conflict causal orders, add p-value statistic
#        based on Fisher’s method into 'identify_partial_causal_order()'. (optional)
#
# Done:
# _TODO: Start reconstruction of Nonlinear-MLC with the clique-based inference, and MLC-LiNGAM
#       with the stage-2 learning.
# _TODO: Encapsulate modules: regression and ind-test.


from __future__ import annotations

# hybrid causal discovery framework
from .hybrid_framework import HybridFrameworkBase

# auxiliary modules in causality instruments
from cadimulc.utils.causality_instruments import (
    get_residuals_scm,
    conduct_ind_test,
)
# linear and non-linear regression
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM

# basic
from cadimulc.utils.extensive_modules import (
    check_1dim_array,
    copy_and_rename
)
from numpy import ndarray

import numpy as np
import networkx as nx
import copy as cp
import time
import warnings
warnings.filterwarnings("ignore")


class GraphPatternManager(object):
    """
    An auxiliary module embedded in the MLC-LiNGAM and Nonlinear-MLC algorithms,
    featuring the algorithmic behavior of the **maximal-cliques pattern** recognition.

    The module as well manages adjacency matrices amidst the procedure between
    causal skeleton learning and causal direction orientation.
    """

    def __init__(
            self,
            init_graph: ndarray,
            managing_adjacency_matrix: ndarray | None = None,
            managing_adjacency_matrix_last: ndarray | None = None,
            managing_parents_set: dict | None = None
    ):
        """
        Parameters:
            init_graph:
                A graphical structure to be managed, usually the causal skeleton.

            managing_adjacency_matrix:
                The dynamic-altering adjacency matrix that is initialized with
                the init_graph and continuously changing during all learning stages,
                ending up with a (partially) oriented adjacency matrix.

            managing_adjacency_matrix_last:
                A temporary copy of the current managing_adjacency_matrix ahead of
                the next learning stage.

            managing_parents_set:
                Record child-parents relations associating with the adjacency matrix.
        """

        self.init_graph = init_graph
        self._managing_skeleton = cp.copy(init_graph)

        if managing_adjacency_matrix is None:
            self.managing_adjacency_matrix = cp.copy(init_graph)
            self.managing_adjacency_matrix_last = cp.copy(init_graph)
        else:
            self.managing_adjacency_matrix = managing_adjacency_matrix
            self.managing_adjacency_matrix_last = managing_adjacency_matrix_last

        if managing_parents_set is None:
            self.managing_parents_set = {}
            for variable_index in range(init_graph.shape[0]):
                self.managing_parents_set[variable_index] = set()
        else:
            self.managing_parents_set = managing_parents_set

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x3_procedure_stage_two_learning
    # Loc: test_hybrid_algorithms.py >> test_0x7_procedure_stage_three_learning

    # ### CODING DATE #####################################################################
    # Module Stable: 2024-01-25
    # Module Update: 2024-04-02

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: MLCLiNGAM -> _algorithm_2

    def identify_partial_causal_order(
            self,
            k_head,
            k_tail,
            _specify_adjacency=False,
            _adjacent_set=None
    ):
        """
        Identify a partial causal order by the K-Head and K-Tail list,
        simultaneously updating ``_managing_adjacency_matrix`` and ``managing_parents_set``.
        """

        def check_unidentified_causal_order(x, y):
            determined_state_xy = self.managing_adjacency_matrix[x][y]
            determined_state_yx = self.managing_adjacency_matrix[y][x]

            return True if determined_state_xy and determined_state_yx else False

        # Orientation by K-Head and K-Tail list should on the adjacent baseline.
        if (not _specify_adjacency) or (_adjacent_set is None):
            # Get the general adjacency relations among variables from the causal skeleton.
            adjacent_set = GraphPatternManager.find_adjacent_set(
                causal_skeleton=self._managing_skeleton
            )
        else:
            # Specify the adjacency set normally for determining the causal order in a subset.
            adjacent_set = _adjacent_set

        # Development notes: In accord with the pseudocode in MLC-LiNGAM:
        #   x_i or i: refer to the exogenous variable
        #   x_j or j: refer to the leaf variable

        # Orient the edge from exogenous variables to their adjacent variables.
        k_head_temp = []
        for i in k_head:
            for j in (adjacent_set[i] - set(k_head_temp)):
                if check_unidentified_causal_order(i, j):
                    self.managing_adjacency_matrix[j][i] = 1
                    self.managing_adjacency_matrix[i][j] = 0

                    # Update the relative parent set.
                    self.managing_parents_set[j].add(i)
                else:
                    # Development notes: Simplify process for conflict causal orders.
                    continue

            k_head_temp.append(i)

        # Orient the edge from leaf-variable-adjacent counterparts to leaf variables.
        k_tail_temp = []
        for j in reversed(k_tail):

            # Development notes: Check out the code from the old implementation versions.
            # # Avoid traverse on an int type.
            # if type(i_adj) is int or type(i_adj) is np.int32:

            for i in (adjacent_set[j] - (set(k_head) | set(k_tail_temp))):
                if check_unidentified_causal_order(i, j):
                    self.managing_adjacency_matrix[i][j] = 0
                    self.managing_adjacency_matrix[j][i] = 1

                    # Update the relative parent set.
                    self.managing_parents_set[j].add(i)

            k_tail_temp.append(j)

        return self

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x2_graph_pattern_manager

    # ### CODING DATE #####################################################################
    # Module Init   : 2024-02-03
    # Module Update : 2024-03-05

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: NonlinearMLC -> fit

    def identify_directed_causal_pair(self, determined_pairs):
        """
        Identify a directed causal pair via simultaneously updating
        ``_managing_adjacency_matrix`` and ``managing_parents_set``.
        """

        if len(determined_pairs) > 0:
            for determined_pair in determined_pairs:
                parent_index = determined_pair[0]
                child_index = determined_pair[1]

                # orientation for the adjacency matrix
                self.managing_adjacency_matrix[child_index][parent_index] = 1
                self.managing_adjacency_matrix[parent_index][child_index] = 0

                # recording relationships for the parent set
                self.managing_parents_set[child_index].add(parent_index)

        return self

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x2_graph_pattern_manager

    # ### CODING DATE #####################################################################
    # Module Init   : 2024-02-03
    # Module Update : 2024-03-04

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: NonlinearMLC -> fit
    # Function: MLCLiNGAM -> _stage_3_learning

    def get_undetermined_cliques(self, maximal_cliques: list[list]) -> list:
        """
        Get the undetermined cliques with respect to (original) maximal cliques
        by checking the latest changing adjacency matrix,
        which implies there is at least one edge remaining undetermined.

        Parameters:
            maximal_cliques:
                The (original) maximal cliques list with the elements
                (maximal clique) in form as well of list.

        Returns:
            undetermined_cliques:
                A list of ``undetermined_cliques``.
        """

        maximal_cliques_undetermined = []

        for maximal_clique in maximal_cliques:
            undetermined = False
            for i in maximal_clique:
                for j in maximal_clique[maximal_clique.index(i):]:

                    # Mark as an undetermined clique if there is
                    # at least one edge remaining undetermined.
                    if (self.managing_adjacency_matrix[i][j] == 1) and (
                        self.managing_adjacency_matrix[j][i] == 1
                    ):
                        undetermined = True
                        break
                if undetermined:
                    break

            if undetermined:
                maximal_cliques_undetermined.append(maximal_clique)

        return maximal_cliques_undetermined

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x2_graph_pattern_manager

    # ### CODING DATE #####################################################################
    # Module Init   : 2024-02-03
    # Module Update : 2024-03-04

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: NonlinearMLC -> fit
    # Function: MLCLiNGAM -> _stage_3_learning

    def check_newly_determined(self, last_undetermined_cliques: list) -> bool:
        """
        Check whether there exists a newly determined edge after the last
        searching round over undetermined cliques.
        This is done by comparing the current managing adjacency matrix with
        the last managing adjacency matrix that is stored ahead of searching.

        Parameters:
            last_undetermined_cliques:
                The undetermined cliques ahead of the last searching round,
                usually gotten by ``get_undetermined_cliques``.

        Returns:
            newly_determined:
                A bool value of ``newly_determined``.
        """

        # Arguments for testing:
        #   managing_adjacency_matrix_last: ndarray

        newly_determined = False

        for undetermined_clique in last_undetermined_cliques:
            for i in undetermined_clique:
                for j in undetermined_clique[undetermined_clique.index(i):]:

                    # Call the determined information of adjacency matrix
                    # relative to the last undetermined cliques.
                    last_info_ij = self.managing_adjacency_matrix_last[i][j]
                    last_info_ji = self.managing_adjacency_matrix_last[j][i]

                    # Mark as newly determined if any current edge within
                    # the range of last undetermined cliques is determined.
                    if (self.managing_adjacency_matrix[i][j] != last_info_ij) or (
                        self.managing_adjacency_matrix[j][i] != last_info_ji
                    ):
                        newly_determined = True
                        break
                if newly_determined:
                    break
            if newly_determined:
                break

        return newly_determined

    def store_last_managing_adjacency_matrix(self):
        """ Store the last managing adjacency matrix, usually for checking
            the newly determined edge after the next round search.
        """
        self.managing_adjacency_matrix_last = cp.copy(
            self.managing_adjacency_matrix
        )
        return self

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x2_graph_pattern_manager

    # ### CODING DATE #####################################################################
    # Module Stable : 2024-03-04
    # Module Update : 2024-04-02

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: NonlinearMLC -> fit
    # Function: MLCLiNGAM -> _stage_3_learning

    @staticmethod
    def recognize_maximal_cliques_pattern(
            causal_skeleton: ndarray,
            adjacency_matrix: ndarray | None = None
    ) -> list[list]:
        """
        Recognize the maximal-cliques pattern based on a causal skeleton (undirected graph)
        or a partial causal adjacency matrix (partially directed graph).

        !!! note "Reference by ``recognize_maximal_cliques_pattern``"
            Bron, Coen, and Joep Kerbosch.
            "Algorithm 457: finding all cliques of an undirected graph."
            Communications of the ACM 16, no. 9 (1973): 575-577.
            (Implementation by [NetworkX](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html))

        Parameters:
            causal_skeleton:
                The undirected graph corresponding to the causal graph.
            adjacency_matrix:
                The (partially) directed acyclic graph (DAG) corresponding to the causal graph.

        Returns:
            maximal_cliques:
                A whole list of "maximal cliques", along with each of the
                "maximal clique" element that is as well in form of list.
        """
        # Remove the edge that has been determined as to the skeleton
        if adjacency_matrix is not None:
            dim = adjacency_matrix.shape[0]
            for i in range(dim):
                for j in range(dim):
                    if (adjacency_matrix[i][j] == 1) and (adjacency_matrix[j][i] == 0):
                        causal_skeleton[i][j] = 0
                        causal_skeleton[j][i] = 0

        # Search maximal cliques by the Bron-Kerbosch algorithm.
        undirected_graph_nx = nx.from_numpy_array(causal_skeleton)
        clique_iter = nx.find_cliques(undirected_graph_nx)
        maximal_cliques_temp = [clique for clique in clique_iter]

        # Remove the trivial graph from maximal_cliques.
        maximal_cliques = []
        for clique in maximal_cliques_temp:
            if len(clique) > 1:
                maximal_cliques.append(clique)

        return maximal_cliques

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x3_procedure_stage_two_learning

    # ### CODING DATE #####################################################################
    # Module Init   : 2024-01-30

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: MLCLiNGAM -> _stage_2_learning

    @staticmethod
    def find_adjacent_set(causal_skeleton: ndarray) -> dict:
        """
        Given a causal skeleton (or a subset of the causal skeleton),
        find out the adjacent variables for each of the variable.

        Parameters:
            causal_skeleton:
                The undirected graph corresponding to the causal graph.

        Returns:
            adjacent_set:
                A dictionary that describes the adjacency relations:
                `{variable: (adjacent variables)}`.
        """

        dim = causal_skeleton.shape[1]
        adjacent_set = {}

        for i in range(dim):
            adjacent_set[i] = set()

        for i in range(dim):
            for j in range(dim):
                if i != j:
                    if causal_skeleton[i][j] == 1:
                        adjacent_set[i].add(j)

        return adjacent_set

    @property
    def managing_skeleton_(self):
        return self._managing_skeleton


class NonlinearMLC(HybridFrameworkBase):
    """
    The **hybrid** algorithm *Nonlinear-MLC*, incorporation of the
    constraint-based and functional-based causal discovery methodology,
    is developed for the **general causal inference** over non-linear data in presence of
    multiple unknown factors.

    A primary feature of *Nonlinear-MLC* lies in exploiting the
    **non-linear causal identification with multiple latent confounders**
    (proposed as the **Latent-ANMs causal identification**),
    which is on the basic of the well-known **ANMs<sup>*</sup>** method.

    !!! note "The ANMs causal discovery approach"
        ANMs, the additive-noise-models, is known as one of the
        [structural-identifiable SCMs](https://xuanzhichen.github.io/cadimulc/generation/).

    <!--
    References:
    Chen, Xuanzhi, Wei Chen, Ruichu Cai.
    "Non-linear Causal Discovery for Additive Noise Model with
    Multiple Latent Confounders". *Xuanzhi's Personal Website.* 2023.

    https://xuanzhichen.github.io/work/papers/nonlinear_mlc.pdf

    Hoyer, Patrik, Dominik Janzing, Joris M. Mooij, Jonas Peters, and Bernhard Schölkopf.
     "Nonlinear causal discovery with additive noise models."
     *Advances in neural information processing systems*. 2008.

     https://proceedings.neurips.cc/paper/2008/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html
    -->
    """

    def __init__(
            self,
            ind_test: str = 'kernel_ci',
            pc_alpha: float = 0.05,
            _regressor: object = LinearGAM()
    ):
        """
        Parameters:
            ind_test:
                Popular non-linear independence-tests methods are recommended:
                Kernel-based Conditional Independence tests (KCI); Hilbert-Schmidt
                Independence Criterion for General Additive Models (HSIC-GAMs).

            pc_alpha:
                Significance level of independence tests (p_value), which is required by
                the constraint-based methodology incorporated in the initial stage of
                the hybrid causal discovery framework.

        <!--
        Attributes:
            _regressor:
                Built-in non-linear regressor module: LinearGAM (General Additive Models).
        -->
        """

        HybridFrameworkBase.__init__(self, pc_alpha=pc_alpha)

        self.ind_test = ind_test
        self.regressor = _regressor

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x5_procedure_fitting
    # Loc: test_hybrid_algorithms.py >> test_0x6_performance_fitting

    # ### CODING DATE #####################################################################
    # Module Stable: 2024-03-05

    # ### AUXILIARY COMPONENT(S) ##########################################################
    # Function: _clique_based_causal_inference (self)
    # Function: recognize_maximal_cliques_pattern
    # Function: get_undetermined_cliques
    # Function: check_newly_determined
    # Class:    GraphPatternManager

    def fit(self, dataset: ndarray) -> object:
        """
        Fitting data via the *Nonlinear-MLC* causal discovery algorithm.

        The procedure comprises the **causal skeleton learning** in the initial stage,
        along with the causal identification procedure involving **non-linear regression**
        and **independence tests** for the subsequence.
        Following the well-known **divide-and-conquer** strategy, non-linear causal inference
        are conducted over the maximal cliques recognized from the estimated causal skeleton.

        Parameters:
            dataset:
                The observational dataset shown as a matrix or table,
                with a format of "sample (n) * dimension (d)."
                (input as Pandas dataframe is also acceptable)

        Returns:
            self:
                Update the ``adjacency_matrix`` represented as an estimated causal graph.
                The ``adjacency_matrix`` is a (d * d) numpy array with 0/1 elements
                characterizing the causal direction.
        """

        start = time.perf_counter()

        # Reconstruct a causal skeleton using the PC-stable algorithm.
        self._causal_skeleton_learning(dataset)

        # Recognize the maximal-clique pattern based on the causal skeleton.
        maximal_cliques = GraphPatternManager.recognize_maximal_cliques_pattern(
            causal_skeleton=self._skeleton
        )

        # Initialize a graph pattern manager for subsequent learning.
        graph_pattern_manager = GraphPatternManager(
            init_graph=self._skeleton
        )

        # Perform the nonlinear-mlc causal discovery.
        continue_search = True
        while continue_search:

            # Obtain the cliques that remain at least one edge undetermined.
            undetermined_maximal_cliques = (
                graph_pattern_manager.get_undetermined_cliques(maximal_cliques)
            )

            # End if all edges over the cliques have been determined.
            if len(undetermined_maximal_cliques) == 0:
                break

            # Temporally store the adjacency matrix ahead of a search round.
            graph_pattern_manager.store_last_managing_adjacency_matrix()

            # In light of the L-ANMs theory (proposed in paper), start the search round
            # by conducting non-linear causal inference based on maximal cliques.
            determined_pairs = self._clique_based_causal_inference(
                undetermined_maximal_cliques=undetermined_maximal_cliques
            )

            # Orient determined causal directions after a search round over maximal cliques.
            graph_pattern_manager.identify_directed_causal_pair(
                determined_pairs=determined_pairs
            )

            # Update the causal adjacency matrix and parent-relations set
            # after a search round over maximal cliques.
            self._adjacency_matrix = (
                graph_pattern_manager.managing_adjacency_matrix
            )
            self._parents_set = (
                graph_pattern_manager.managing_parents_set
            )

            # Check if new causal relations have been determined
            # after the last round searching.
            newly_determined = (
                graph_pattern_manager.check_newly_determined(
                    undetermined_maximal_cliques
                )
            )

            # End if there is none of new causal relation advancing the further search.
            if not newly_determined:
                continue_search = False

        end = time.perf_counter()

        self._running_time += (end - start)

        return self

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: self.fit

    def _clique_based_causal_inference(
            self,
            undetermined_maximal_cliques: list[list]
    ) -> list:
        """
        For each of the undetermined maximal cliques (e.g. at least one edge
        within a maximal clique remains undirected) with respect to the whole
        maximal-clique patterns,
        the algorithm conducts non-linear regression and independence tests with
        the **additional explanatory variables** selected from the
        undetermined maximal clique.

        This strategy is argued to enhance the **efficiency** and **robustness** as to
        the **non-linear causal discovery with multiple latent confounders**,
        serving as the essence of the *Nonlinear-MLC* algorithm (See "Latent-ANMs Lemma" for
        details in the [relevant paper](https://xuanzhichen.github.io/work/papers/nonlinear_mlc.pdf),
        Section 3).

        Parameters:
            undetermined_maximal_cliques:
                A list of undetermined maximal cliques in which the element
                (maximal clique) involves at least one edge remaining undirected
                (e.g. `[[X, Y, Z]]` could stand for a maximal clique <X, Y, Z>,
                with both determined and undetermined relations "
                 'X <-> Y', 'Y <-> Z', and 'X -> Z'").

        Returns:
            The list of determined pairs over the inputting undetermined cliques
             after searching
             (e.g. `[[X, Y], [Y, Z]]` stands for two of the determined pairs
             "X -> Y" and "Y -> Z" after searching).
        """

        # Arguments for Testing:
        #     _adjacency_matrix(attribute): ndarray
        #     _dataset(attribute): ndarray
        #     _parents_set(attribute): dict

        determined_pairs = []

        # Conduct non-linear causal inference based on each maximal clique unit.
        for undetermined_maximal_clique in undetermined_maximal_cliques:

            # Initialize the lists with elements of undetermined causal relations.
            # e.g. the element (cause, effect) specifies "cause -> effect"
            undetermined_pairs = []

            # Get undetermined pairs within a clique.
            for i in undetermined_maximal_clique:
                for j in undetermined_maximal_clique[
                    undetermined_maximal_clique.index(i) + 1:
                ]:
                    if (self._adjacency_matrix[i][j] == 1) and (
                        self._adjacency_matrix[j][i] == 1
                    ):
                        undetermined_pairs.append([i, j])

            # Conduct pairwise non-linear regression and independence tests.
            for pair in undetermined_pairs:
                determined = False

                p_value_max = self.pc_alpha
                causation = copy_and_rename(pair)

                # Unravel the pairwise inferred directions respectively.
                pair_temp = cp.copy(pair)
                pair_temp.reverse()
                pair_reversed = copy_and_rename(pair_temp)

                for cause, effect in zip(pair, pair_reversed):

                    # ================= Empirical Regressor Construction ==================

                    # initialization of explanatory-and-explained variables
                    explanatory_vars = set()
                    explained_var = set()

                    # basic explanatory-and-explained variables: cause-effect
                    explanatory_vars.add(cause)
                    explained_var.add(effect)  # namely the effect variable

                    # Add explanatory variables to strengthen empirical regression:

                    # determined parent-relations amidst the algorithm memory
                    explanatory_vars = explanatory_vars | set(self._parents_set[effect])

                    # undetermined connections within the maximal clique
                    explanatory_vars = explanatory_vars | (
                            set(undetermined_maximal_clique) - {effect}
                    )

                    # Regress the effect variable on empirical explanatory variables
                    # (in an attempt to cancel unobserved confounding).

                    explanatory_data = cp.copy(
                        self._dataset[:, list(explanatory_vars)]
                    )

                    # namely the data with respect to the effect variable
                    explained_data = cp.copy(
                        self._dataset[:, list(explained_var)]
                    )

                    # regressing residuals via fitting SCMs

                    # Development notes:
                    # The following IF branch is added due to a bug, occurring when
                    # reinitializing the regressor (GAM) instance for fitting pairwise data.
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
                            regressor=self.regressor
                        )

                    # Remove effects of parent-relations from the cause variable
                    # (in an attempt to cancel unobserved confounding).

                    cause_parents = list(self._parents_set[cause])

                    if len(cause_parents) > 0:
                        # Development notes:
                        # The following IF branch is added due to a bug, occurring when
                        # reinitializing the regressor (GAM) instance for fitting pairwise data.
                        if len(cause_parents) == 1:
                            cause_explanatory_data = check_1dim_array(
                                cp.copy(self._dataset[:, cause_parents])
                            )
                            cause_explained_data = check_1dim_array(
                                cp.copy(self._dataset[:, cause])
                            )

                            regressor = LinearGAM()
                            regressor.fit(cause_explanatory_data, cause_explained_data)
                            est_cause_explained_data = regressor.predict(cause_explanatory_data)
                            est_cause_explained_data = check_1dim_array(est_cause_explained_data)

                            cause_residuals = cause_explained_data - est_cause_explained_data
                            cause_data = copy_and_rename(cause_residuals)
                        else:
                            cause_data = get_residuals_scm(
                                explanatory_data=self._dataset[:, cause_parents],
                                explained_data=self._dataset[:, cause],
                                regressor=self.regressor
                            )
                    else:
                        cause_data = cp.copy(self._dataset[:, cause])

                    # ========================== Independence Test ========================

                    # Conduct the independence test
                    # between the cause variable and regressing residuals.
                    p_value = conduct_ind_test(
                        explanatory_data=cause_data,
                        residuals=residuals,
                        ind_test_method=self.ind_test
                    )

                    # One single inferred causal direction is determined given the
                    # maximal p-value exceeding the threshold of the significant level.
                    if p_value > p_value_max:
                        determined = True

                        p_value_max = p_value
                        causation = (cause, effect)

                if determined:
                    determined_pairs.append(causation)

        return determined_pairs


class MLCLiNGAM(HybridFrameworkBase):
    """
    *MLC-LiNGAM stands* for a **hybrid** causal discovery method for the **LiNGAM** approach
    with **multiple latent confounders**.
    It serves as an enhancement of **LiNGAM<sup>*</sup>** via combining the advantages of
    **constraint-based** and **functional-based** causality methodology.

    !!! note "The LiNGAM causal discovery approach"
        LiNGAM, the linear non-Gaussian acyclic model, is known as one of the
        [structural-identifiable SCMs](https://xuanzhichen.github.io/cadimulc/generation/).

    ***MLC-LiNGAM* was proposed to alleviate the following issues**:

    - how to detect the latent confounders;
    - how to uncover the causal relations among observed and latent variables.

    <!--
    References:
    Chen, Wei, Ruichu Cai, Kun Zhang, and Zhifeng Hao.
    "Causal discovery in linear non-gaussian acyclic model with multiple latent confounders. "
    *IEEE Transactions on Neural Networks and Learning Systems.* 2021.

    https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Causal+discovery+in+linear+non-gaussian+acyclic+model+with+multiple+latent+confounders&btnG=#d=gs_cit&t=1711554753714&u=%2Fscholar%3Fq%3Dinfo%3AzEuwtDsRA24J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den

    Shimizu, Shohei, Patrik O. Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan.
    A linear non-Gaussian acyclic model for causal discovery.
    *Journal of Machine Learning Research.* 2006.

    https://scholar.google.com/citations?view_op=view_citation&hl=en&user=OpLI4xcAAAAJ&citation_for_view=OpLI4xcAAAAJ:7PzlFSSx8tAC
    -->
    """

    def __init__(
            self,
            pc_alpha: float = 0.05,
            _latent_confounder_detection: list[list] = []
    ):
        """
        Parameters:
            pc_alpha:
                Significance level of independence tests (p_value), which is required by
                the constraint-based methodology incorporated in the initial stage of
                the hybrid causal discovery framework.

        <!--
        Attributes:
            _latent_confounder_detection:
                The list elements given by `_latent_confounder_detection` are
                undirected maximal cliques after stage-III learning, suggesting that the
                variables within the undirected maximal clique share an unknown common cause.
        -->
        """

        HybridFrameworkBase.__init__(self, pc_alpha=pc_alpha)
        self._latent_confounder_detection = _latent_confounder_detection

    # ### AUXILIARY COMPONENT(S) ##########################################################
    # Function: _stage_1_learning
    # Function: _stage_2_learning
    # Function: _stage_3_learning

    def fit(self, dataset: ndarray) -> object:
        """
        Fitting data via the *MLC-LiNGAM* causal discovery algorithm:

        - **Stage-I**: Utilize the constraint-based method to learn a **causal skeleton**.
        - **Stage-II**: Identify the causal directions by conducting **regression** and **independence tests**
        on the adjacent pairs in the causal skeleton.
        - **Stage-III**: Detect the latent confounders with the help of the **maximal clique patterns**
        raised by the latent confounders,
        and uncover the causal structure with latent variables.

        Parameters:
            dataset:
                The observational dataset shown as a matrix or table,
                with a format of "sample (n) * dimension (d)."
                (input as Pandas dataframe is also acceptable)

        Returns:
            self:
                Update the ``adjacency_matrix`` represented as an estimated causal graph.
                The ``adjacency_matrix`` is a (d * d) numpy array with 0/1 elements
                characterizing the causal direction.
        """

        # stage-1: causal skeleton reconstruction(PC-stable algorithm)
        self._stage_1_learning(dataset)

        graph_pattern_manager = GraphPatternManager(init_graph=self._skeleton)

        # stage-2: partial causal orders identification
        self._stage_2_learning(graph_pattern_manager)

        graph_pattern_manager.store_last_managing_adjacency_matrix()

        # stage-3: latent confounders' detection
        self._stage_3_learning(graph_pattern_manager)

        return self

    # ### CORRESPONDING TEST ##############################################################
    # # Loc: test_hybrid_algorithms.py >> test_causal_skeleton_learning

    # ### AUXILIARY COMPONENT(S) ##########################################################
    # Class: HybridFrameworkBase

    def _stage_1_learning(self, dataset: ndarray) -> object:
        """
        **Stage-I**: Causal skeleton construction (based on the PC-stable algorithm).

        Stage-I begins with a complete undirected graph and performs **conditional
        independence tests** to delete the edges between independent variables pairs,
        reducing the computational cost of subsequent regressions and independence tests.

        Parameters:
            dataset:
                The observational dataset shown as a matrix or table,
                with a format of "sample (n) * dimension (d)."
                (input as Pandas dataframe is also acceptable)

        Returns:
            self:
                Update `_skeleton` as the estimated undirected graph corresponding to
                the causal graph, initialize `_adjacency_matrix` via a copy of `_skeleton`,
                and record `_stage1_time` as the stage-1 computational time.
        """

        self._causal_skeleton_learning(dataset)

        return self

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x3_procedure_stage_two_learning
    # Loc: test_hybrid_algorithms.py >> test_0x4_performance_stage_two_learning

    # ### CODING DATE #####################################################################
    # Module Stable: 2024-01-25
    # Modula Update: 2024-04-02

    # ### AUXILIARY COMPONENT(S) ##########################################################
    # Class:    GraphPatternManager
    # Function: _algorithm_2

    def _stage_2_learning(self, graph_pattern_manager) -> object:
        """
        **Stage-II**: Partial causal order identification.

        Based on the causal skeleton by stage-I,
        stage II in *MLC-LiNGAM* identifies causal directions among the adjacent variables
        that are implied by the skeleton.
        Causal orders relative to all variables can be partially determined by **regression
        and independence tests**, if variables that are relatively exogenous or endogenous
        do not be affected by latent confounders.

        Parameters:
            graph_pattern_manager:
                An auxiliary module embedded in the MLC-LiNGAM algorithm,
                managing adjacency matrices amidst the procedure between causal skeleton
                learning and causal direction orientation.

        Returns:
            self:
                Update `_adjacency_matrix` as the estimated (partial) directed acyclic
                graph (DAG) corresponding to the causal graph,
                and `record _stage2_time` as the `stage-2` computational time.
        """

        # Arguments for testing:
        #   _skeleton(attribute): ndarray
        #   _dataset(attribute): ndarray
        #   _dim(attribute): int

        start = time.perf_counter()

        # Reconstruction of the causal skeleton entails specific pairs of adjacent variables,
        # rather than all pairs of variables.
        causal_skeleton = self._skeleton

        # MLC-LiNGAM performs regression and independence tests efficiently
        # based on the adjacency set.
        adjacent_set = GraphPatternManager.find_adjacent_set(
            causal_skeleton=causal_skeleton
        )

        # Apply Algorithm-2 (given by the MLC-LiNGAM algorithm).
        self._algorithm_2(
            corresponding_adjacent_set=adjacent_set,
            corresponding_dataset=cp.copy(self._dataset),
            corresponding_variables=np.arange(self._dim),
            graph_pattern_manager=graph_pattern_manager
        )

        # Record computational time.
        end = time.perf_counter()
        self._stage2_time = end - start

        return self

    # ### CORRESPONDING TEST ##############################################################
    # Loc: test_hybrid_algorithms.py >> test_0x7_procedure_stage_three_learning

    # ### CODING DATE #####################################################################
    # Module Stable:: 2024-04-02

    # ### AUXILIARY COMPONENT(S) ##########################################################
    # Class:    GraphPatternManager
    # Function: _algorithm_2

    def _stage_3_learning(self, graph_pattern_manager) -> object:
        """
        **Stage-III**: Latent confounders' detection

        Stage-III will learn more causal orders if some variables are not affected
        by the latent confounders but are in the remaining subset.
        Meanwhile, the stage-III learning makes use of the causal skeleton information
        to reduce the testing space of remaining variables from all subsets to typical
        **maximal cliques**.

        Notice that the maximal cliques, including the undirected relations that cannot
        be determined, are possibly formed by latent confounders. This in turn provides
        insight to detect the latent confounders, and uncover the causal relations
        among observed and latent variables.

        Parameters:
            graph_pattern_manager:
                An auxiliary module embedded in the MLC-LiNGAM algorithm,
                featuring the algorithmic behavior of the maximal-cliques pattern recognition.

        Returns:
            self:
                Update ``_adjacency_matrix`` as the estimated (partial) directed acyclic
                graph (DAG) corresponding to the causal graph,
                ``_latent_confounder_detection`` as the undirected maximal cliques after
                 stage-III learning, and `record _stage3_time` as the `stage-3`
                 computational time.
        """

        # Arguments for testing:
        #   _skeleton (attribute)
        #   _adjacency_matrix (attribute)
        #   _parents_set (attribute)
        #   _dataset(attribute): ndarray

        start = time.perf_counter()

        # Recognize the maximal-clique pattern based on the causal skeleton.
        maximal_cliques_completely_undetermined = (
            GraphPatternManager.recognize_maximal_cliques_pattern(
                causal_skeleton=self._skeleton,
                adjacency_matrix=self._adjacency_matrix
            )
        )

        # Setup of regression, referring to MLC-LiNGAM default settings.
        regressor = LinearRegression()
        residuals_dataset = cp.copy(self._dataset)

        # Replace the variables in the clique with their corresponding residuals via
        # regressing out the effect of their confounded parents that are outside the clique.
        for maximal_clique in maximal_cliques_completely_undetermined:
            # Record: Each of the variable requires a single replacement if necessary.
            variables_replaced = {}
            for variable in maximal_clique:
                variables_replaced[variable] = set()

            # Get undetermined pairs within a clique.
            for i in maximal_clique:
                for j in maximal_clique[maximal_clique.index(i) + 1:]:
                    parents_i = graph_pattern_manager.managing_parents_set[i]
                    parents_j = graph_pattern_manager.managing_parents_set[j]

                    # Conduct residuals replacement if the variables share the same parents.
                    if (parents_i & parents_j) != set():
                        confounded_parents = parents_i & parents_j

                        for confounder in confounded_parents:
                            data_confounder = residuals_dataset[:, confounder]

                            if confounder not in variables_replaced[i]:
                                variables_replaced[i].add(confounder)

                                data_i = residuals_dataset[:, i]
                                residuals_i = get_residuals_scm(
                                    explanatory_data=data_confounder,
                                    explained_data=data_i,
                                    regressor=regressor
                                )
                                residuals_dataset[:, i] = residuals_i.squeeze()

                            if confounder not in variables_replaced[j]:
                                variables_replaced[j].add(confounder)

                                data_j = residuals_dataset[:, j]
                                residuals_j = get_residuals_scm(
                                    explanatory_data=data_confounder,
                                    explained_data=data_j,
                                    regressor=regressor
                                )
                                residuals_dataset[:, j] = residuals_j.squeeze()

        # Apply Algorithm-2 on the maximal cliques.
        for maximal_clique in maximal_cliques_completely_undetermined:
            # Get adjacent set with respect to the variables within maximal cliques.
            adjacent_set_clique = {}
            for variable in maximal_clique:
                adjacent_set_clique[variable] = set(maximal_clique) - {variable}

            # Apply Algorithm-2 (given by the MLC-LiNGAM algorithm).
            self._algorithm_2(
                corresponding_adjacent_set=adjacent_set_clique,
                corresponding_dataset=residuals_dataset,
                corresponding_variables=np.array(maximal_clique),
                graph_pattern_manager=graph_pattern_manager,
                _specify_adjacency=True,
                _adjacent_set=adjacent_set_clique
            )

        # Update latent confounder detection
        graph_pattern_manager.store_last_managing_adjacency_matrix()
        self._latent_confounder_detection = (
            graph_pattern_manager.get_undetermined_cliques(
                maximal_cliques=maximal_cliques_completely_undetermined
            )
        )

        # Record computational time.
        end = time.perf_counter()
        self._stage3_time = end - start

        return self

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # Function: MLCLiNGAM -> _stage_2_learning
    # Function: MLCLiNGAM -> _stage_3_learning

    def _algorithm_2(
            self,
            corresponding_adjacent_set: dict,
            corresponding_dataset: ndarray,
            corresponding_variables: ndarray,
            graph_pattern_manager,
            _specify_adjacency=False,
            _adjacent_set=None
    ) -> object:
        """
        Implementation of the module "Algorithm-2" in the MLC-LiNGAM algorithm.
        """

        # ================================ INITIALIZATION =================================

        # Initialize the dataset and the relative variable set.
        adjacent_set = copy_and_rename(corresponding_adjacent_set)
        _X = copy_and_rename(corresponding_dataset)
        _x = copy_and_rename(corresponding_variables)

        # Order list for the sequential search of exogenous variables and leaf variables
        k_head = []
        k_tail = []

        # Setup of regression and independence tests, referring to MLC-LiNGAM default settings.
        regressor = LinearRegression()
        ind_test_method = 'kernel_ci'

        # ========================= IDENTIFY EXOGENOUS VARIABLES ==========================

        # Development notes: In accord with the pseudocode in MLC-LiNGAM:
        #   x_i or i: refer to the exogenous variable

        # Perform up-down search targeting at exogenous variables.
        repeat = True
        while repeat:

            # The last remaining variable is endogenous respectively.
            if len(k_head) == (len(_x) - 1):
                break

            # Development notes: An addition loop is combined to search the most
            # exogenous variable (to strengthen the MLC-LiNGAM algorithm).

            # Search for the most exogenous variable based on relative p-values.
            p_values_x_all = {}
            for x_i in (set(_x) - set(k_head)):

                # Get adjacent set of the candidate variable.
                adjacent_set_i = adjacent_set[x_i]

                # Check if the variable x_i is in form of a trivial sub-graph.
                if len(adjacent_set_i) == 0:
                    k_head.append(x_i)
                    continue

                # Exclude the ones in K-head-list in which
                # regressing and supplanting other variables with residuals have been performed.
                adjacent_set_i = adjacent_set_i - set(k_head)

                # Check if the variables are respectively the most exogenous.
                if len(adjacent_set_i) == 0:
                    k_head.append(x_i)
                    continue

                # Development notes: Check out code of Xuanzhi's old implementation version.
                # (perhaps only required in stage-3)
                # _ = i_adj.copy()
                # for j in _:
                #     if self._check_identity(i, j):
                #         i_adj.remove(j)

                # Separately regress on adjacent variables of the candidate variable x_i
                # and check if all residuals are independent of it.
                p_values_x_i = []
                for x_j in adjacent_set_i:
                    residuals = get_residuals_scm(
                        explanatory_data=_X[:, x_i],
                        explained_data=_X[:, x_j],
                        regressor=regressor
                    )

                    p_value = conduct_ind_test(
                        explanatory_data=_X[:, x_i],
                        residuals=residuals,
                        ind_test_method=ind_test_method
                    )

                    p_values_x_i.append(p_value)

                # Check if the candidate variable satisfying exogeneity.
                if np.min(p_values_x_i) >= self.pc_alpha:
                    p_values_x_all[x_i] = np.min(p_values_x_i)

            # End if none of the candidate variable satisfying exogeneity.
            if len(p_values_x_all.values()) == 0:
                repeat = False

            else:
                # Mark continuous searching.
                repeat = True

                # Determine the most exogenous variable.
                p_value_max = cp.copy(self.pc_alpha)
                x_exogenous = None
                for x_i, p_value in p_values_x_all.items():
                    if p_value > p_value_max:
                        p_value_max = p_value
                        x_exogenous = x_i

                # Append the exogenous variable sequentially to k-head-list.
                k_head.append(x_exogenous)

                # Regress and supplant other variables with the residuals
                # regressed by the exogenous variable.
                for x_j in (adjacent_set[x_exogenous] - set(k_head)):
                    supplanting_residuals = get_residuals_scm(
                        explanatory_data=_X[:, x_exogenous],
                        explained_data=_X[:, x_j],
                        regressor=regressor
                    )

                    # Development notes: Residuals for supplanting are additionally computed
                    # to save memory.
                    _X[:, x_j] = supplanting_residuals.ravel()

        # ============================ IDENTIFY LEAF VARIABLES ============================

        # Development notes: In accord with the pseudocode in MLC-LiNGAM:
        #   x_j or j: refer to leaf variable

        # Perform bottom-up search targeting at leaf variables
        # if the causal order presents more than two variables staying undetermined.
        if len(k_head) < (len(_x) - 2):

            repeat = True
            while repeat:

                # The last remaining variable is endogenous respectively.
                if len(k_head) + len(k_tail) == (len(_x) - 1):
                    break

                # Development notes: An addition loop is combined to search the most
                # endogenous (leaf) variable (to strengthen the MLC-LiNGAM algorithm).

                # Search for the most endogenous variable based on relative p-values.
                p_values_x_all = {}
                for x_j in (set(_x) - (set(k_head) | set(k_tail))):

                    # Get adjacent set of the candidate variable.
                    adjacent_set_j = adjacent_set[x_j]

                    # Development notes: Check out code of Xuanzhi's old implementation version.
                    # if len(i_adj) == 0:
                    #     continue

                    # Exclude ones in K-head-list in which
                    # regressing and supplanting residuals have been performed.
                    adjacent_set_j = adjacent_set_j - set(k_head)

                    # Ignore the ones in K-tail-list that are explained variables relative to x_j.
                    adjacent_set_j = adjacent_set_j - set(k_tail)

                    # Check if the variables are respectively the most exogenous.
                    if len(adjacent_set_j) == 0:
                        # k_tail.insert(0, x_j)
                        k_head.append(x_j)
                        continue

                    #  Regress the candidate variable x_j on all its adjacent variables
                    #  and check if its residuals are all independent of them.
                    residuals = get_residuals_scm(
                        explanatory_data=_X[:, list(adjacent_set_j)],
                        explained_data=_X[:, x_j],
                        regressor=regressor
                    )
                    p_value = conduct_ind_test(
                        explanatory_data=_X[:, list(adjacent_set_j)],
                        residuals=residuals,
                        ind_test_method=ind_test_method
                    )
                    p_values_x_j = copy_and_rename(p_value)

                    # Check if the candidate variable is likely the leaf variable.
                    if p_values_x_j >= self.pc_alpha:
                        p_values_x_all[x_j] = p_values_x_j

                # End if none of the candidate variable is likely the leaf variable.
                if len(p_values_x_all.values()) == 0:
                    repeat = False

                else:
                    # Mark continuous searching.
                    repeat = True

                    # Determine the most endogenous variable.
                    p_value_max = cp.copy(self.pc_alpha)
                    x_leaf = None
                    for x_j, p_value in p_values_x_all.items():
                        if p_value > p_value_max:
                            p_value_max = p_value
                            x_leaf = x_j

                    # Insert the leaf variable at the top of k-tail-list.
                    k_tail.insert(0, x_leaf)

        # ======================== IDENTIFY PARTIAL CAUSAL ORDER ==========================

        # Update causal skeleton to partial causal structure according to
        # K-Head and K-Tail list.
        graph_pattern_manager.identify_partial_causal_order(
            k_head=k_head,
            k_tail=k_tail
        )

        self._adjacency_matrix = graph_pattern_manager.managing_adjacency_matrix
        self._parents_set = graph_pattern_manager.managing_parents_set

        return self

    @property
    def latent_confounder_detection_(self):
        return self._latent_confounder_detection


if __name__ == "__main__":
    np.random.seed(42)
    pass
