"""Non-linear causal discovery with multiple latent confounders."""

# Author:  Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License

# Demonstration:
# ../paper_2023/tutorial_causal_discovery_with_multiple_latent_confounders

# Testing: ../tests/test_hybrid_algorithms.py


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * The fact that procedures of identifying leaf variables comes necessary
#   hints exogenous identification may get "stuck" by hidden confounding.
#
# * Implementation of the time-series-Nonlinear-MLC version are migrated
#   and commented in the file `../paper_2023/nonlinearmlc.py`, acting as a
#   static snapshot for reproduction.


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Reconstruct initial frameworks of Nonlinear-MLC and MLC-LiNGAM in one
#   file, adding a base-class and static-methods.             21st.Jan, 2024
#
#
# * Start externally on adding a plotting function and internally on data
#   stream interactions. Went further in coding "stage learning", found that
#   encapsulating modules is urgent.                          01st.Nov, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: TBD.
#
# Done:
# TODO: Reconstruction of Nonlinear-MLC starts with the stage-1 learning.
# TODO: Encapsulate modules: regression and ind-test.


# hybrid causal discovery framework
from .hybrid_framework import HybridFrameworkBase

# auxiliary modules in causality instruments
from cadimulc.utils.causality_instruments import (
    get_residuals_scm,
    conduct_ind_test,
)

# linear and non-linear regression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from pygam import LinearGAM

# independent test
from causallearn.search.FCMBased.lingam.hsic2 import hsic_gam
from causallearn.utils.KCI.KCI import KCI_UInd

# basic
from cadimulc.utils.extensive_modules import (
    check_1dim_array,
    copy_and_rename
)

import numpy as np
import networkx as nx
import copy as cp
import time
import warnings
warnings.filterwarnings("ignore")


class GraphPatternManager(object):
    """
    Write down some descriptions here.

    Parameters
    ----------
    init_graph : ndarray
        Numpy array of causal skeleton.

    Attributes
    ----------
    managing_skeleton : ndarray
        Write down some descriptions here.
    managing_adjacency_matrix : ndarray
        Write down some descriptions here.
    managing_adjacency_matrix_last : ndarray
        Write down some descriptions here.
    managing_parents_set : dict
        Write down some descriptions here.
    """

    def __init__(
            self,
            init_graph
    ):
        self.managing_skeleton = cp.copy(init_graph)
        self.managing_adjacency_matrix = cp.copy(init_graph)
        self.managing_adjacency_matrix_last = None
        self.managing_parents_set = {}

        for variable_index in range(init_graph.shape[0]):
            self.managing_parents_set[variable_index] = set()

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: MLCLiNGAM -> _stage_2_learning

    def identify_partial_causal_order(self, k_head, k_tail):
        """
        Write down some descriptions here.

        Arguments
        ---------
        _managing_skeleton (attribute) : ndarray
            Write down some descriptions here.
        _managing_adjacency_matrix (attribute) : ndarray
            Write down some descriptions here.

        Parameters
        ----------
        k_head : list
            Write down some descriptions here.

        k_tail : list
            Write down some descriptions here.

        Returns
        -------
        self : object
            Update ``_managing_adjacency_matrix``, ``_managing_parents_set``.

        _managing_adjacency_matrix (update) : ndarray
        _managing_parents_set (update) : set
        """

        # Orientation by K-Head and K-Tail list should on the adjacent baseline.
        adjacent_set = GraphPatternManager.find_adjacent_set(
            causal_skeleton=self.managing_skeleton
        )

        # Notes for developer: In accord with the pseudocode in MLC-LiNGAM:
        # x_i or i: refer to exogenous variable
        # x_j or j: refer to leaf variable

        # Orient the edge from exogenous variables to their adjacent variables.
        k_head_temp = []
        for i in k_head:
            for j in (adjacent_set[i] - set(k_head_temp)):
                self.managing_adjacency_matrix[j][i] = 1
                self.managing_adjacency_matrix[i][j] = 0

                # Update the relative parent set.
                self.managing_parents_set[j].add(i)

                k_head_temp.append(i)

        # Orient the edge from leaf-variable-adjacent counterparts to leaf variables.
        k_tail_temp = []
        for j in reversed(k_tail):
            # Notes for developer: Need careful check for TBD-code fragments
            # from my old implementation versions.

            # # Avoid traverse on an int type.
            # if type(i_adj) is int or type(i_adj) is np.int32:

            for i in (adjacent_set[j] - (set(k_head) | set(k_tail_temp))):
                self.managing_adjacency_matrix[i][j] = 0
                self.managing_adjacency_matrix[j][i] = 1

                # Update the relative parent set.
                self.managing_parents_set[j].add(i)

                k_tail_temp.append(j)

        return self

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: NonlinearMLC -> fit

    def identify_directed_causal_pair(self, determined_pairs):
        """
        Write down some descriptions here.

        Arguments
        ---------
        testing_text (parameter) : testing_text
        testing_text (attribute) : testing_text

        Parameters
        ----------
        determined_pairs : list
            Identified pairs after inference over a maximal clique.

        Returns
        -------
        self : object
            Update ``_managing_adjacency_matrix``, ``_managing_parents_set``.

        _managing_adjacency_matrix (update) : ndarray
        _managing_parents_set (update) : set
        """

        # Start the code line

        # # Case-1: Testing outcome of both causal directions
        # #         i -> j and j -> i reaches the significant level.
        # if (p_value_ij > self.pc_alpha) and (p_value_ji > self.pc_alpha):
        #     if p_value_ij > p_value_ji:
        #         self._adjacency_matrix = (
        #             GraphPatternManager._orient_causal_directions(
        #                 adjacency=False,
        #                 parent=index_i,
        #                 child=index_j
        #             )
        #         )
        #
        #         self._orient_adjacency_matrix(i_adj=[j], i=i)
        #         self._parents_set[i].add(j)
        #     else:
        #         self._orient_adjacency_matrix(i_adj=[i], i=j)
        #         self._parents_set[j].add(i)
        #
        # # Case-2: Testing outcome of both causal directions
        # #         i -> j and j -> i reaches the significant level.
        # elif (pval_ij > self._alpha) and (pval_ji < self._alpha):
        #     self._orient_adjacency_matrix(i_adj=[j], i=i)
        #     self._parents_set[i].add(j)
        #
        # # Case-1: Testing outcome of both causal directions
        # #         i -> j and j -> i reaches the significant level.
        # elif (pval_ij < self._alpha) and (pval_ji > self._alpha):
        #     self._orient_adjacency_matrix(i_adj=[i], i=j)
        #     self._parents_set[j].add(i)
        #
        # # Case-1: Testing outcome of both causal directions
        # #         i -> j and j -> i reaches the significant level.
        # else:
        #     continue

        return self

    def store_last_managing_adjacency_matrix(self):
        """ Store the last managing adjacency matrix, usually for checking
            the newly determined edge in the next round search (over cliques).
        """
        self.managing_adjacency_matrix_last = cp.copy(
            self.managing_adjacency_matrix
        )
        return self

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_hybrid_algorithms.py >> test_0x2_graph_pattern_manager

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: NonlinearMLC -> fit
    # Function: MLCLiNGAM -> _stage_3_learning

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: None

    def get_undetermined_cliques(self, maximal_cliques):
        """
        Get undetermined cliques with respect to (original) maximal cliques,
        which implies there is at least one edge remaining undetermined.

        Parameters
        ----------
        maximal_cliques : list
            The maximal cliques along with the element (maximal clique)
            in form as well of list.

        Returns
        -------
        undetermined_cliques : list
            The list of ``undetermined_cliques``.
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

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_hybrid_algorithms.py >> test_0x2_graph_pattern_manager

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: NonlinearMLC -> fit
    # Function: MLCLiNGAM -> _stage_3_learning

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: None

    def check_newly_determined(self, last_undetermined_cliques):
        """
        Check whether there exists a newly determined edge over the searching
        range of the last undetermined cliques.

        Parameters
        ----------
        last_undetermined_cliques : list
            The last undetermined cliques along with the element
            (maximal clique) in form as well of list.

        Arguments
        ---------
        managing_adjacency_matrix (attribute): ndarray
        managing_adjacency_matrix_last (attribute): ndarray

        Returns
        -------
        newly_determined : bool
            The bool value of ``newly_determined``.
        """

        newly_determined = False

        for undetermined_clique in last_undetermined_cliques:
            for i in undetermined_clique:
                for j in undetermined_clique[undetermined_clique.index(i):]:

                    # Call the determined information of adjacency matrix
                    # relative to the last undetermined cliques.
                    last_info_ij = self.managing_adjacency_matrix_last[i][j]
                    last_info_ji = self.managing_adjacency_matrix_last[j][i]

                    # Mark as newly determined if any current edge within
                    # the range of the undetermined cliques is determined.
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

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_hybrid_algorithms.py >> test_0x2_graph_pattern_manager

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: NonlinearMLC -> fit
    # Function: MLCLiNGAM -> _stage_3_learning

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: None

    @staticmethod
    def recognize_maximal_cliques_pattern(causal_skeleton):
        """
        Recognize the maximal cliques pattern over a causal skeleton.

        Parameters
        ----------
        causal_skeleton : ndarray
            Numpy array of causal skeleton.

        Returns
        -------
        maximal_cliques : list
            The ``maximal cliques`` list along with the element (maximal clique)
            in form as well of list.
        """

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

    # ### CORRESPONDING TEST ###############################################
    # Loc: None

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: MLCLiNGAM -> _stage_2_learning

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: None

    @staticmethod
    def find_adjacent_set(causal_skeleton):
        """
        Write down some descriptions here.

        Parameters
        ----------
        causal_skeleton : ndarray
            Write down some descriptions here.

        Returns
        -------
        adjacent_set : dict
            Write down some descriptions here.
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


class NonlinearMLC(HybridFrameworkBase):
    """
    Nonlinear-MLC is a causal discovery framework.

    References
    ----------
    **Chen, XZ.***, Chen, W.*, Cai, RC.
    Non-linear Causal Discovery for Additive Noise Model with
    Multiple Latent Confounders. In Xuanzhi CHEN's website. 2023.

    http://www.auai.org/~w-auai/uai2020/proceedings/579_main_paper.pdf

    Parameters
    ----------
    regressor : object
        recommend: ``from pygam import LinearGAM``;
        recommend: ``from sklearn.neural_network import MLPRegressor``

    ind_test : string
        KCI, HSIC-GAM

    pc_alpha : float (default: 0.5)
        Write down some descriptions here.


    Attributes
    ----------
    _dataset : dataframe
        Write down some descriptions here.

    _dim : int
        Write down some descriptions here.

    _skeleton : ndarray
        Write down some descriptions here.

    _adjacency_matrix : ndarray
        Write down some descriptions here.

    _parents_set : dictionary
        Write down some descriptions here.

    _stage1_time : float
        Write down some descriptions here.

    _stage2_time : float
        Write down some descriptions here.

    _stage3_time : float
        Write down some descriptions here.


    Examples
    --------
    # >>> #################### USAGE-1 ########################
    # >>> # Recommended setting for non-sequential.
    # >>> nonlin_mlc = NonlinearMLC(
    #                                handle_time_series=False,
    #                                Reg="mlp",
    #                                IndTest="kci"
    #                                )
    # >>> nonlin_mlc.fit(X)
    #
    # >>> #################### USAGE-2 ########################
    # >>> # Recommended setting for time series.
    # >>> nonlin_mlc = NonlinearMLC(
    #                                handle_time_series=True,
    #                                max_lag=1
    #                                Reg="gam",
    #                                IndTest="hsic"
    #                        )
    # >>> nonlin_mlc.fit(X)

    Notes
    -----
    * Write down some descriptions here.
    * Write down some descriptions here.
    * Write down some descriptions here.
    """

    def __init__(
            self,
            regressor=LinearGAM(),
            ind_test='kernel_hsic',
            pc_alpha=0.05,

            # useless code fragment (TBD)
            # skeleton_prior=None,
            # lv_info=None,
    ):

        HybridFrameworkBase.__init__(self, pc_alpha=pc_alpha)

        self.regressor = regressor
        self.ind_test = ind_test

        # useless code fragment (TBD)
        # self._dag_gt = dag_gt
        # self._lv_info = lv_info

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_hybrid_algorithms.py >> test_0x5_procedure_fitting
    # Loc: test_hybrid_algorithms.py >> test_0x6_performance_fitting

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: fit

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: _clique_based_causal_inference (self)
    # Function: recognize_maximal_cliques_pattern
    # Function: get_undetermined_cliques
    # Function: check_newly_determined
    # Class:    GraphPatternManager

    # ### CODING DATE ######################################################
    # Module Construction: 2024-__-__ | xx:xx (pass)
    # Module Update      : 2024-__-__ | xx:xx (pass)

    def fit(self, dataset):
        """
        Write down some descriptions here.

        Parameters
        ----------
        dataset : ndarray or dataframe (sample * dimension)
            Write down some descriptions here.

        Returns
        -------
        self : object
            Write down some descriptions here.
        _adjacency_matrix (update) : ndarray
        _running_time (update) : float
        """

        start = time.perf_counter()

        # Reconstruct the causal skeleton using the PC-stable algorithm.
        self._skeleton = self._causal_skeleton_learning(dataset)

        # Recognize the maximal-clique pattern.
        maximal_cliques = GraphPatternManager.recognize_maximal_cliques_pattern(
            causal_skeleton=self._skeleton
        )

        # Initialize a graph pattern manager for subsequent learning.
        graph_pattern_manager = GraphPatternManager(
            init_graph=self._skeleton
        )

        # Perform the nonlinear-mlc causal discovery (based on the L-ANM theory)
        continue_search = True
        while continue_search:

            # Obtain cliques that remain at least one edge undetermined.
            undetermined_maximal_cliques = (
                graph_pattern_manager.get_undetermined_cliques(maximal_cliques)
            )

            # End if all edges over the causal skeleton have been determined.
            if len(undetermined_maximal_cliques) == 0:
                break

            # Conduct pairwise non-linear causal inference over each clique unit.
            determined_pairs = self._clique_based_causal_inference(
                undetermined_maximal_cliques=undetermined_maximal_cliques
            )

            # Orient the determined causal directions.
            graph_pattern_manager.identify_directed_causal_pair(
                determined_pairs=determined_pairs
            )

            # Update the causal adjacency matrix.
            self._adjacency_matrix = (
                graph_pattern_manager.managing_adjacency_matrix
            )

            # Temporally store the adjacency matrix after managing a search round.
            graph_pattern_manager.store_last_managing_adjacency_matrix()

            # Check if new causal relations have been determined
            # after the last round searching
            newly_determined = (
                graph_pattern_manager.check_newly_determined(
                    undetermined_maximal_cliques
                )
            )

            # End if none of new causal relation advancing the further search.
            if not newly_determined:
                continue_search = False

        end = time.perf_counter()

        self._running_time += (end - start)

        return self

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_hybrid_algorithms.py >> test_0x5_procedure_fitting
    # Loc: test_hybrid_algorithms.py >> test_0x6_performance_fitting

    # ### SUBORDINATE COMPONENT(S) #########################################
    # Function: fit

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: None

    def _clique_based_causal_inference(self, undetermined_maximal_cliques):
        """
        Write down some descriptions here.

        Arguments
        ---------
        pc_alpha (parameter)  : float
        regressor (parameter) : object
        ind_test (parameter)  : string
        _adjacency_matrix (attribute) : ndarray
        _parents_set (attribute) : dict
        _dataset (attribute) : ndarray

        Parameters
        ----------
        undetermined_maximal_cliques : list
            The list of undetermined cliques in which each of the clique
            is in form of list as well (element: int of vertex).

        Returns
        -------
        determined_pairs : list
            The list of determined pairs over the inputting undetermined cliques
            after the maximal-clique-based causal inference.
        """

        # Perform non-linear causal inference based on each maximal clique unit.
        for undetermined_maximal_clique in undetermined_maximal_cliques:
            # Initialize the lists of undetermined and determined causal relations.
            undetermined_pairs = []
            # (in form of "[cause -> effect] == [(cause, effect)]")
            determined_pairs = []

            # Get undetermined pairs within the clique.
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

                    # initialization of explanatory-and-explained variables
                    explanatory_vars = set()
                    explained_var = set()

                    # basic explanatory-and-explained variables:
                    # cause and effect
                    explanatory_vars.add(cause)
                    # namely the effect variable
                    explained_var.add(effect)

                    # Add explanatory variables to strengthen empirical regression.

                    # determined parent-relations amidst the algorithm memory
                    explanatory_vars.add(self._parents_set[effect])
                    # undetermined connections within the maximal clique
                    explanatory_vars.add(
                        set(undetermined_maximal_clique) - (cause | effect)
                    )

                    # Regress the effect variable on empirical explanatory variables.
                    # (in an attempt to cancel unobserved confounding)
                    explanatory_data = cp.copy(
                        self._dataset[:, list(explanatory_vars)]
                    )
                    # namely the data with respect to the effect variable
                    explained_data = cp.copy(
                        self._dataset[:, explained_var]
                    )
                    residuals = get_residuals_scm(
                        explanatory_data=explanatory_data,
                        explained_data=explained_data,
                        regressor=self.regressor
                    )

                    # Remove effects of parent-relations from the cause variable.
                    # (in an attempt to cancel unobserved confounding)
                    cause_parents = list(self._parents_set[cause])
                    if len(cause_parents) > 0:
                        cause_data = get_residuals_scm(
                            explanatory_data=self._dataset[:, cause_parents],
                            explained_data=self._dataset[:, cause],
                            regressor=self.regressor
                        )
                    else:
                        cause_data = cp.copy(self._dataset[:, cause])

                    # Conduct the independence test
                    # between the cause variable variables and regressing residuals
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

    # ######################################################################
    # ### OLD ##############################################################
    # ######################################################################
    # def _stage_2_learning(self):
    #     start = time.perf_counter()
    #
    #     Adj_set = get_Adj_set(self._skeleton)  # Quarry by Adj_set[variable] = {adjacent variable set}
    #     d = X.shape[1]
    #     T = X.shape[0]
    #     X_ = copy.copy(X)
    #     U = np.arange(d)
    #
    #     unorient_pairs = []
    #     for i in range(d):
    #         for j in range(d):
    #             if i != j:
    #                 if (self._adjacency_matrix[i][j] == 1) and (self._adjacency_matrix[i][j] == 1):
    #                     unorient_pairs.append((i, j))
    #
    #     if len(unorient_pairs) >= 2:
    #         for pair in unorient_pairs:
    #             pair = list(pair)
    #             i = pair[0]
    #             j = pair[1]
    #
    #             # Test j -> i pairwisely:
    #             if self._Reg == "gam":
    #                 reg = LinearGAM()
    #             elif self._Reg == "xgboost":
    #                 reg = XGBRegressor()
    #             elif self._Reg == "mlp":
    #                 reg = MLPRegressor()
    #             else:
    #                 raise ValueError("Module haven't been built.")
    #             residual = residual_by_nonlinreg(X=X_[:, j], y=X_[:, i], Reg=reg)
    #             if self._IndTest == "hsic":
    #                 pval_ij = hsic2.hsic_gam(residual, check_vector(X_[:, j]), mode="pvalue")
    #             elif self._IndTest == "kci":
    #                 kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
    #                 pval_ij, _ = kci.compute_pvalue(check_vector(X_[:, j]), residual)
    #             else:
    #                 raise ValueError("Module haven't been built.")
    #
    #             # Test i -> j pairwisely:
    #             if self._Reg == "gam":
    #                 reg = LinearGAM()
    #             elif self._Reg == "xgboost":
    #                 reg = XGBRegressor()
    #             elif self._Reg == "mlp":
    #                 reg = MLPRegressor()
    #             else:
    #                 raise ValueError("Module haven't been built.")
    #             residual = residual_by_nonlinreg(X=X_[:, i], y=X_[:, j], Reg=reg)
    #             if self._IndTest == "hsic":
    #                 pval_ji = hsic2.hsic_gam(residual, check_vector(X_[:, i]), mode="pvalue")
    #             elif self._IndTest == "kci":
    #                 kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
    #                 pval_ji, _ = kci.compute_pvalue(check_vector(X_[:, i]), residual)
    #             else:
    #                 raise ValueError("Module haven't been built.")
    #
    #             if (pval_ij > self._alpha) and (pval_ji > self._alpha):
    #                 if pval_ij > pval_ji:
    #                     self._orient_adjacency_matrix(i_adj=[j], i=i)
    #                     self._parents_set[i].add(j)
    #                 else:
    #                     self._orient_adjacency_matrix(i_adj=[i], i=j)
    #                     self._parents_set[j].add(i)
    #
    #             elif (pval_ij > self._alpha) and (pval_ji < self._alpha):
    #                 self._orient_adjacency_matrix(i_adj=[j], i=i)
    #                 self._parents_set[i].add(j)
    #
    #             elif (pval_ij < self._alpha) and (pval_ji > self._alpha):
    #                 self._orient_adjacency_matrix(i_adj=[i], i=j)
    #                 self._parents_set[j].add(i)
    #
    #             else:
    #                 continue
    #
    #     end = time.perf_counter()
    #
    #     self._stage2_time = end - start
    #
    #     return self

    # ######################################################################
    # ### OLD ##############################################################
    # ######################################################################

    # def _stage_3_learning(self, X):
    #
    #     start = time.perf_counter()
    #
    #     X_ = copy.copy(X)
    #     T = X.shape[0]
    #     new_edge_determine = True
    #
    #     while new_edge_determine:
    #         maximal_cliques = self._get_maximal_cliques()
    #         # Remove cliques which have been completely oriented in stage 2 and remind ones incompletely.
    #         maximal_cliques_incomplete = self._check_incomplete(maximal_cliques)
    #
    #         if len(maximal_cliques_incomplete) == 0:
    #             new_edge_determine = False
    #         else:
    #             for maximal_clique in maximal_cliques_incomplete:
    #                 pairs = []
    #                 for vi in maximal_clique:
    #                     for vj in maximal_clique[maximal_clique.index(vi) + 1:]:
    #                         if not self._check_have_been_determined((vi, vj)):
    #                             pairs.append((vi, vj))
    #
    #                 for pair in pairs:
    #                     pvals_direction = []  # P values for i->j first index and j->i second index.
    #                     i = pair[0]
    #                     j = pair[1]
    #                     for direction in [[i, j], [j, i]]:
    #                         # 1. For pairs haven't oriented, suppose we now want to test x -> y and consider explantory variables as follow:
    #                         #  (i)   x is definitely the explantory of y. (explantory_inside)
    #                         #  (ii)  y's parents have already found in stage 2. (explantory_outside)
    #                         #  (iii) force consideration of "third" variables in the maximal clique. (bfd_variables_in_clique)
    #                         explain = {direction[1]}
    #                         explantory_inside = {direction[0]}
    #                         explantory_inside_and_outside = explantory_inside | (self._parents_set[list(explain)[0]])
    #                         bfd_variables_in_clique = set(maximal_clique) - (
    #                                 explantory_inside | explain)  # "bpd" refers to "back door or front door".
    #
    #                         # 2. Remove the nonlinear effect we have already known with the help of stage2 from explantory.
    #                         if not len(self._parents_set[list(explantory_inside)[0]]) > 0:
    #                             explantory_inside_data_remove_parents = X_[:, list(explantory_inside)]
    #                         else:
    #                             if self._Reg == "gam":
    #                                 reg = LinearGAM()
    #                             elif self._Reg == "xgboost":
    #                                 reg = XGBRegressor()
    #                             elif self._Reg == "mlp":
    #                                 reg = MLPRegressor()
    #                             else:
    #                                 raise ValueError("Module haven't been built.")
    #                             explantory_inside_data_remove_parents = residual_by_nonlinreg(
    #                                 X=X_[:, list(self._parents_set[list(explantory_inside)[0]])],
    #                                 y=X_[:, list(explantory_inside)],
    #                                 Reg=reg,
    #                             )
    #
    #                         # 3. Fit and perform independent test.
    #                         if self._Reg == "gam":
    #                             reg = LinearGAM()
    #                         elif self._Reg == "xgboost":
    #                             reg = XGBRegressor()
    #                         elif self._Reg == "mlp":
    #                             reg = MLPRegressor()
    #                         else:
    #                             raise ValueError("Module haven't been built.")
    #                         residual = residual_by_nonlinreg(
    #                             X=X_[:, list((explantory_inside_and_outside) | (bfd_variables_in_clique))],
    #                             y=X_[:, list(explain)],
    #                             Reg=reg
    #                         )
    #
    #                         if self._IndTest == "hsic":
    #                             pval = hsic2.hsic_gam(
    #                                 check_vector(explantory_inside_data_remove_parents), residual,
    #                                 mode="pvalue",
    #                             )
    #                         elif self._IndTest == "kci":
    #                             kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
    #                             pval, _ = kci.compute_pvalue(check_vector(explantory_inside_data_remove_parents),
    #                                                          residual)
    #                         else:
    #                             raise ValueError("Module haven't been built.")
    #
    #                         pvals_direction.append(pval)
    #
    #                     pval_ji = pvals_direction[0]
    #                     pval_ij = pvals_direction[1]
    #
    #                     # Check result for both two directions.
    #                     if (pval_ij > self._alpha) and (pval_ji > self._alpha):
    #                         if pval_ij > pval_ji:
    #                             self._orient_adjacency_matrix(i_adj=[j], i=i)
    #                             self._parents_set[i].add(j)
    #                         else:
    #                             self._orient_adjacency_matrix(i_adj=[i], i=j)
    #                             self._parents_set[j].add(i)
    #
    #                     elif (pval_ij > self._alpha) and (pval_ji < self._alpha):
    #                         self._orient_adjacency_matrix(i_adj=[j], i=i)
    #                         self._parents_set[i].add(j)
    #
    #                     elif (pval_ij < self._alpha) and (pval_ji > self._alpha):
    #                         self._orient_adjacency_matrix(i_adj=[i], i=j)
    #                         self._parents_set[j].add(i)
    #
    #                     else:
    #                         new_edge_determine = False
    #
    #     end = time.perf_counter()
    #     self._stage3_time = end - start
    #     return self


class MLCLiNGAM(HybridFrameworkBase):
    """
    MLC-LiNGAM is a causal discovery framework.

    References
    ----------
    Chen, Wei, et al.
    "Causal discovery in linear non-gaussian acyclic model
    with multiple latent confounders." IEEE Transactions on Neural Networks
    and Learning Systems 33.7 (2021): 2816-2827.

    Parameters
    ----------
    pc_alpha : float (default: 0.5)
        Write down some descriptions here.


    Attributes
    ----------
    _dataset : dataframe
        Write down some descriptions here.

    _dim : int
        Write down some descriptions here.

    _skeleton : ndarray
        Write down some descriptions here.

    _adjacency_matrix : ndarray
        Write down some descriptions here.

    _parents_set : dictionary
        Write down some descriptions here.

    _stage1_time : float
        Write down some descriptions here.

    _stage2_time : float
        Write down some descriptions here.

    _stage3_time : float
        Write down some descriptions here.


    Examples
    --------
    # >>> #################### USAGE-1 ########################
    # >>> # Recommended setting for non-sequential.
    # >>> nonlin_mlc = NonlinearMLC(
    #                                handle_time_series=False,
    #                                Reg="mlp",
    #                                IndTest="kci"
    #                                )
    # >>> nonlin_mlc.fit(X)
    #
    # >>> #################### USAGE-2 ########################
    # >>> # Recommended setting for time series.
    # >>> nonlin_mlc = NonlinearMLC(
    #                                handle_time_series=True,
    #                                max_lag=1
    #                                Reg="gam",
    #                                IndTest="hsic"
    #                        )
    # >>> nonlin_mlc.fit(X)

    Notes
    -----
    * Write down some descriptions here.
    * Write down some descriptions here.
    * Write down some descriptions here.
    """

    def __init__(
            self,
            pc_alpha=0.05,
    ):

        HybridFrameworkBase.__init__(self, pc_alpha=pc_alpha)

        # useless code fragment (TBD)
        # self._U_res = []
        # self._maximal_cliques = []

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_file.py >> test_function

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: _stage_1_learning
    # Function: _stage_2_learning
    # Function: _stage_3_learning
    def fit(self, dataset):
        """
        Write down some descriptions here.

        Parameters
        ----------
        dataset : ndarray or dataframe (sample * dimension)
            Write down some descriptions here.

        Returns
        -------
        _parents_set : dictionary (update)
            Write down some descriptions here.

        _dim : int
            Write down some descriptions here.

        self : object
            Write down some descriptions here.
        """

        # Stage1: causal skeleton reconstruction(PC-stable algorithm)
        self._stage_1_learning(dataset)

        graph_pattern_manager = GraphPatternManager(
            init_graph=cp.copy(self._skeleton)
        )

        # stage II: partial causal orders identification
        self._stage_2_learning()

        # Stage3: latent confounders' detection
        self._stage_3_learning(graph_pattern_manager)

        return self

    # ### CORRESPONDING TEST ###############################################
    # Loc: None

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: None
    def _stage_1_learning(self, dataset):
        """
        Stage I: Causal skeleton construction (based on PC-stable algorithm).

        Stage I begins with a complete undirected graph and performs conditional
        independence tests delete the edges between independent variables pairs,
        reducing the number of subsequent regressions and independence tests.

        Returns
        -------
        _skeleton (update) : ndarray
            Write down some descriptions here.
        _adjacency_matrix (update) : ndarray
            Write down some descriptions here.
        _stage1_time (update) : float
        Write down some descriptions here.

        self : object
        """

        self._causal_skeleton_learning(dataset)

        return self

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_hybrid_algorithms.py
    # >>   test_0x3_procedure_stage_two_learning
    # >>   test_0x4_performance_stage_two_learning

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: None
    # Class:    GraphPatternManager
    def _stage_2_learning(self):
        """
        Stage II: Partial causal order identification.

        Based on the causal skeleton obtained by stage I,
        stage II in MLC-LiNGAM  aims to determine the causal directions,
        identifying the observed variables that are not affected by
        latent confounders and the partial causal orders among
        adjacent observed variables in the causal skeleton.

        Arguments
        ---------
        pc_alpha (parameter) : float
        _skeleton (attribute) : ndarray
        _dataset (attribute) : dataframe
        _dim (attribute) : int

        Returns
        -------
        self : object
            Self (update ``_adjacency_matrix`` and ``_stage2_time``).
        _adjacency_matrix (update) : ndarray
        _stage2_time (update) : float
        """

        start = time.perf_counter()

        # ======================== INITIALIZATION ==========================

        # Reconstruction of the causal skeleton entails specific pairs of
        # adjacent variables, rather than all pairs of variables.
        causal_skeleton = self._skeleton

        # MLC-LiNGAM performs regression and independence tests efficiently
        # among adjacent set.
        adjacent_set = GraphPatternManager.find_adjacent_set(
            causal_skeleton=causal_skeleton
        )

        # dataset and relative variable set
        _X = cp.copy(self._dataset)
        _x = np.arange(self._dim)

        # order list for sequential exogenous variables and leaf variables
        k_head = []
        k_tail = []

        # Set up the combination of regression and the independence test,
        # referring to default settings in MLC-LiNGAM.
        regressor = LinearRegression()
        ind_test_method = 'kernel_hsic'

        # ================= IDENTIFY EXOGENOUS VARIABLES ===================

        # Notes for developer: In accord with the pseudocode in MLC-LiNGAM:
        # x_i or i: refer to exogenous variable

        # Perform up-down search targeting at exogenous variables.
        repeat = True
        while repeat:

            # Notes for developer: Need careful check for TBD-code fragments
            # from my old implementation versions.

            # if len(U) == 1:
            #     break

            # The last remaining variable is endogenous respectively.
            if len(k_head) == (len(_x) - 1):
                break

            # Notes for developer: An addition loop is combined to search the
            # most exogenous variable. (To strengthen MLC-LiNGAM)

            # Search for the most exogenous variable based on relative p-values.
            p_values_x_all = {}
            for x_i in (set(_x) - set(k_head)):

                # Get adjacent set of the candidate variable.
                adjacent_set_i = adjacent_set[x_i]

                # Check if variables are in form of trivial sub-graphs.
                if len(adjacent_set_i) == 0:
                    k_head.append(x_i)
                    continue

                # Exclude ones in K-head-list in which regressing and supplanting
                # other variables with residuals have been performed.
                adjacent_set_i = adjacent_set_i - set(k_head)

                # Check if variables are most exogenous respectively.
                if len(adjacent_set_i) == 0:
                    k_head.append(x_i)
                    continue

                # Notes for developer: Need careful check for TBD-code
                # fragments from my old implementation.(may (only) for stage-3)

                # _ = i_adj.copy()
                # for j in _:
                #     if self._check_identity(i, j):
                #         i_adj.remove(j)

                # Separately regress on adjacent variables of the candidate
                # variable and check if all residuals are independent of it.
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

                # Notes for developer: Need careful check for TBD-code
                # fragments from my old implementation versions.

                # U = U[U != i]

                # Regress and supplant other variables with residuals
                # regressed by the exogenous variable.
                for x_j in (adjacent_set[x_exogenous] - set(k_head)):
                    supplanting_residuals = get_residuals_scm(
                        explanatory_data=_X[:, x_exogenous],
                        explained_data=_X[:, x_j],
                        regressor=regressor
                    )

                    # Notes for developer: Residuals for supplanting are
                    # additionally computed to save memory.
                    _X[:, x_j] = supplanting_residuals.ravel()

        # Notes for developer: Need careful check for TBD-code
        # fragments from my old implementation versions.

        # self._U_res = U.copy()

        # ==================== IDENTIFY LEAF VARIABLES =====================

        # Notes for developer: In accord with the pseudocode in MLC-LiNGAM:
        # x_j or j: refer to leaf variable

        # Perform bottom-up search targeting at leaf variables
        # if causal orders of more than two variables stay undetermined.
        if len(k_head) < (self._dim - 2):

            repeat = True
            while repeat:

                # The last remaining variable is endogenous respectively.
                if len(k_head) + len(k_tail) == (len(_x) - 1):
                    break

                # Notes for developer: An addition loop is combined to search
                # the most leaf variable. (To strengthen MLC-LiNGAM)

                # Search for the most leaf variable based on relative p-values.
                p_values_x_all = {}
                for x_j in (set(_x) - (set(k_head) | set(k_tail))):

                    # Get adjacent set of the candidate variable.
                    adjacent_set_j = adjacent_set[x_j]

                    # Notes for developer: Need careful check for TBD-code
                    # fragments from my old implementation.

                    # if len(i_adj) == 0:
                    #     continue

                    # Exclude ones in K-head-list in which regressing and
                    # supplanting residuals have been performed.
                    adjacent_set_j = adjacent_set_j - set(k_head)

                    # Ignore ones in K-tail-list that are explained variables
                    # respective to x_j.
                    adjacent_set_j = adjacent_set_j - set(k_tail)

                    # Check if variables are most exogenous respectively.
                    if len(adjacent_set_j) == 0:
                        # k_tail.insert(0, x_j)
                        k_head.append(x_j)
                        continue

                    #  Regress candidate variable on all its adjacent variables
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

                    # Determine the most leaf variable.
                    p_value_max = cp.copy(self.pc_alpha)
                    x_leaf = None
                    for x_j, p_value in p_values_x_all.items():
                        if p_value > p_value_max:
                            p_value_max = p_value
                            x_leaf = x_j

                    # Insert the leaf variable at the top of k-tail-list.
                    k_tail.insert(0, x_leaf)

        # ================= IDENTIFY PARTIAL CAUSAL ORDER ==================

        # Update causal skeleton to partial causal structure according to
        # K-Head and K-Tail list.
        graph_pattern_manager = GraphPatternManager(init_graph=self._skeleton)

        graph_pattern_manager.identify_partial_causal_order(
            k_head=k_head,
            k_tail=k_tail
        )

        self._adjacency_matrix = graph_pattern_manager.managing_adjacency_matrix
        self._parents_set = graph_pattern_manager.managing_parents_set

        # Record computational time.
        end = time.perf_counter()
        self._stage2_time = end - start

        return self

    # ######################################################################
    # ### NEW ##############################################################
    # ######################################################################

    # ### CORRESPONDING TEST ###############################################
    # Loc: test_file.py >> test_function

    # ### AUXILIARY COMPONENT(S) ###########################################
    # Function: _get_maximal_cliques
    # Class: component_2
    # Object: component_3
    def _stage_3_learning(self, graph_pattern_manager):
        """
        Write down some descriptions here.

        Arguments
        ---------
        testing_text (parameter) : testing_text
            Write down some descriptions here.
        testing_text (attribute) : testing_text
            Write down some descriptions here.

        Parameters
        ----------
        testing_text : testing_text
            Write down some descriptions here.

        Returns
        -------
        testing_text : testing_text (update)
            Write down some descriptions here.

        self : object
            Write down some descriptions here.
        """

        maximal_cliques = GraphPatternManager.recognize_maximal_cliques_pattern(
            causal_skeleton=self._skeleton
        )

        regressor = LinearRegression()
        ind_test_method = 'fisher_hsic'

        for maximal_clique in incomplete_maximal_cliques:
            X = cp.copy(self._dataset[:, maximal_clique])
            x = np.arange(self._variables.index(maximal_clique))
            k_head = []
            k_tail = []

        return

    # ######################################################################
    # ### OLD ##############################################################
    # ######################################################################

    # def _stage_3_learning(self, X):
    #     start = time.perf_counter()
    #
    #     if len(self._U_res) > 2:
    #         maximal_cliques = self._get_maximal_cliques()
    #
    #         if len(maximal_cliques) == 0:
    #             end = time.perf_counter()
    #             self._stage3_time = end - start
    #             return self
    #
    #         else:
    #             for maximal_clique in maximal_cliques:
    #                 # Remove effect of observed confounder outside clique.
    #                 X_ = copy.copy(X)
    #                 for vi in maximal_clique:
    #                     for vj in maximal_clique[vi:]:
    #                         if len(self._parents_set[vi] & self._parents_set[vj]) > 0:
    #                             confounder_set_out = self._parents_set[vi] & self._parents_set[vj]
    #                             for confounder in confounder_set_out:
    #                                 X_[:, vi] = (residual_by_linreg(X=X_[:, confounder], y=X_[:, vi])).ravel()
    #                                 X_[:, vj] = (residual_by_linreg(X=X_[:, confounder], y=X_[:, vj])).ravel()
    #
    #                 for size in range(len(maximal_clique), (2 - 1), -1):
    #                     new_edge_determine = True
    #
    #                     while new_edge_determine:
    #                         new_edge_determine = False
    #                         for maximal_clique_subset in subset(maximal_clique, size=size):
    #                             # "complete" refers to undirect graph which is the concept of clique.
    #                             complete = True
    #                             for vi in list(maximal_clique_subset):
    #                                 for vj in list(maximal_clique_subset)[vi:]:
    #                                     if self._check_identity(vi, vj):
    #                                         complete = False
    #
    #                             if not complete:
    #                                 continue
    #
    #                             U = np.array(list(maximal_clique_subset))
    #
    #                             # Identify exogenous variable in a clique.
    #                             repeat = True
    #                             while repeat:
    #                                 if len(U) == 1:
    #                                     break
    #                                 repeat = False
    #
    #                                 for i in U:
    #                                     is_exo = True
    #                                     i_adj = set(U) - {i}
    #
    #                                     _ = i_adj.copy()
    #                                     for j in _:
    #                                         if self._check_identity(i, j):
    #                                             i_adj.remove(j)
    #
    #                                     if len(i_adj) == 0:
    #                                         is_exo = False
    #                                         continue
    #
    #                                     for j in i_adj:
    #                                         # Remove effect of observed confounder inside clique.
    #                                         if len(self._parents_set[i] & set(maximal_clique)) > 0:
    #                                             confounder_set_in = self._parents_set[i] & set(maximal_clique)
    #                                             explantory = copy.copy(confounder_set_in)
    #                                             explantory.add(i)
    #                                             residual = residual_by_linreg(X=X_[:, list(explantory)], y=X_[:, j])
    #                                         else:
    #                                             residual = residual_by_linreg(X=X_[:, i], y=X_[:, j])
    #
    #                                         pval = fisher_hsic(check_vector(X_[:, i]), residual)
    #
    #                                         is_exo = True if pval > self._alpha else False
    #
    #                                     if is_exo:
    #                                         repeat = True
    #                                         new_edge_determine = True
    #                                         U = U[U != i]
    #                                         for j in i_adj:
    #                                             self._orient_adjacency_matrix(explanatory=i, explain=j)
    #                                             self._parents_set[j].add(i)
    #
    #                             if len(U) > 2:
    #                                 # Identify leaf variable in a clique.
    #                                 repeat = True
    #                                 while repeat:
    #                                     if len(U) == 1:
    #                                         break
    #                                     repeat = False
    #
    #                                     for i in U:
    #                                         i_adj = set(U) - {i}
    #
    #                                         _ = i_adj.copy()
    #                                         for j in _:
    #                                             if self._check_identity(i, j):
    #                                                 i_adj.remove(j)
    #
    #                                         if len(i_adj) == 0:
    #                                             continue
    #
    #                                         # Remove effect of observed confounder inside clique.
    #                                         if len(self._parents_set[i] & set(maximal_clique)) > 0:
    #                                             confounder_set_in = self._parents_set[i] & set(maximal_clique)
    #                                             explantory = i_adj | confounder_set_in
    #                                             residual = residual_by_linreg(X=X_[:, list(explantory)], y=X_[:, i])
    #                                         else:
    #                                             residual = residual_by_linreg(X=X_[:, list(i_adj)], y=X_[:, i])
    #
    #                                         pval = fisher_hsic(check_vector(X_[:, list(i_adj)]), residual)
    #
    #                                         if pval > self._alpha:
    #                                             repeat = True
    #                                             new_edge_determine = True
    #                                             U = U[U != i]
    #                                             self._orient_adjacency_matrix(explanatory=i_adj, explain=i)
    #                                             self._parents_set[i].union(i_adj)
    #
    #         for maximal_clique in maximal_cliques:
    #             complete = True
    #             undirect_exist = False
    #             for vi in maximal_clique:
    #                 for vj in maximal_clique[vi:]:
    #                     if vi != vj:
    #                         if self._check_identity(vi, vj):
    #                             complete = False
    #                         else:
    #                             undirect_exist = True
    #
    #             if (not complete and undirect_exist):
    #                 self._cliques.append(maximal_clique)
    #             elif complete:
    #                 self._cliques.append(maximal_clique)
    #             else:
    #                 continue
    #
    #         end = time.perf_counter()
    #         self._stage3_time = end - start
    #         return self


if __name__ == "__main__":
    np.random.seed(42)
    pass
