"""Non-linear causal discovery with multiple latent confounders."""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * Implementations of the time-series-Nonlinear-MLC version are migrated
#   and commented in the file `../paper_2023/nonlinearmlc.py`, acting as a
#   static snapshot for reproduction.


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
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


# auxiliary modules in causality instruments
from cadimulc.utils.causality_instruments import (
    get_skeleton_from_pc,
    get_residuals_scm,
    conduct_ind_test,
)

# auxiliary modules in data structure managements
from cadimulc.utils.extensive_modules import (
    check_1dim_array,
    # get_adjacent_vars,
    # get_subsets,
)

# basic
import numpy as np
import networkx as nx
import copy as cp
import time
import warnings
warnings.filterwarnings("ignore")


class NonlinearMLC(object):
    """
    ----------------------------------------------------------------------------

    Nonlinear-MLC is a causal discovery framework.

    References
    ----------
    **Chen, XZ.***, Chen, W.*, Cai, RC.
    Non-linear Causal Discovery for Additive Noise Model with
    Multiple Latent Confounders

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
            regressor,
            ind_test,
            pc_alpha=0.05,
            skeleton_prior=None,

            # useless code fragment (TBD)
            # lv_info=None,
    ):

        self._regressor = regressor
        self._ind_test = ind_test
        self.pc_alpha = pc_alpha

        # useless code fragment (TBD)
        # self._dag_gt = dag_gt
        # self._lv_info = lv_info

        self._dataset = None
        self._skeleton = None
        self._adjacency_matrix = None
        self._parents_set = {}
        self._stage1_time = 0
        self._stage2_time = 0
        self._stage3_time = 0

    # ### CORRESPONDING TEST #####################################################
    # Loc: test_file.py >> test_function

    # ### AUXILIARY COMPONENT(S) #################################################
    # Function: _stage_1_learning
    # Function: _stage_2_learning
    # Function: _stage_3_learning
    def fit(self, dataset):
        """
        # ------------------------------------------------------------------------

        Write down some descriptions here.

        Parameters
        ----------
        dataset : ndarray or dataframe
            Write down some descriptions here.

        Returns
        -------
        _parents_set : dictionary (update)
            Write down some descriptions here.

        self : object
            Write down some descriptions here.
        """

        # Start your first code line
        pass

        # # d = dataset.shape[1]
        #
        # for i in range(self._dim):
        #     self._parents_set[i] = set()
        #
        # # Stage1: causal skeleton reconstruction(PC-stable algorithm)
        # self._stage_1_learning()
        #
        # # Stage2: partial causal orders identification
        # self._stage_2_learning()
        #
        # # Stage3: latent confounders' detection
        # self._stage_3_learning()
        #
        # return self

    # ### CORRESPONDING TEST #####################################################
    # Loc: test_file.py >> test_function

    # ### AUXILIARY COMPONENT(S) #################################################
    # Function: get_skeleton_from_pc
    # Class: component_2
    # Object: component_3
    def _stage_1_learning(self):
        """
        # Tab: * 2
        # ------------------------------------------------------------------------

        Write down some descriptions here.

        Arguments
        ---------
        _dataset (attribute) : ndarray
            Write down some descriptions here.

        testing_text (attribute) : testing_text
            Write down some descriptions here.

        Parameters
        ----------
        testing_text (attribute) : testing_text
            Write down some descriptions here.

        Returns
        -------
        _skeleton : ndarray (update)
            Write down some descriptions here.

        _adjacency_matrix : ndarray (update)
            Write down some descriptions here.

        _stage1_time : float (update)
            Write down some descriptions here.

        testing_text : testing_text (update)
            Write down some descriptions here.

        self : object
            Write down some descriptions here.
        """

        # Start your first code line
        pass

        # skeleton, running_time = get_skeleton_from_stable_pc(X, alpha_level=self._alpha, return_time=True)
        #
        # self._skeleton = copy.copy(skeleton)
        # self._adjacency_matrix = copy.copy(skeleton)
        # self._stage1_time = running_time

    # ### CORRESPONDING TEST #####################################################
    # Loc: test_file.py >> test_function

    # ### AUXILIARY COMPONENT(S) #################################################
    # Function: get_adjacent_vars
    # Function: get_subsets
    # Class: component_2
    # Object: component_3
    def _stage_2_learning(self):
        """
        # Tab: * 2
        # ------------------------------------------------------------------------

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

        # Start your first code line
        pass
        start = time.perf_counter()

        Adj_set = get_Adj_set(self._skeleton)  # Quarry by Adj_set[variable] = {adjacent variable set}
        d = X.shape[1]
        T = X.shape[0]
        X_ = copy.copy(X)
        U = np.arange(d)

        unorient_pairs = []
        for i in range(d):
            for j in range(d):
                if i != j:
                    if (self._adjacency_matrix[i][j] == 1) and (self._adjacency_matrix[i][j] == 1):
                        unorient_pairs.append((i, j))

        if len(unorient_pairs) >= 2:
            for pair in unorient_pairs:
                pair = list(pair)
                i = pair[0]
                j = pair[1]

                # Test j -> i pairwisely:
                if self._Reg == "gam":
                    reg = LinearGAM()
                elif self._Reg == "xgboost":
                    reg = XGBRegressor()
                elif self._Reg == "mlp":
                    reg = MLPRegressor()
                else:
                    raise ValueError("Module haven't been built.")
                residual = residual_by_nonlinreg(X=X_[:, j], y=X_[:, i], Reg=reg)
                if self._IndTest == "hsic":
                    pval_ij = hsic2.hsic_gam(residual, check_vector(X_[:, j]), mode="pvalue")
                elif self._IndTest == "kci":
                    kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
                    pval_ij, _ = kci.compute_pvalue(check_vector(X_[:, j]), residual)
                else:
                    raise ValueError("Module haven't been built.")

                # Test i -> j pairwisely:
                if self._Reg == "gam":
                    reg = LinearGAM()
                elif self._Reg == "xgboost":
                    reg = XGBRegressor()
                elif self._Reg == "mlp":
                    reg = MLPRegressor()
                else:
                    raise ValueError("Module haven't been built.")
                residual = residual_by_nonlinreg(X=X_[:, i], y=X_[:, j], Reg=reg)
                if self._IndTest == "hsic":
                    pval_ji = hsic2.hsic_gam(residual, check_vector(X_[:, i]), mode="pvalue")
                elif self._IndTest == "kci":
                    kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
                    pval_ji, _ = kci.compute_pvalue(check_vector(X_[:, i]), residual)
                else:
                    raise ValueError("Module haven't been built.")

                if (pval_ij > self._alpha) and (pval_ji > self._alpha):
                    if pval_ij > pval_ji:
                        self._orient_adjacency_matrix(i_adj=[j], i=i)
                        self._parents_set[i].add(j)
                    else:
                        self._orient_adjacency_matrix(i_adj=[i], i=j)
                        self._parents_set[j].add(i)

                elif (pval_ij > self._alpha) and (pval_ji < self._alpha):
                    self._orient_adjacency_matrix(i_adj=[j], i=i)
                    self._parents_set[i].add(j)

                elif (pval_ij < self._alpha) and (pval_ji > self._alpha):
                    self._orient_adjacency_matrix(i_adj=[i], i=j)
                    self._parents_set[j].add(i)

                else:
                    continue

        end = time.perf_counter()

        self._stage2_time = end - start

        return self

    # ### CORRESPONDING TEST #####################################################
    # Loc: test_file.py >> test_function

    # ### AUXILIARY COMPONENT(S) #################################################
    # Function: _get_maximal_cliques
    # Class: component_2
    # Object: component_3
    def _stage_3_learning(self, X):
        """
        # Tab: * 2
        # ------------------------------------------------------------------------

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

        # Start your first code line
        pass

        start = time.perf_counter()

        X_ = copy.copy(X)
        T = X.shape[0]
        new_edge_determine = True

        while new_edge_determine:
            maximal_cliques = self._get_maximal_cliques()
            # Remove cliques which have been completely oriented in stage 2 and remind ones incompletely.
            maximal_cliques_incomplete = self._check_incomplete(maximal_cliques)

            if len(maximal_cliques_incomplete) == 0:
                new_edge_determine = False
            else:
                for maximal_clique in maximal_cliques_incomplete:
                    pairs = []
                    for vi in maximal_clique:
                        for vj in maximal_clique[maximal_clique.index(vi) + 1:]:
                            if not self._check_have_been_determined((vi, vj)):
                                pairs.append((vi, vj))

                    for pair in pairs:
                        pvals_direction = []  # P values for i->j first index and j->i second index.
                        i = pair[0]
                        j = pair[1]
                        for direction in [[i, j], [j, i]]:
                            # 1. For pairs haven't oriented, suppose we now want to test x -> y and consider explantory variables as follow:
                            #  (i)   x is definitely the explantory of y. (explantory_inside)
                            #  (ii)  y's parents have already found in stage 2. (explantory_outside)
                            #  (iii) force consideration of "third" variables in the maximal clique. (bfd_variables_in_clique)
                            explain = {direction[1]}
                            explantory_inside = {direction[0]}
                            explantory_inside_and_outside = explantory_inside | (self._parents_set[list(explain)[0]])
                            bfd_variables_in_clique = set(maximal_clique) - (
                                    explantory_inside | explain)  # "bpd" refers to "back door or front door".

                            # 2. Remove the nonlinear effect we have already known with the help of stage2 from explantory.
                            if not len(self._parents_set[list(explantory_inside)[0]]) > 0:
                                explantory_inside_data_remove_parents = X_[:, list(explantory_inside)]
                            else:
                                if self._Reg == "gam":
                                    reg = LinearGAM()
                                elif self._Reg == "xgboost":
                                    reg = XGBRegressor()
                                elif self._Reg == "mlp":
                                    reg = MLPRegressor()
                                else:
                                    raise ValueError("Module haven't been built.")
                                explantory_inside_data_remove_parents = residual_by_nonlinreg(
                                    X=X_[:, list(self._parents_set[list(explantory_inside)[0]])],
                                    y=X_[:, list(explantory_inside)],
                                    Reg=reg,
                                )

                            # 3. Fit and perform independent test.
                            if self._Reg == "gam":
                                reg = LinearGAM()
                            elif self._Reg == "xgboost":
                                reg = XGBRegressor()
                            elif self._Reg == "mlp":
                                reg = MLPRegressor()
                            else:
                                raise ValueError("Module haven't been built.")
                            residual = residual_by_nonlinreg(
                                X=X_[:, list((explantory_inside_and_outside) | (bfd_variables_in_clique))],
                                y=X_[:, list(explain)],
                                Reg=reg
                            )

                            if self._IndTest == "hsic":
                                pval = hsic2.hsic_gam(
                                    check_vector(explantory_inside_data_remove_parents), residual,
                                    mode="pvalue",
                                )
                            elif self._IndTest == "kci":
                                kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
                                pval, _ = kci.compute_pvalue(check_vector(explantory_inside_data_remove_parents),
                                                             residual)
                            else:
                                raise ValueError("Module haven't been built.")

                            pvals_direction.append(pval)

                        pval_ji = pvals_direction[0]
                        pval_ij = pvals_direction[1]

                        # Check result for both two directions.
                        if (pval_ij > self._alpha) and (pval_ji > self._alpha):
                            if pval_ij > pval_ji:
                                self._orient_adjacency_matrix(i_adj=[j], i=i)
                                self._parents_set[i].add(j)
                            else:
                                self._orient_adjacency_matrix(i_adj=[i], i=j)
                                self._parents_set[j].add(i)

                        elif (pval_ij > self._alpha) and (pval_ji < self._alpha):
                            self._orient_adjacency_matrix(i_adj=[j], i=i)
                            self._parents_set[i].add(j)

                        elif (pval_ij < self._alpha) and (pval_ji > self._alpha):
                            self._orient_adjacency_matrix(i_adj=[i], i=j)
                            self._parents_set[j].add(i)

                        else:
                            new_edge_determine = False

        end = time.perf_counter()
        self._stage3_time = end - start
        return self

    # ### SUBORDINATE COMPONENT(S) ###############################################
    # Function: _stage_2_learning

    # ### AUXILIARY COMPONENT(S) #################################################
    # Function: component_1
    # Class: component_2
    # Object: component_3
    def _orient_adjacency_matrix(self, i_adj, i):
        """
        # Tab: * 2
        # ------------------------------------------------------------------------

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

        # Start your first code line
        pass

        for j in i_adj:
            if (self._adjacency_matrix[i][j] == 1) and (self._adjacency_matrix[j][i] == 1):
                self._adjacency_matrix[i][j] = 1
                self._adjacency_matrix[j][i] = 0

        return self

    # ### SUBORDINATE COMPONENT(S) ###############################################
    # Function: _stage_3_learning

    # ### AUXILIARY COMPONENT(S) #################################################
    # Function: component_1
    # Class: component_2
    # Object: component_3
    def _get_maximal_cliques(self):
        """
        # Tab: * 2
        # ------------------------------------------------------------------------

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

        # Start your first code line
        pass

        undirect_graph_nx = nx.from_numpy_array(self._skeleton)
        iter = nx.find_cliques(undirect_graph_nx)
        temp = [clique for clique in iter]

        maximal_cliques = []

        for item in temp:
            if len(item) > 1:
                maximal_cliques.append(item)

        return maximal_cliques

    # def _check_incomplete(self, maximal_cliques):
    #     maximal_cliques_incomplete = []
    #     for maximal_clique in maximal_cliques:
    #         incomplete = False
    #         for vi in maximal_clique:
    #             for vj in maximal_clique[vi:]:
    #                 if (self._adjacency_matrix[vi][vj] == 1) and (self._adjacency_matrix[vj][vi] == 1):
    #                     incomplete = True
    #         if incomplete:
    #             maximal_cliques_incomplete.append(maximal_clique)
    #
    #     return maximal_cliques_incomplete
    #
    # def _check_have_been_determined(self, pair):
    #     i = pair[0]
    #     j = pair[1]
    #
    #     if (self._adjacency_matrix[i][j] == 1) and (self._adjacency_matrix[j][i] == 1):
    #         return False
    #     else:
    #         return True

    def display(self, stage_num):
        """
        # Tab: * 2
        # ------------------------------------------------------------------------

        Plot based on three stages, mark maximal cliques.

        Parameters
        ----------
        stage_num : string
            0, 1, 2, 3.
        """

        # Start your first code line
        pass

    @property
    def dataset_(self):
        return self._dataset

    @property
    def skeleton_(self):
        return self._skeleton

    @property
    def adjacency_matrix_(self):
        return self._adjacency_matrix

    @property
    def parents_set_(self):
        return self._parents_set

    @property
    def stage1_time_(self):
        return self._stage1_time

    @property
    def stage2_time_(self):
        return self._stage2_time

    @property
    def stage3_time_(self):
        return self._stage3_time

if __name__ == "__main__":
    np.random.seed(42)
    pass
