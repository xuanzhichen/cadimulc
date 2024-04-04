"""The old implementation version of Nonlinear-MLC by Xuanzhi Chen."""

# Basic
import numpy as np
import networkx as nx
import copy
import time
import warnings
warnings.filterwarnings("ignore")

# Causal Discover
# from tigramite.pcmci import PCMCI
from causallearn.search.ConstraintBased.PC import pc

# Nonlinear Regression
from pygam import LinearGAM
# from xgboost.sklearn import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Independent Test
# from tigramite.independence_tests import GPDC
from causallearn.search.FCMBased.lingam import hsic2
from causallearn.utils.KCI.KCI import KCI_UInd
from my-causallearn.utils.causal_tools import fisher_hsic

# Accompanying code for NonlinearMLC as follow
from my-causallearn.utils.causal_tools import get_skeleton_from_stable_pc, get_skeleton_from_pcmci, residual_by_nonlinreg
from my-causallearn.utils.basic_tools import get_Adj_set, check_vector, subset


class NonlinearMLC():
    def __init__(self, handle_time_series, Reg, IndTest, alpha=0.05, max_lag=None, dag_gt=None, lv_info=None, stage1_input_pdag=False):
        """
        Example
        -------
        >>> #################### USAGE-1 ########################
        >>> # Recommended setting for non-sequential.
        >>> nonlin_mlc = NonlinearMLC(
                                       handle_time_series=False,
                                       Reg="mlp",
                                       IndTest="kci"
                                       )
        >>> nonlin_mlc.fit(X)
        >>> #################### USAGE-2 ########################
        >>> # Recommended setting for time series.
        >>> nonlin_mlc = NonlinearMLC(
                                       handle_time_series=True,
                                       max_lag=1
                                       Reg="gam",
                                       IndTest="hsic"
                               )
        >>> nonlin_mlc.fit(X)
        >>> #################### USAGE-3 ########################
        >>> # Test NonlinearMLC, you can pass the skeleton ground-truth instead of learning from PC algorithm.
        >>> # true_dag is DAG represented by numpy array and lv_info is as follow:
        >>> latent_variable_info = {
                                        "confounder" : {
                                            0 : {0, 1},  # lv0 confounding n0 and n1
                                            1 : {1, 2},  # lv1 confounding n1 and n2
                                        },
                                        "intermediator" : {
                                            # 2: [1, 2],  # lv2 inserts n1 and n2
                                        }
                                    }
        >>> nonlin_mlc = NonlinearMLC(
                                       handle_time_series=Any, Reg=Any, IndTest=Any,
                                       dag_gt = true_dag,
                                       lv_info = latent_variable_info
                               )
        >>> nonlin_mlc.fit(X)
        """
        self._handle_time_series = handle_time_series
        self._Reg = Reg
        self._IndTest = IndTest
        self._alpha = alpha
        self._max_lag = max_lag
        self._stage1_input_pdag = stage1_input_pdag

        # Use for testing
        self._dag_gt = dag_gt # Skeleton ground-truth with latent variables.
        self._lv_info = lv_info # Latent variables information containing confounders and intermediators.

        self._skeleton = None
        self._adjacency_matrix = None
        self._parents_set = {}
        self._stage1_time = 0
        self._stage2_time = 0
        self._stage3_time = 0


    def fit(self, X):
        d = X.shape[1]

        for i in range(d):
            self._parents_set[i] = set()

        # Stage1: causal skeleton reconstruction(PC-stable algorithm)
        # Test performance by passing skeleton ground-truth with latent variables.
        if (self._dag_gt is not None) and (self._lv_info is not None):
            from utils.data_processing import SkeletonGtLv
            skeleton_gt_lv = SkeletonGtLv.get_skeleton_gt_lv(
                true_dag=self._dag_gt, latent_variable_info=self._lv_info
            )

            self._skeleton = copy.copy(skeleton_gt_lv)
            self._adjacency_matrix = copy.copy(skeleton_gt_lv)

        else:
            # Stage1: causal skeleton reconstruction(PC-stable algorithm)
            self._stage_1_learning(X)

        # Stage2: partial causal orders identification
        self._stage_2_learning(X)

        # Stage3: latent confounders' detection
        self._stage_3_learning(X)

        return self


    def _stage_1_learning(self, X):
        if not self._handle_time_series:
            skeleton, running_time = get_skeleton_from_stable_pc(X, alpha_level=self._alpha, return_time=True)

            self._skeleton = copy.copy(skeleton)
            self._adjacency_matrix = copy.copy(skeleton)
            self._stage1_time = running_time

            return self

        else:
            begin = time.perf_counter()
            skeleton, adjacency_matrix = get_skeleton_from_pcmci(X, alpha_level=self._alpha)
            end = time.perf_counter()

            self._skeleton = copy.copy(skeleton)

            if self._stage1_input_pdag:
                self._adjacency_matrix = copy.copy(adjacency_matrix)
            else:
                self._adjacency_matrix = copy.copy(skeleton)

            self._stage1_time = end - begin

            return self


    def _stage_2_learning(self, X):
        start = time.perf_counter()

        Adj_set = get_Adj_set(self._skeleton)
        d = X.shape[1]
        T = X.shape[0]
        X_ = copy.copy(X)
        U = np.arange(d)

        repeat = True
        while repeat:
            if len(U) == 1:
                break

            repeat = False
            pval_is_leaf = 0

            for i in U:
                i_adj = Adj_set[i] & set(U)
                if len(i_adj) == 0:
                    U = U[U != i]
                    continue

                if not self._handle_time_series:
                    explain = X_[:, i]
                    explantory= X_[:, list(i_adj)]

                    if self._Reg == "gam":
                        reg = LinearGAM()
                    elif self._Reg == "xgboost":
                        reg = XGBRegressor()
                    elif self._Reg == "mlp":
                        reg = MLPRegressor()
                    else:
                        raise ValueError("Module haven't been built.")
                    # print(explantory.shape, explain.shape)
                    residual = residual_by_nonlinreg(X=explantory, y=explain, Reg=reg)

                    # Perform independent Test
                    if self._IndTest == "hsic":
                        pval = hsic2.hsic_gam(check_vector(explantory), residual, mode="pvalue")
                    elif self._IndTest == "kci":
                        kci = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian")
                        pval, _ = kci.compute_pvalue(check_vector(explantory), residual)
                    else:
                        raise ValueError("Module haven't been built.")

                else:
                    explain = X_[self._max_lag:, i]

                    explantory_current = X_[self._max_lag:, list(i_adj)]
                    explantory_historical = X_[self._max_lag-1:-1, list(i_adj | {i})]
                    if self._max_lag > 1:
                        for p in range(2, self._max_lag + 1):
                            explantory_historical = np.hstack((explantory_historical, X_[self._max_lag-p:-p, list(i_adj | {i})]))
                    explantory = np.hstack((explantory_historical, explantory_current))

                    if self._Reg == "gam":
                        reg = LinearGAM()
                    else:
                        raise ValueError("Module haven't been built.")

                    residual = residual_by_nonlinreg(X=explantory, y=explain, Reg=reg)

                    expaltory_shift = X_[self._max_lag:, list(i_adj)]
                    for shift in range(1, 5):
                        temp = np.zeros([T - self._max_lag, len(i_adj)])

                        if shift == 1:
                            temp[self._max_lag:, ] = X_[self._max_lag:-shift, list(i_adj)]
                        else:
                            temp[self._max_lag + (shift - 1):, ] = X_[self._max_lag:-shift, list(i_adj)]

                        expaltory_shift = np.hstack((expaltory_shift, temp))

                    if self._IndTest == "hsic":
                        pval = fisher_hsic(expaltory_shift, residual)
                    else:
                        raise ValueError("Module haven't been built.")

                if pval > pval_is_leaf:
                    pval_is_leaf = pval
                    sink = i
                    sink_parent = i_adj

            if pval_is_leaf > self._alpha:
                repeat = True
                U = U[U != sink]
                self._orient_adjacency_matrix(i_adj=sink_parent, i=sink)
                self._parents_set[sink].union(sink_parent)

        unorient_pairs = []
        for i in range(d):
            for j in range(d):
                if i != j:
                    if (self._adjacency_matrix[i][j] == 1) and (self._adjacency_matrix[i][j] == 1):
                        unorient_pairs.append((i,j))

        if len(unorient_pairs) >= 2:
            for pair in unorient_pairs:
                pair = list(pair)
                i = pair[0]
                j = pair[1]

                if not self._handle_time_series:
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

                else:
                    # Test j -> i pairwisely:
                    explain = X_[self._max_lag:, i]
                    explantory_current = check_vector(X_[self._max_lag:, j])
                    explantory_historical = X_[self._max_lag - 1:-1, [i, j]]
                    if self._max_lag > 1:
                        for p in range(2, self._max_lag + 1):
                            explantory_historical = np.hstack((explantory_historical, X_[self._max_lag - p:-p, [i, j]]))
                    explantory = np.hstack((explantory_historical, explantory_current))

                    if self._Reg == "gam":
                        reg = LinearGAM()
                    else:
                        raise ValueError("Module haven't been built.")
                    residual = residual_by_nonlinreg(X=explantory, y=explain, Reg=reg)

                    expaltory_shift = check_vector(X_[self._max_lag:, j])
                    for shift in range(1, 5):
                        temp = np.zeros([T - self._max_lag, 1])

                        if shift == 1:
                            temp[self._max_lag:, ] = check_vector(X_[self._max_lag:-shift, j])
                        else:
                            temp[self._max_lag + (shift - 1):, ] = check_vector(X_[self._max_lag:-shift, j])

                        expaltory_shift = np.hstack((expaltory_shift, temp))

                    if self._IndTest == "hsic":
                        pval_ij = fisher_hsic(expaltory_shift, residual)
                    else:
                        raise ValueError("Module haven't been built.")

                    # Test i -> j pairwisely:
                    explain = X_[self._max_lag:, j]
                    explantory_current = check_vector(X_[self._max_lag:, i])
                    explantory_historical = X_[self._max_lag - 1:-1, [i, j]]
                    if self._max_lag > 1:
                        for p in range(2, self._max_lag + 1):
                            explantory_historical = np.hstack((explantory_historical, X_[self._max_lag - p:-p, [i, j]]))
                    explantory = np.hstack((explantory_historical, explantory_current))

                    if self._Reg == "gam":
                        reg = LinearGAM()
                    else:
                        raise ValueError("Module haven't been built.")
                    residual = residual_by_nonlinreg(X=explantory, y=explain, Reg=reg)

                    expaltory_shift = check_vector(X_[self._max_lag:, i])
                    for shift in range(1, 5):
                        temp = np.zeros([T - self._max_lag, 1])

                        if shift == 1:
                            temp[self._max_lag:, ] = check_vector(X_[self._max_lag:-shift, i])
                        else:
                            temp[self._max_lag + (shift - 1):, ] = check_vector(X_[self._max_lag:-shift, i])

                        expaltory_shift = np.hstack((expaltory_shift, temp))

                    if self._IndTest == "hsic":
                        pval_ji = fisher_hsic(expaltory_shift, residual)
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


    def _stage_3_learning(self, X):
        start = time.perf_counter()

        X_ = copy.copy(X)
        T = X.shape[0]
        new_edge_determine = True

        while new_edge_determine:
            maximal_cliques = self._get_maximal_cliques()
            maximal_cliques_incomplete = self._check_incomplete(maximal_cliques)

            if len(maximal_cliques_incomplete) == 0:
                new_edge_determine = False
            else:
                for maximal_clique in maximal_cliques_incomplete:
                    pairs = []
                    for vi in maximal_clique:
                        for vj in maximal_clique[maximal_clique.index(vi)+1:]:
                            if not self._check_have_been_determined((vi, vj)):
                                pairs.append((vi, vj))

                    for pair in pairs:
                        pvals_direction = []
                        i = pair[0]
                        j = pair[1]
                        for direction in [[i, j], [j, i]]:
                            explain = {direction[1]}
                            explantory_inside = {direction[0]}
                            explantory_inside_and_outside = explantory_inside | (self._parents_set[list(explain)[0]])
                            bfd_variables_in_clique = set(maximal_clique) - (explantory_inside | explain) # "bpd" refers to "back door or front door".

                            if not len(self._parents_set[list(explantory_inside)[0]]) > 0:
                                explantory_inside_data_remove_parents = X_[:, list(explantory_inside)]
                            else:
                                if not self._handle_time_series:
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
                                else:
                                    explantory_inside_data_remove_parents = X_[:, list(explantory_inside)]

                                    data_current = X_[self._max_lag:, list(self._parents_set[list(explantory_inside)[0]])]
                                    data_historical = X_[self._max_lag - 1:-1, list(self._parents_set[list(explantory_inside)[0]])]
                                    if self._max_lag > 1:
                                        for p in range(2, self._max_lag + 1):
                                            data_historical = np.hstack(
                                                (data_historical,  X_[self._max_lag - p:-p, list(self._parents_set[list(explantory_inside)[0]])])
                                            )
                                    data = np.hstack((data_historical, data_current))

                                    if self._Reg == "gam":
                                        reg = LinearGAM()
                                    else:
                                        raise ValueError("Module haven't been built.")
                                    explantory_inside_data_remove_parents[self._max_lag:, ] = residual_by_nonlinreg(
                                        X=data,
                                        y=X_[self._max_lag:, list(explantory_inside)],
                                        Reg=reg,
                                    )

                            if not self._handle_time_series:
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
                                    pval, _ = kci.compute_pvalue(check_vector(explantory_inside_data_remove_parents), residual)
                                else:
                                    raise ValueError("Module haven't been built.")

                                pvals_direction.append(pval)

                            else:
                                explain_data = X_[self._max_lag:, list(explain)]
                                explantory_inside_and_outside_data_current = X_[self._max_lag:, list((explantory_inside_and_outside) | (bfd_variables_in_clique))]


                                explantory_inside_and_outside_data_historical = X_[self._max_lag - 1:-1, list((explantory_inside_and_outside) | (bfd_variables_in_clique) | explain)]
                                if self._max_lag > 1:
                                    for p in range(2, self._max_lag + 1):
                                        explantory_inside_and_outside_data_historical = np.hstack((
                                            explantory_inside_and_outside_data_historical,
                                            X_[self._max_lag - p:-p, list((explantory_inside_and_outside) | (bfd_variables_in_clique)) | explain]
                                        ))
                                explantory_inside_and_outside_data = np.hstack(
                                    (explantory_inside_and_outside_data_current, explantory_inside_and_outside_data_historical)
                                )

                                if self._Reg == "gam":
                                    reg = LinearGAM()
                                else:
                                    raise ValueError("Module haven't been built.")
                                residual = residual_by_nonlinreg(
                                    X=explantory_inside_and_outside_data,
                                    y=explain_data,
                                    Reg=reg
                                )

                                explantory_inside_data_remove_parents_shift = copy.copy(explantory_inside_data_remove_parents[self._max_lag:, ])
                                explantory_inside_data_remove_parents_shift = check_vector(explantory_inside_data_remove_parents_shift)
                                for shift in range(1, 5):
                                    temp = np.zeros([T - self._max_lag, 1])

                                    if shift == 1:
                                        temp[self._max_lag:, ] = check_vector(explantory_inside_data_remove_parents[self._max_lag:-shift, ])
                                    else:
                                        temp[self._max_lag + (shift - 1):, ] = check_vector(explantory_inside_data_remove_parents[self._max_lag:-shift, ])

                                    explantory_inside_data_remove_parents_shift = np.hstack((explantory_inside_data_remove_parents_shift, temp))

                                if self._IndTest == "hsic":
                                    pval = fisher_hsic( explantory_inside_data_remove_parents_shift, residual)
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


    def _orient_adjacency_matrix(self, i_adj, i):
        """ Orient from Adj(i) to i.
        """
        for j in i_adj:
            if (self._adjacency_matrix[i][j] == 1) and (self._adjacency_matrix[j][i] == 1):
                self._adjacency_matrix[i][j] = 1
                self._adjacency_matrix[j][i] = 0

        return self


    def _get_maximal_cliques(self):
        """ Get maximal cliques (size > 1) directly from skeleton.
        """
        undirect_graph_nx = nx.from_numpy_array(self._skeleton)
        iter = nx.find_cliques(undirect_graph_nx)
        temp = [clique for clique in iter]

        maximal_cliques = []

        for item in temp:
            if len(item) > 1:
                maximal_cliques.append(item)

        return maximal_cliques


    def _check_incomplete(self, maximal_cliques):
        maximal_cliques_incomplete = []
        for maximal_clique in maximal_cliques:
            incomplete = False
            for vi in maximal_clique:
                for vj in maximal_clique[vi:]:
                    if (self._adjacency_matrix[vi][vj] == 1) and (self._adjacency_matrix[vj][vi] == 1):
                        incomplete = True
            if incomplete:
                maximal_cliques_incomplete.append(maximal_clique)

        return maximal_cliques_incomplete


    def _check_have_been_determined(self, pair):
        i = pair[0]
        j = pair[1]

        if (self._adjacency_matrix[i][j] == 1) and (self._adjacency_matrix[j][i] == 1):
            return False
        else:
            return True


    # def get_pair_pvalue_stat(X, x, y):
    #     U_adj = get_adj_dictionary(self._skeleton)
    #     n = X.shape[0]
    #
    #     # estimate x->y
    #     residual_forward = self._get_residual_by_linear_regression(X=X[:, list(set(U_adj[x]) | set([x]))], y=X[:, y])
    #     pval_forward = self._independent_test(X[:, x], residual_forward)
    #
    #     # estimate y->x
    #     residual_backward = self._get_residual_by_linear_regression(X=X[: ,list(set(U_adj[y]) | set([y]))], y=X[:, x])
    #     pval_backward = self._independent_test(X[:, y], residual_backward)
    #
    #     p_stat = -2 * (np.log(pval_forward) + np.log(pval_backward))
    #
    #     return p_stat


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