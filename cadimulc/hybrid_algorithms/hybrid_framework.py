from cadimulc.utils.causality_instruments import get_skeleton_from_pc
from abc import ABCMeta, abstractmethod
from numpy import ndarray

import copy as cp


class HybridFrameworkBase(metaclass=ABCMeta):
    """
    A hybrid causal discovery framework with established implementations
    of **discovering the causal skeleton** by the ***Peter-Clark* algorithm (PC algorithm)**.
    The framework is incorporated into the initial stage of both the
    *Nonlinear-MLC* and *MLC-LiNGAM* causal discovery algorithms.

    <!--
    Spirtes, Peter, Clark N. Glymour, and Richard Scheines.
    Causation, prediction, and search.
    MIT press, 2000.
    -->
    """

    def __init__(
            self,
            pc_alpha: float = 0.5,
            _dataset: ndarray = None,
            _dim: int = None,
            _skeleton: ndarray = None,
            _adjacency_matrix: ndarray = None,
            _parents_set: dict = {}
    ):
        """
        Parameters:
            pc_alpha:
                Significance level of independence tests (p_value), which is required by
                the constraint-based methodology incorporated in the initial stage of
                the hybrid causal discovery framework.
            _dataset:
                The observational dataset shown as a matrix or table,
                with a format of "sample (n) * dimension (d)."
                (input as Pandas dataframe is also acceptable)
            _dim: int
                The variable dimension corresponding to the causal graph.
            _skeleton:
                The estimated undirected graph corresponding to the causal graph.
            _adjacency_matrix:
                The estimated directed acyclic graph (DAG) corresponding to the causal graph.
            _parents_set: dict
                The child-parents relations associating with the adjacency matrix.
        """

        self.pc_alpha = pc_alpha

        self._dataset = _dataset
        self._dim = _dim
        self._skeleton = _skeleton
        self._adjacency_matrix = _adjacency_matrix
        self._parents_set = _parents_set
        self._running_time = 0.0
        self._stage1_time = 0.0
        self._stage2_time = 0.0
        self._stage3_time = 0.0

    @abstractmethod
    def fit(self, dataset):
        """
        Write down some descriptions here.

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

        # Start with the code line.

        return self

    # ### CORRESPONDING TEST ###################################################
    # Loc: test_hybrid_algorithms.py >> test_causal_skeleton_learning

    # ### SUBORDINATE COMPONENT(S) #############################################
    # Class: MLCLiNGAM, NonlinearMLC

    # ### AUXILIARY COMPONENT(S) ###############################################
    # Function: get_skeleton_from_pc()

    def _causal_skeleton_learning(self, dataset: ndarray) -> object:
        """
        Causal skeleton construction (based on the PC-stable algorithm).

        Parameters:
            dataset:
                The observational dataset shown as a matrix or table,
                with a format of "sample (n) * dimension (d)."
                (input as Pandas dataframe is also acceptable)

        Returns:
            self:
                Update `_skeleton` as the estimated undirected graph corresponding to
                the causal graph, initialize `_adjacency_matrix` via a copy of `_skeleton`,
                and record `_stage1_time` as the stage-1 computational time
                (causal skeleton learning is usually the first stage in
                hybrid-based causal discovery algorithm) .
        """

        # Arguments for testing:
        #   pc_alpha(parameter): float
        #   _dataset(attribute): dataframe

        self._dim = dataset.shape[1]
        self._dataset = dataset
        for i in range(self._dim):
            self._parents_set[i] = set()

        data = cp.copy(self._dataset)

        # Development notes:
        # Unify the linear independence test even for nonlinear-mlc.
        skeleton, running_time = get_skeleton_from_pc(
            data=data,
            alpha=self.pc_alpha,
            ind_test_type='linear'
        )

        self._skeleton = cp.copy(skeleton)
        self._adjacency_matrix = cp.copy(skeleton)
        self._stage1_time = running_time

        return self

    def display(self, stage_num: str = 'whole'):
        """
        Write down some descriptions here.

        Parameters:
            stage_num: Write down some descriptions here.
        """

        # Start with the code line.

        return

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

    @property
    def running_time_(self):
        return self._running_time
