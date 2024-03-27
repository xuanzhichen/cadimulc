"""Write down some descriptions here."""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * Write down some descriptions here.
#
# * Write down some descriptions here.


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Write down some descriptions here.                        20th.Jan, 2024
#
# * Write down some descriptions here.


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: Programming for stage-1 learning and the display function.
#
# Done:
# None


# auxiliary modules in causality instruments
from cadimulc.utils.causality_instruments import get_skeleton_from_pc

# basic
from abc import ABCMeta, abstractmethod

import copy as cp
from numpy import ndarray


class HybridFrameworkBase(metaclass=ABCMeta):
    """
    A hybrid causal discovery framework with established implementations
    of discovering causal skeleton by *Peter-Clark* algorithm.
    The framework is incorporated into the initial stage of both
    *Nonlinear-MLC* and *MLC-LiNGAM* causal discovery algorithms.
    """

    def __init__(
            self,
            pc_alpha: float = 0.5,
            _dataset: ndarray = None,
            _dim: int = None,
            _skeleton: ndarray = None,
            _adjacency_matrix: ndarray = None,
            _parents_set: dict = {},
            _running_time: float = 0.0
    ):
        """
        Parameters:
            pc_alpha: float (default: 0.5)
                Write down some descriptions here.
            _dataset: dataframe
                Write down some descriptions here.

            _dim: int
                Write down some descriptions here.

            _skeleton: ndarray
                Write down some descriptions here.

            _adjacency_matrix: ndarray
                Write down some descriptions here.

            _parents_set: dict
                Write down some descriptions here.
        """

        self.pc_alpha = pc_alpha

        self._dataset = _dataset
        self._dim = _dim
        self._skeleton = _skeleton
        self._adjacency_matrix = _adjacency_matrix
        self._parents_set = _parents_set
        self._running_time = _running_time

    @abstractmethod
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

        # self._dim = dataset.shape[1]
        #
        # for i in range(self._dim):
        #     self._parents_set[i] = set()
        #
        # self._stage_1_learning()

        # Start the code line

        return self

    # ### CORRESPONDING TEST ###################################################
    # Loc: test_hybrid_algorithms.py >> test_causal_skeleton_learning

    # ### SUBORDINATE COMPONENT(S) #############################################
    # Function: None
    # Class: MLCLiNGAM, NonlinearMLC

    # ### AUXILIARY COMPONENT(S) ###############################################
    # Function: None
    def _causal_skeleton_learning(self, dataset: ndarray) -> object:
        """
        Write down some descriptions here.

        <!--
        Arguments:
            pc_alpha (parameter) : float
                Write down some descriptions here.
            _dataset (attribute) : ndarray
                Write down some descriptions here.
        -->

        Parameters:
            dataset: Write down some descriptions here.

        Returns:
            Update ``_skeleton``, ``_adjacency_matrix``, and ``_stage1_time``
        """

        self._dim = dataset.shape[1]
        self._dataset = dataset

        for i in range(self._dim):
            self._parents_set[i] = set()

        data = cp.copy(self._dataset)

        # Notes for developer:
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

    def display(self, stage_num):
        """
        Plot based on three stages; Mark maximal cliques.

        Parameters
        ----------
        stage_num : string
            0, 1, 2, 3.
        """

        # Start the code line

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
