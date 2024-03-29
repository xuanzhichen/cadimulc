"""The empirical data simulation based on structure causal models"""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License

# Testing: cadimulc/tests/test_generation.py


# ### DEVELOPMENT PROGRESS (LEAST) ########################################################
# * Programming of Generator and testing were done, ready to stack out to test the basic
#   usage without taking account of latent confounding generation.           11th.Dec, 2023
#
# * Stacked into 'generation.py' before continuing
#   'test_hybrid_algorithms.py > test_procedure_causal_asymmetry()' complementing parameters
#   within the (original) Generator class.                                   20th.Nov, 2023


# ### TO-DO LIST (LEAST) ##################################################################
# Done:
# _TODO: The beginning of the project refactorings starts with the data generation module
#       (refer to the old project).


from __future__ import annotations
from numpy import ndarray
from copy import copy

import numpy as np
import networkx as nx


# ### CODING DATE #########################################################################
# Module Init   : 2023-11-20
# Module Update : 2024-03-29

class Generator(object):
    """
    The `Generator` simulates the empirical data implied by the
    **structure causal models** (SCMs).
    Primary parameters for `Generator`'s simulation consist of the **model classes**
    (e.g. linear or non-linear) and the **(independent) noise distributions** (e.g.
    Gaussian or non-Gaussian).

    Take causation in **graphical context**,
    where a variable $y_i$ is supposed to be the effect of its parents $pa(y_i)$.
    Then the data relative to $y_i$ is expected to be generated
    given the (group of) data relative to $pa(y_i)$, following the causal mapping
    mechanism $F$ characterized as SCMs.

    Currently, research in causal discovery has suggested that
    **structural-identifiable** empirical data should be further generated by
    a special "genus" of the SCMs, which is normally referred to as the
    **additive noise models** (ANMs) shown in the following

    $$
    y_{i} := F(pa(y_i), e_{i}):= \sum_{x_{j} \in pa(y_i)} f( x_{j}) + e_{i},
    $$

    where $f(\cdot)$ denotes the linear or non-linear function, and $e_{i}$ refers to the
    independent noise obeying the Gaussian or non-Gaussian distributions.

    !!! note "Structural-identifiable SCMs simulation in CADIMULC in light of related literature"
        **linear**: linear non-Gaussian acyclic models (LiNGAM)<sup>[1]</sup>,
        referring to the experiment setup by MLC-LiNGAM<sup>[2]</sup>.

        **non-linear**: causal additive models (CAM)<sup>[3]</sup>,
        referring to the experiment setup by CAM-UV<sup>[4]</sup>.

    <!--
    [1] Shimizu, Shohei, Patrik O. Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan.
    "A linear non-Gaussian acyclic model for causal discovery."
    Journal of Machine Learning Research. 2006.

    [2] Chen, Wei, Ruichu Cai, Kun Zhang, and Zhifeng Hao.
    Causal discovery in linear non-gaussian acyclic model with multiple latent confounders.
    IEEE Transactions on Neural Networks and Learning Systems. 2021.

    [3] Bühlmann, Peter, Jonas Peters, and Jan Ernest.
    "CAM: Causal additive models, high-dimensional order search and penalized regression." 2014.

    [4] Maeda, Takashi Nicholas, and Shohei Shimizu.
    "Causal additive models with unobserved variables."
    In Uncertainty in Artificial Intelligence. 2021.
    -->
    """

    def __init__(self,
                 graph_node_num: int,
                 sample: int,
                 causal_model: str = 'hybrid_nonlinear',
                 noise_type: str = 'Gaussian',
                 noise_scale: int | str = 'default',
                 sparsity: float = 0.3,
                 _skeleton: ndarray | None = None,
                 _dag: ndarray | None = None,
                 _dataset: ndarray | None = None):
        """
        Parameters:
            graph_node_num:
                Number of the vertex in a causal graph (ground-truth),
                which represents the number of the variable given a causal model
                (recommend: < 15).
            sample:
                Size of the dataset generated from the SCMs
                (recommend: < 10000).
            causal_model:
                **Refer to structural-identifiable SCMs simulation in light of related literature.**
                e.g. LiNGAM (str: lingam), CAM (str: hybrid_nonlinear).
            noise_type:
                **Refer to structural-identifiable SCMs simulation in light of related literature.**
                e.g. Gaussian (str: Gaussian),
                uniform distribution as non-Gaussian (str: non-Gaussian).
            noise_scale:
                "Default" as following the experiment setup in light of related literature.
            sparsity:
                Control the sparsity of a causal graph (ground-truth) (recommend: 0.3).

        !!! warning "The causal model should be carefully paired with the noise type"
            - If `causal model = "lingam"`, the noise distribution
                must satisfy `noise_type="non-Gaussian"`.
            - If `causal model = "hybrid_nonlinear"`, the noise distribution
                can either choose `noise_type="Gaussian"` or `noise_type="non-Gaussian"`.
                However, **evaluation in CADIMULC suggests that Gaussian noise is more preferable
                to yield identifiable results.**

        <!--
        Attributes:
            _skeleton:
                The undirected graph corresponding to the causal graph (ground-truth).
            _dag:
                The directed acyclic graph (DAG) corresponding to the causal graph (ground-truth).
            _dataset:
                The generated dataset in a format (n * d), n = sample, d = graph_node_num.
        -->
        """

        self.graph_node_num = graph_node_num
        self.sample = sample
        self.causal_model = causal_model
        self.sparsity = sparsity

        # Development notes: Fine tune the associating noise simulation setup according to
        # the empirical conclusions in 'test_causality_instruments.py'.

        if causal_model == 'lingam':
            if noise_type != 'non-Gaussian':
                raise ValueError("lingam model stands for Non-Gaussian noise.")

            self.noise_type = noise_type

            # relatively smaller noise scale for linear models (empirical setting)
            if noise_scale == 'default':
                self.noise_scale = 1
            else:
                self.noise_scale = noise_scale

        elif causal_model == 'hybrid_nonlinear':
            if noise_type == 'non-Gaussian':
                self.noise_type = noise_type

                # relatively larger noise scale for non-linear models (empirical setting)
                if noise_scale == 'default':
                    self.noise_scale = 10
                else:
                    self.noise_scale = noise_scale

            elif noise_type == 'Gaussian':
                self.noise_type = noise_type

                if noise_scale == 'default':
                    self.noise_scale = 1
                else:
                    raise ValueError("Gaussian noise stands a scale that equals to 1.")

            else:
                raise ValueError("Please input established st ring parameters: "
                                 "'Gaussian' or 'non-Gaussian'.")

        else:
            raise ValueError("Please input established string parameters: "
                             "'lingam' or 'hybrid_nonlinear'.")

        self.skeleton = _skeleton
        self.dag = _dag
        self.data = _dataset

    def run_generation_procedure(self) -> object:
        """ Run the common **two-steps** procedure for SCMs data generation:

        1. Generate a random DAG in light of the well-known Erdős–Rényi model;
        2. Provided a topological order converted by the DAG, generate each variable $y_i$ by
        summarizing the effects of its parents $pa(y_i)$.

        Returns:
            self:
                Update the `Generator`'s attributes: `_skeleton` as the undirected graph
                corresponding to the causal graph (ground-truth), `_dag` as the
                directed acyclic graph (DAG) corresponding to the (ground-truth),
                and `_dataset` in a format (n * d) (n = `sample`, d = `graph_node_num`).
        """

        self._clear()

        # Generate a random DAG in light of the well-known Erdős–Rényi model.
        self._generate_dag()

        # Provided a topological order converted by the DAG, generate each variable by
        # summarizing the effects of its parents
        self._generate_data()

        return self

    def unpack(self):
        return self.dag, self.data

    def _clear(self):
        self.skeleton = None
        self.dag = None
        self.data = None

        return self

    # ### AUXILIARY COMPONENT(S) ##########################################################
    # Function: _get_undigraph
    # Function: _orient

    def _generate_dag(self) -> object:
        """
        Generate a random DAG in light of the well-known Erdős–Rényi model.

        Returns:
            self: Update `_dag` (DAG) represented as a bool adjacency matrix.
        """

        # undirected graph
        undigraph = self._get_undigraph(
            graph_node_num=self.graph_node_num,
            sparsity=self.sparsity
        )
        self.skeleton = copy(undigraph)

        # directed graph
        digraph = self._orient(undigraph)
        self.dag = copy(digraph)

        return self

    # ### AUXILIARY COMPONENT(S) ##########################################################
    # Function: _add_random_noise()
    # Function: _simulate_causal_model()

    def _generate_data(self) -> object:
        """
        Provided a topological order converted by the DAG, generate each variable by
        summarizing the effects of its parents.

        Returns:
            self: Update `_data` represented as a (n * d) numpy array (n = sample,
             d = graph_node_num).
        """

        self.data = np.zeros([self.sample, self.graph_node_num])

        # topological order converted by the DAG
        dag_nx = nx.DiGraph(self.dag.T)
        topo_order = list(nx.topological_sort(dag_nx))

        # data generation in light of additive noise model
        for child_index, child_var in enumerate(topo_order):
            parent_vars = list(dag_nx.predecessors(child_var))
            parent_indexes = [topo_order.index(var) for var in parent_vars]

            # independence noise
            self._add_random_noise(var_index=child_index)

            # summary of the parents effects
            if len(parent_vars) > 0:
                for parent_index in parent_indexes:
                    self._simulate_causal_model(
                        child_index=child_index,
                        parent_index=parent_index
                    )

        # standard deviation of the dataset (default setting)
        self.data = self.data / np.std(self.data, axis=0)

        return self

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # _generate_dag()

    @staticmethod
    def _get_undigraph(graph_node_num, sparsity):
        undigraph_nx = nx.random_graphs.erdos_renyi_graph(
            n=graph_node_num,
            p=sparsity
        )
        undigraph_np = nx.to_numpy_array(undigraph_nx)

        return undigraph_np

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # _generate_dag()

    @staticmethod
    def _orient(undigraph):
        """
        The code fragments in _orient() refer to:
        https://github.com/huawei-noah/trustworthyAI/blob/22eb5a674b66e33a3ee3d00e7dac9d0a0ecd7bf3/gcastle/castle/datasets/simulator.py#L53
        """

        # Generate a permutation matrix in preparation for the undirected graph
        # represented as a matrix.
        permu_mat = np.random.permutation(np.eye(undigraph.shape[0]))

        # first permutation (after an undirected graph)
        graph_temp1 = permu_mat.T @ undigraph @ permu_mat

        # extract the lower triangle part of graph_temp1
        # by excluding the main diagonal and elements above it
        graph_temp2 = np.tril(graph_temp1, -1)

        # second permutation (after a directed graph)
        permu_mat2 = np.random.permutation(np.eye(undigraph.shape[0]))
        digraph = permu_mat2.T @ graph_temp2 @ permu_mat2

        return digraph

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # _generate_data()

    def _simulate_causal_model(self, child_index, parent_index):

        # Development notes:  Default settings of hyperparameters might require
        # being fine-tuned depends on different purposes for simulation.

        if self.causal_model == 'hybrid_nonlinear':
            corr = round(np.random.uniform(low=0.3, high=0.5), 3)

            nonlinear_label = np.random.randint(low=1, high=3 + 1)
            if nonlinear_label == 1:
                self.data[:, child_index] += corr * np.sin(
                    self.data[:, parent_index]
                )
            elif nonlinear_label == 2:
                self.data[:, child_index] += corr * np.sqrt(
                    np.abs(self.data[:, parent_index])
                )
            else:
                self.data[:, child_index] += corr * np.power(
                    self.data[:, parent_index], 3
                )

        elif self.causal_model == 'lingam':
            corr = round(np.random.uniform(low=0.7, high=0.9), 3)

            self.data[:, child_index] += corr * self.data[:, parent_index]

        else:
            raise ValueError("Please input established string parameters: "
                             "'lingam' or 'hybrid_nonlinear'.")

        return self

    # ### SUBORDINATE COMPONENT(S) ########################################################
    # _generate_data()

    def _add_random_noise(self, var_index):

        # Development notes:  Default settings of hyperparameters might require
        # being fine-tuned depends on different purposes for simulation.

        if self.noise_type == 'Gaussian':
            self.data[:, var_index] += np.random.normal(
                loc=0.0,
                scale=self.noise_scale,
                size=self.sample
            )

        elif self.noise_type == 'non-Gaussian':
            self.data[:, var_index] += np.random.uniform(
                low=np.negative(self.noise_scale),
                high=self.noise_scale,
                size=self.sample
            )

        else:
            raise ValueError("Please input established string parameters: "
                             "'Gaussian' or 'non-Gaussian'.")

        return self
