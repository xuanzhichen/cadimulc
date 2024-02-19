"""Description"""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * None


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * def _generate_data() passes the test.                     11th.Dec, 2023
#
# * Working in associating functions (causal model and noise) of
#   def _generate_data(), ready to stack out to test the basic usage of Gen-
#   erator without taking account latent confounders.         23rd.Nov, 2023
#
# * Stack into generation.py before basically run through
#   test_hybrid_algorithms.py > test_procedure_causal_asymmetry(), complementing
#   parameters with Generator class.                          20th.Nov, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# None
#
# Done:
# TODO: Build def _generate_data() (refer to my-causallearn)


import numpy as np
import networkx as nx

from sklearn.utils import check_array
from copy import copy


class Generator(object):
    """
    DAG with / without latent confounders

    Parameters
    ----------
    graph_node_num : int
        - description-1
        - description-2
    causal_model : string
        - description-1
        - description-2
    sample : int
        - description-1
        - description-2
    noise_type : string
        - description-1
        - description-2
    noise_scale : int
        - description-1
        - description-2

    Attributes
    ----------
    skeleton : ndarray
        - description-1
        - description-2
    dag : ndarray
        - description-1
        - description-2
    dataset : ndarray
        - description-1
        - description-2

    Examples
    --------
    >>>
    >>>
    Notes
    -----
    * MLCLiNGAM's simulating assumption
    * CAM-UV baseline model
    """

    def __init__(self,
                 graph_node_num,
                 sample,
                 causal_model='hybrid_nonlinear',
                 noise_type='Gaussian',
                 noise_scale='default',
                 sparsity=0.3):

        self.graph_node_num = graph_node_num
        self.sample = sample
        self.causal_model = causal_model
        self.sparsity = sparsity

        # Fine tune the associating noise simulating setting according to
        # the empirical conclusions in test_causality_instruments.py.
        if causal_model == 'lingam':
            if noise_type != 'non-Gaussian':
                raise ValueError("lingam model stands for Non-Gaussian noise.")

            self.noise_type = noise_type

            # relatively small noise scale for linear model (empirical setting)
            if noise_scale == 'default':
                self.noise_scale = 1
            else:
                self.noise_scale = noise_scale

        elif causal_model == 'hybrid_nonlinear':
            if noise_type == 'non-Gaussian':
                self.noise_type = noise_type

                # relatively large noise scale for non-linear model (empirical setting)
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
                raise ValueError("Please input established string parameters: "
                                 "'Gaussian' or 'non-Gaussian'.")

        else:
            raise ValueError("Please input established string parameters: "
                             "'lingam' or 'hybrid_nonlinear'.")

        self.skeleton = None
        self.dag = None
        self.data = None

    def run_generation_procedure(self):
        """ generate DAG and simulate data

        Write down some descriptions here.

        """
        self._clear()
        self._generate_dag()
        self._generate_data()

        return self

    def unpack(self):
        """ Return ndarray of ``dag`` and ndarray of ``data``.
        """
        return self.dag, self.data

    def _clear(self):
        self.skeleton = None
        self.dag = None
        self.data = None

        return self

    # ### CORRESPONDING TEST ##################################################
    # test_generation.py > test_dag_generation()

    # ### AUXILIARY COMPONENT(S) ##############################################
    # get_undigraph()
    # orient()
    def _generate_dag(self, sparsity=0.3):
        """randomly / ground-truth of DAG in addition to associated skeleton

        Write down some descriptions here.

        Arguments
        ----------
        sparsity : int (parameter)

        Return
        ------
        self : object
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

    # ### CORRESPONDING TEST ##################################################
    # test_generation.py > test_data_generation()

    # ### AUXILIARY COMPONENT(S) ##############################################
    # _add_random_noise()
    # _simulate_causal_model()
    def _generate_data(self):
        """structure identifiability / restrict function classes of SCMs

        Write down some descriptions here.

        Arguments
        ---------
        self.sample : int (parameter)
        self.graph_node_num : int (parameter)
        self.dag : ndarray (attribute)

        Returns
        -------
        self.data : ndarray (update)
        """
        self.data = np.zeros([self.sample, self.graph_node_num])

        dag_nx = nx.DiGraph(self.dag.T)
        topo_order = list(nx.topological_sort(dag_nx))

        for child_index, child_var in enumerate(topo_order):
            parent_vars = list(dag_nx.predecessors(child_var))
            parent_indexes = [topo_order.index(var) for var in parent_vars]

            self._add_random_noise(var_index=child_index)

            if len(parent_vars) > 0:
                for parent_index in parent_indexes:
                    self._simulate_causal_model(
                        child_index=child_index,
                        parent_index=parent_index
                    )

        self.data = self.data / np.std(self.data, axis=0)

        return self

    # ### SUBORDINATE COMPONENT(S) ############################################
    # _generate_dag()
    @staticmethod
    def _get_undigraph(graph_node_num, sparsity=0.3):
        """ER algorithm / implementation by built-in function in networkx.

        Write down some descriptions here.

        Parameters
        ----------
        parameter-1 : object
            - description-1
            - description-2

        sparsity : float
            - causal graph tends to be sparse in reality
            - common setting 0.3

        Return
        ------
        undigraph_np : numpy array
        """

        undigraph_nx = nx.random_graphs.erdos_renyi_graph(
            n=graph_node_num,
            p=sparsity
        )
        undigraph_np = nx.to_numpy_array(undigraph_nx)

        return undigraph_np

    # ### SUBORDINATE COMPONENT(S) ############################################
    # _generate_dag()
    @staticmethod
    def _orient(undigraph):
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

    # ### SUBORDINATE COMPONENT(S) ############################################
    # _generate_data()
    def _simulate_causal_model(self, child_index, parent_index):
        """ Custom-built simulation, hybrid non-linear function and LiNGAM

        Write down some descriptions here.

        Parameters
        ----------
        child_index : int

        parent_index : int

        Return
        ------
        self : self
             update self.data
        """
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

    # ### SUBORDINATE COMPONENT(S) ############################################
    # _generate_data()
    def _add_random_noise(self, var_index):
        """ Simulate exogenous variables or additive random perturbation.

        Write down some descriptions here.

        Parameters
        ----------
        var_index : int

        Return
        ------
        self : self
             update self.data
        """
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


class GeneralSkeleton(object):
    """
    Skeleton ground-truth with latent confouders

    Parameters
    ----------
    parameter-1 : object
        - description-1
        - description-2

    Attributes
    ----------
    attribute-1 : object
        - description-1
        - description-2

    Examples
    --------
    >>>
    >>>
    Notes
    -----
    * search "redundant edge" / d-connection
    """

    @staticmethod
    def get_general_skeleton(self):
        pass

    @staticmethod
    def print_info(self):
        pass

# def simulate_toy_scm(multivariate, model, sample, seed):
#     """
#     Returns
#     -------
#     data : numpy array
#         shape (n, 2) or (n, d)
#     ground_truth : numpy array
#         shape (2, 2) or (d, d)
#     """
#
#     # SCRATCHES ############################################
#     # - linear: (noise: Non-Gaussian) / LiNGAM / paper setup
#     #   - univariate: y := bx + e
#     #   - multivariate: y := BX + E
#     #
#     # - non-linear: (noise: Gaussian) / ANM / paper setup
#     #   - univariate: y := x ** 2 + e
#     #   - multivariate: y := x ** 3 + sin(x) + e
#     #
#     # - CAM-UV generation (model)
#     #
#     # - general nonlinear generation
#     # ######################################################
#
#     def fix_case_lingam(sample):
#         pass
#
#     def fix_case_anm(sample):
#         pass
#
#     # def fix_case_causal_model():
#     #     # TODO (Optional): Add any other generation
#     #     pass
#
#     return fix_case_lingam(sample) if model == 'LiNGAM' else fix_case_anm(sample)

